import enum
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone
from filelock import FileLock
from filelock import ReadWriteLock
from filelock import Timeout

from documents.models import Document
from documents.models import PaperlessTask
from documents.utils import IterWrapper
from documents.utils import QuerySetStream
from documents.utils import identity
from documents.versioning import annotate_effective_content
from documents.versioning import root_document_ids
from paperless.config import AIConfig
from paperless_ai.db import db_connection_released
from paperless_ai.embedding import build_llm_index_text
from paperless_ai.embedding import get_configured_model_name
from paperless_ai.embedding import get_embedding_model
from paperless_ai.exceptions import EmbeddingCapacityDeferred
from paperless_ai.exceptions import EmbeddingRateLimitError

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode
    from llama_index.core.schema import NodeWithScore

    from paperless_ai.vector_store import PaperlessSqliteVecVectorStore


logger = logging.getLogger("paperless_ai.indexing")

RAG_NUM_OUTPUT = 512
RAG_CHUNK_OVERLAP = 200

# update_llm_index(): row count per .iterator() batch when streaming
# documents for a rebuild/update via QuerySetStream, matching
# _DocumentViewerStream's chunk size in documents/search/_backend.py.
_INDEX_STREAM_CHUNK_SIZE = 1000


def queue_llm_index_update_if_needed(*, rebuild: bool, reason: str) -> bool:
    # NOTE: The check-then-enqueue sequence below is non-atomic (TOCTOU): two
    # concurrent workers can both observe no running task and both enqueue a
    # full rebuild. This is wasteful but not data-corrupting — update_llm_index
    # is itself protected by settings.LLM_INDEX_LOCK, so only one rebuild runs at a
    # time and the second one is serialised after the first completes.
    from documents.tasks import llmindex_index

    has_running = PaperlessTask.objects.filter(
        task_type=PaperlessTask.TaskType.LLM_INDEX,
        status__in=[PaperlessTask.Status.PENDING, PaperlessTask.Status.STARTED],
    ).exists()
    has_recent = PaperlessTask.objects.filter(
        task_type=PaperlessTask.TaskType.LLM_INDEX,
        date_created__gte=(timezone.now() - timedelta(minutes=5)),
    ).exists()
    if has_running or has_recent:
        return False

    llmindex_index.apply_async(
        kwargs={"rebuild": rebuild},
        headers={"trigger_source": PaperlessTask.TriggerSource.SYSTEM},
    )
    logger.warning(
        "Queued LLM index update%s: %s",
        " (rebuild)" if rebuild else "",
        reason,
    )
    return True


def get_vector_store() -> "PaperlessSqliteVecVectorStore":
    from paperless_ai.vector_store import PaperlessSqliteVecVectorStore

    settings.LLM_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return PaperlessSqliteVecVectorStore(
        uri=str(settings.LLM_INDEX_DIR),
    )


# --- LLM index locking ---------------------------------------------------
#
# Two locks guard the index; they answer different questions and are NOT
# interchangeable:
#
# * settings.LLM_INDEX_LOCK (FileLock, exclusive) - serializes WRITERS against
#   each other, so only one rebuild/upsert/delete/compaction runs at a time.
#   Taken by write_store(). Readers never take it, so it never blocks reads.
#
# * settings.LLM_INDEX_RWLOCK (ReadWriteLock) - coordinates readers against the
#   compaction/migration file swap. read_store() takes it SHARED (readers run
#   concurrently); _exclude_readers() takes it EXCLUSIVE, only for the swap, so
#   the database file is never replaced while a reader connection is open (that
#   would alias the old WAL onto the new file and corrupt it).
#
#                    | vs another writer | vs a reader
#   -----------------+-------------------+----------------------------
#   normal write     | LLM_INDEX_LOCK    | nothing (WAL gives MVCC)
#   compaction/swap  | LLM_INDEX_LOCK    | LLM_INDEX_RWLOCK (exclusive)
#   reader           | nothing (WAL)     | LLM_INDEX_RWLOCK (shared)
#
# They can't be merged into one ReadWriteLock: a normal write must exclude other
# writers WITHOUT blocking readers (WAL already gives reader/writer concurrency),
# and ReadWriteLock has no "exclusive vs writers, shared vs readers" mode. Only
# the swap needs to exclude readers.
def _index_rwlock() -> ReadWriteLock:
    """Return a fresh read/write lock instance for the index swap.

    ``is_singleton=False`` so reads and the swap always coordinate through
    SQLite (the actual cross-process case) rather than hitting the in-process
    reentrant-upgrade guard; callers must ``close()`` it (the context managers
    below do).
    """
    settings.LLM_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return ReadWriteLock(str(settings.LLM_INDEX_RWLOCK), is_singleton=False)


@contextmanager
def read_store():
    """Acquire the shared read lock and yield the vector store for a read.

    The shared lock is held for the whole lifetime of the connection (and
    closed on exit) so the compaction/migration swap, which takes the exclusive
    lock, never runs while this connection is open. Concurrent readers do not
    block each other; only the swap does.
    """
    lock = _index_rwlock()
    try:
        with lock.read_lock(), get_vector_store() as store:
            yield store
    finally:
        lock.close()


@contextmanager
def _exclude_readers():
    """Acquire exclusive index access, blocking until readers have drained.

    The exclusive counterpart to ``read_store()``: a compaction or migration
    must not run while any reader connection is open. Raises
    :class:`filelock.Timeout` if active readers do not drain within
    ``LLM_INDEX_COMPACTION_LOCK_TIMEOUT``; callers skip the operation on timeout.
    """
    lock = _index_rwlock()
    try:
        with lock.write_lock(timeout=settings.LLM_INDEX_COMPACTION_LOCK_TIMEOUT):
            yield
    finally:
        lock.close()


def _with_exclusive_access(operation: str, fn):
    """Run ``fn()`` with exclusive index access (see ``_exclude_readers()``),
    for compaction/migration file swaps that must not run while readers are
    active. Returns ``fn()``'s result, or None (after logging) if active
    readers do not drain within ``LLM_INDEX_COMPACTION_LOCK_TIMEOUT`` --
    callers skip the operation this run; it retries next time.
    """
    try:
        with _exclude_readers():
            return fn()
    except Timeout:
        logger.info(
            "Skipping LLM index %s: index readers are active; will retry next run.",
            operation,
        )
        return None


@contextmanager
def write_store(embed_model_name: str | None = None):
    """Acquire the write lock and yield the vector store.

    All mutating operations (upsert, delete, rebuild, compact) must go through
    this context manager to serialise concurrent Celery writers.
    Read paths use ``read_store()`` so they hold the shared read lock.

    Pass ``embed_model_name`` whenever the operation may create the table so
    the model name is recorded in the schema metadata for future mismatch checks.
    """
    from paperless_ai.vector_store import PaperlessSqliteVecVectorStore

    settings.LLM_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(settings.LLM_INDEX_LOCK),
        PaperlessSqliteVecVectorStore(
            uri=str(settings.LLM_INDEX_DIR),
            embed_model_name=embed_model_name,
        ) as store,
    ):
        yield store


class MigrationCheckResult(enum.Enum):
    """Outcome of _check_and_run_migrations().

    CURRENT: no migration was pending, or a pending structural migration
    was applied successfully - safe to write.

    REEMBED_REQUIRED: a pending migration needs fresh embeddings, which is
    never triggered automatically - the caller must force a rebuild.

    DEFERRED: a migration was pending but could not run because active
    index readers did not drain within LLM_INDEX_COMPACTION_LOCK_TIMEOUT --
    the store is still on its old schema. Callers must NOT proceed to
    write: collapsing this into the same falsy value as CURRENT (as a
    plain bool return once did) would let a write proceed against an
    unmigrated schema.
    """

    CURRENT = "current"
    REEMBED_REQUIRED = "reembed_required"
    DEFERRED = "deferred"


def _check_and_run_migrations(
    store: "PaperlessSqliteVecVectorStore",
) -> MigrationCheckResult:
    """Run any pending structural migrations, reporting the outcome as a
    tri-state result. Safe to call before any write, including
    delete()/upsert_document(): has_pending_migration() (see its docstring)
    keeps this a no-op, with no exclusive access taken, once the store is
    current.
    """
    if not store.has_pending_migration():
        return MigrationCheckResult.CURRENT
    result = _with_exclusive_access(
        "migration check",
        store.check_and_run_migrations,
    )
    if result is None:
        return MigrationCheckResult.DEFERRED
    return (
        MigrationCheckResult.REEMBED_REQUIRED
        if result
        else MigrationCheckResult.CURRENT
    )


def _safe_related_name(document: Document, field: str) -> str | None:
    """
    Returns the ``name`` of a related object (correspondent, document_type,
    storage_path), or None if the FK is unset or points at a row that has
    since been deleted (e.g. concurrently with this call).
    """
    try:
        related = getattr(document, field)
    except ObjectDoesNotExist:
        return None
    return related.name if related else None


def build_document_node(
    document: Document,
    *,
    chunk_size: int | None = None,
) -> list["BaseNode"]:
    """
    Given a Document, returns parsed Nodes ready for indexing.
    """
    text = build_llm_index_text(document)
    metadata = {
        "document_id": str(document.id),
        "title": document.title,
        "tags": [t.name for t in document.tags.all()],
        "correspondent": _safe_related_name(document, "correspondent"),
        "document_type": _safe_related_name(document, "document_type"),
        "filename": document.filename,
        "storage_path": _safe_related_name(document, "storage_path"),
        "archive_serial_number": document.archive_serial_number,
        "created": document.created.isoformat() if document.created else None,
        "added": document.added.isoformat() if document.added else None,
        "modified": document.modified.isoformat(),
    }
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import SimpleNodeParser

    # Exclude all metadata keys from the embedding text — build_llm_index_text
    # already encodes this info in the body, so prepending it again would double
    # the token count and exceed embedding models with small context windows
    # (e.g. nomic-embed-text via Ollama defaults to num_ctx=2048).
    doc = LlamaDocument(
        id_=str(document.id),
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata.keys()),
        excluded_llm_metadata_keys=["document_id"],
    )
    chunk_size = chunk_size or get_rag_chunk_size()
    parser = SimpleNodeParser(
        chunk_size=chunk_size,
        chunk_overlap=get_rag_chunk_overlap(chunk_size),
    )
    return parser.get_nodes_from_documents([doc])


def load_or_build_index(config: AIConfig, store: "PaperlessSqliteVecVectorStore"):
    """Return a VectorStoreIndex backed by ``store``.

    ``store`` is supplied by the caller's ``read_store()`` context so the shared
    read lock and the connection stay alive for the whole retrieval.
    """
    import llama_index.core.settings as llama_settings
    from llama_index.core import VectorStoreIndex

    embed_model = get_embedding_model(config)
    llama_settings.Settings.embed_model = embed_model
    return VectorStoreIndex.from_vector_store(
        vector_store=store,
        embed_model=embed_model,
    )


def llm_index_exists() -> bool:
    """True when the index table exists on disk."""
    with read_store() as store:
        return store.table_exists()


def get_rag_chunk_size() -> int:
    return AIConfig().llm_embedding_chunk_size


def get_rag_chunk_overlap(chunk_size: int | None = None) -> int:
    chunk_size = chunk_size or get_rag_chunk_size()
    return min(RAG_CHUNK_OVERLAP, chunk_size - 1)


def get_rag_prompt_helper(
    *,
    chunk_size: int | None = None,
    context_size: int | None = None,
):
    from llama_index.core.indices.prompt_helper import PromptHelper

    if chunk_size is None or context_size is None:
        config = AIConfig()
        chunk_size = chunk_size or config.llm_embedding_chunk_size
        context_size = context_size or config.llm_context_size

    return PromptHelper(
        context_window=context_size,
        num_output=RAG_NUM_OUTPUT,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=chunk_size,
    )


_RETRY_AFTER_RE = re.compile(r"retry-after(?:-ms)?\D+(\d+)", re.IGNORECASE)
_EMBEDDING_MIN_INTERVAL_SECONDS = 7.5
_EMBEDDING_TPM_LIMIT = 24_000
_EMBEDDING_TPM_WINDOW_SECONDS = 60.0
_EMBEDDING_CHECKPOINT_TTL = timedelta(days=7)
_EMBEDDING_CHECKPOINT_CLEANUP_INTERVAL = timedelta(hours=1)


def _embedding_pacing_path(model_name: str) -> Path:
    checkpoint_dir = Path(settings.DATA_DIR) / "embedding-checkpoints"
    checkpoint_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(model_name.encode()).hexdigest()
    return checkpoint_dir / f".{digest}.pacing.json"


def _write_embedding_pacing(path: Path, value: dict[str, object]) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(value, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        Path(temporary).unlink(missing_ok=True)
        raise


def _reserve_embedding_capacity(model_name: str, texts: list[str]) -> None:
    """Reserve a provider request or defer without sleeping under the DB lock."""
    request_tokens = _estimate_embedding_tokens(texts)
    if request_tokens > _EMBEDDING_TPM_LIMIT:
        raise ValueError(
            "A single embedding batch exceeds the metered provider token limit; "
            "reduce the configured chunk size before indexing",
        )
    path = _embedding_pacing_path(model_name)
    lock = FileLock(str(path) + ".lock")
    try:
        with lock.acquire(timeout=0):
            now = time.time()
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, ValueError):
                state = {}
            if not isinstance(state, dict):
                state = {}
            events = []
            for event in state.get("events", []):
                try:
                    timestamp = float(event[0])
                    tokens = int(event[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if (
                    math.isfinite(timestamp)
                    and tokens > 0
                    and now - timestamp < _EMBEDDING_TPM_WINDOW_SECONDS
                ):
                    events.append([timestamp, tokens])
            try:
                last_request = float(state.get("last_request", 0))
            except (TypeError, ValueError):
                last_request = 0.0
            if not math.isfinite(last_request):
                last_request = 0.0
            wait = max(
                0.0,
                _EMBEDDING_MIN_INTERVAL_SECONDS - (now - last_request),
            )
            used_tokens = sum(event[1] for event in events)
            if used_tokens + request_tokens > _EMBEDDING_TPM_LIMIT and events:
                wait = max(
                    wait,
                    _EMBEDDING_TPM_WINDOW_SECONDS - (now - events[0][0]),
                )
            if wait > 0:
                raise EmbeddingCapacityDeferred(wait)
            events.append([now, request_tokens])
            _write_embedding_pacing(path, {"last_request": now, "events": events})
    except Timeout as exc:
        raise EmbeddingCapacityDeferred(1.0) from exc


def _estimate_embedding_tokens(texts: list[str]) -> int:
    """Estimate provider tokens using the same conservative pacing formula."""
    return max(1, math.ceil(sum(len(text.encode("utf-8")) for text in texts) / 3))


def _embedding_batch_slices(
    texts: list[str],
    batch_size: int,
    *,
    metered: bool,
) -> Iterable[list[str]]:
    """Yield deterministic provider batches within the metered token limit.

    Local embedding backends retain their configured batch size. Metered
    OpenAI-compatible backends additionally split batches before the pacing
    reservation so a large document cannot be deferred forever as an
    impossible request.
    """
    batch_size = max(1, batch_size)
    if not metered:
        for start in range(0, len(texts), batch_size):
            yield texts[start : start + batch_size]
        return

    start = 0
    while start < len(texts):
        end = min(start + batch_size, len(texts))
        batch: list[str] = []
        tokens = 0
        while start + len(batch) < end:
            text = texts[start + len(batch)]
            text_tokens = _estimate_embedding_tokens([text])
            if text_tokens > _EMBEDDING_TPM_LIMIT:
                raise ValueError(
                    "A single embedding chunk exceeds the metered provider token "
                    "limit; reduce the configured chunk size before indexing",
                )
            if batch and tokens + text_tokens > _EMBEDDING_TPM_LIMIT:
                break
            batch.append(text)
            tokens += text_tokens
        yield batch
        start += len(batch)


def _embedding_requires_pacing(
    embed_model: object,
    model_name: str | None = None,
) -> bool:
    """Return whether this backend is subject to remote provider quotas.

    Local HuggingFace and Ollama embeddings have no upstream RPM/TPM quota;
    pacing them only slows indexing and makes local rebuilds fail under the
    deferred retry path. OpenAI-compatible embeddings are metered by default;
    explicitly allowlisted local aliases bypass the cross-worker reservation.
    """
    module = type(embed_model).__module__.lower()
    if "openai" not in module:
        return False
    model_name = model_name or getattr(embed_model, "model_name", "")
    unmetered_models = {
        alias.strip()
        for alias in os.environ.get(
            "PAPERLESS_AI_UNMETERED_EMBEDDING_MODELS",
            "",
        ).split(",")
        if alias.strip()
    }
    return model_name not in unmetered_models


def _embedding_retry_delay(exc: Exception) -> float | None:
    """Return the provider's transient retry delay, if any."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status is None:
        match = re.search(r"\b(429|502|503|504)\b", str(exc))
        status = int(match.group(1)) if match else None
    if status not in (429, 502, 503, 504):
        return None
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
                return max(1.0, value)
            except (TypeError, ValueError):
                try:
                    return max(
                        1.0,
                        parsedate_to_datetime(raw).timestamp() - time.time(),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        value = float(match.group(1))
        if not math.isfinite(value):
            return 1.0
        if "-ms" in match.group(0).lower():
            value /= 1000
        return max(1.0, value)
    return 1.0


def _embedding_checkpoint_path(model_name: str, batch: list[str]) -> Path:
    checkpoint_dir = Path(settings.DATA_DIR) / "embedding-checkpoints"
    checkpoint_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps([model_name, batch], ensure_ascii=False).encode(),
    ).hexdigest()
    return checkpoint_dir / f"{digest}.json"


def _read_embedding_checkpoint(
    path: Path,
    expected: int,
) -> list[list[float]] | None:
    try:
        values = json.loads(path.read_text())
    except (FileNotFoundError, OSError, ValueError):
        return None
    if (
        not isinstance(values, list)
        or len(values) != expected
        or not all(isinstance(value, list) for value in values)
    ):
        return None
    if not values or not values[0]:
        return None
    dimension = len(values[0])
    if any(
        len(value) != dimension
        or any(
            not isinstance(component, (int, float)) or not math.isfinite(component)
            for component in value
        )
        for value in values
    ):
        return None
    return values


def _write_embedding_checkpoint(path: Path, values: list[list[float]]) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        os.fchmod(fd, 0o600)
        handle = os.fdopen(fd, "w")
        fd = -1  # ownership transferred to handle
        with handle:
            json.dump(values, handle, separators=(",", ":"))
        Path(temporary).replace(path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            Path(temporary).unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        raise


def _prune_embedding_checkpoints(checkpoint_dir: Path) -> None:
    marker = checkpoint_dir / ".cleanup"
    now = timezone.now()
    try:
        if (
            now
            - datetime.fromtimestamp(
                marker.stat().st_mtime,
                tz=timezone.get_current_timezone(),
            )
            < _EMBEDDING_CHECKPOINT_CLEANUP_INTERVAL
        ):
            return
    except FileNotFoundError:
        pass
    marker.touch(mode=0o600, exist_ok=True)
    cutoff = now - _EMBEDDING_CHECKPOINT_TTL
    for checkpoint in checkpoint_dir.glob("*.json"):
        try:
            modified = datetime.fromtimestamp(
                checkpoint.stat().st_mtime,
                tz=timezone.get_current_timezone(),
            )
            if modified < cutoff:
                checkpoint.unlink()
        except FileNotFoundError:
            continue


def _embed_nodes(
    nodes: list["BaseNode"],
    embed_model,
    *,
    model_name: str | None = None,
) -> None:
    """Embed ``nodes`` in place using ``embed_model``."""
    if not nodes:
        return
    from llama_index.core.schema import MetadataMode

    texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    batch_size = getattr(embed_model, "embed_batch_size", None) or len(texts)
    embeddings = []
    model_name = model_name or getattr(
        embed_model,
        "model_name",
        type(embed_model).__name__,
    )
    checkpoint_dir = Path(settings.DATA_DIR) / "embedding-checkpoints"
    try:
        checkpoint_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _prune_embedding_checkpoints(checkpoint_dir)
    except OSError:
        # Checkpointing is an optimization. A read-only or unavailable
        # scratch directory must not turn a successful provider response into
        # a failed embedding task.
        checkpoint_dir = None
    checkpoint_paths = []
    for batch in _embedding_batch_slices(
        texts,
        batch_size,
        metered=_embedding_requires_pacing(embed_model, model_name),
    ):
        if checkpoint_dir is not None:
            try:
                checkpoint = _embedding_checkpoint_path(model_name, batch)
            except OSError:
                checkpoint = None
        else:
            checkpoint = None
        if checkpoint is not None:
            checkpoint_paths.append(checkpoint)
        cached = (
            _read_embedding_checkpoint(checkpoint, len(batch))
            if checkpoint is not None
            else None
        )
        if cached is not None:
            embeddings.extend(cached)
            continue
        try:
            if _embedding_requires_pacing(embed_model, model_name):
                _reserve_embedding_capacity(model_name, batch)
            batch_embeddings = embed_model.get_text_embedding_batch(batch)
        except Exception as exc:
            delay = _embedding_retry_delay(exc)
            if delay is None:
                raise
            # Never sleep while the caller holds the vector-store write lock.
            # The Celery owner defers the whole embedding task for the exact
            # provider window, and completed documents remain persisted.
            raise EmbeddingRateLimitError(delay) from exc
        if len(batch_embeddings) != len(batch):
            raise ValueError(
                "Embedding provider returned a different number of vectors than inputs",
            )
        embeddings.extend(batch_embeddings)
        if checkpoint is not None:
            try:
                _write_embedding_checkpoint(checkpoint, batch_embeddings)
            except (OSError, TypeError, ValueError):
                logger.warning("Could not persist embedding checkpoint", exc_info=True)

    for node, emb in zip(nodes, embeddings, strict=True):
        node.embedding = emb
    for checkpoint in checkpoint_paths:
        try:
            checkpoint.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove embedding checkpoint", exc_info=True)


def document_id_filters(doc_ids):
    """Return a MetadataFilters IN filter scoped to ``doc_ids``."""
    from llama_index.core.vector_stores.types import FilterOperator
    from llama_index.core.vector_stores.types import MetadataFilter
    from llama_index.core.vector_stores.types import MetadataFilters

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="document_id",
                operator=FilterOperator.IN,
                value=list(doc_ids),
            ),
        ],
    )


def _exclude_document_id_filter(document_id: int | str):
    """Return a MetadataFilters NE filter excluding ``document_id``."""
    from llama_index.core.vector_stores.types import FilterOperator
    from llama_index.core.vector_stores.types import MetadataFilter
    from llama_index.core.vector_stores.types import MetadataFilters

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="document_id",
                operator=FilterOperator.NE,
                value=str(document_id),
            ),
        ],
    )


def exclude_document_ids_filter(doc_ids):
    """Return a MetadataFilters NIN filter excluding every id in ``doc_ids``."""
    from llama_index.core.vector_stores.types import FilterOperator
    from llama_index.core.vector_stores.types import MetadataFilter
    from llama_index.core.vector_stores.types import MetadataFilters

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="document_id",
                operator=FilterOperator.NIN,
                value=list(doc_ids),
            ),
        ],
    )


def update_llm_index(
    *,
    iter_wrapper: IterWrapper[Document] = identity,
    rebuild=False,
    document_ids: Iterable[int] | None = None,
) -> str:
    """Rebuild or incrementally update the LLM index.

    ``document_ids``, when given, scopes an incremental update to just those
    documents instead of scanning the whole library - callers that already
    know which documents changed (e.g. a bulk edit) should pass this to avoid
    an O(library size) scan per call. Ignored whenever a rebuild actually
    happens, since a rebuild always covers the whole library regardless.
    """
    with write_store() as store:
        migration_result = _check_and_run_migrations(store)
        if migration_result is MigrationCheckResult.REEMBED_REQUIRED:
            logger.warning(
                "LLM index migration requires re-embedding; forcing rebuild.",
            )
            rebuild = True
        elif migration_result is MigrationCheckResult.DEFERRED:
            logger.info(
                "Skipping LLM index update: migration check deferred while "
                "index readers are active; will retry next run.",
            )
            return (
                "Skipping LLM index update: migration check deferred; "
                "will retry next run."
            )
    documents = annotate_effective_content(
        Document.objects.filter(root_document__isnull=True)
        .select_related("correspondent", "document_type", "storage_path")
        .prefetch_related("tags", "notes", "custom_fields__field"),
    )
    no_documents = not documents.exists()

    # Fast exit before touching config: nothing to index and no existing index.
    if no_documents and not rebuild and not llm_index_exists():
        logger.warning("No documents found to index.")
        return "No documents found to index."

    config = AIConfig()
    model_name = get_configured_model_name(config)

    # A deferred retry must still verify the stored model identity. Never mix
    # vectors if configuration changes while a provider window is pending.
    if not rebuild:
        with read_store() as store:
            if store.table_exists() and store.config_mismatch(model_name):
                logger.warning("Embedding model changed; forcing LLM index rebuild.")
                rebuild = True

    if no_documents:
        logger.warning("No documents found to index.")

    chunk_size = config.llm_embedding_chunk_size
    embed_model = get_embedding_model(config)

    with write_store(embed_model_name=model_name) as store:
        if rebuild or not store.table_exists():
            logger.info("Rebuilding LLM index.")
            store.drop_table()
            for document in iter_wrapper(
                QuerySetStream(documents, chunk_size=_INDEX_STREAM_CHUNK_SIZE),
            ):
                nodes = build_document_node(document, chunk_size=chunk_size)
                try:
                    _embed_nodes(nodes, embed_model, model_name=model_name)
                except EmbeddingRateLimitError as exc:
                    # A scoped request can have been promoted to a full
                    # rebuild by model/migration checks. Tell the Celery
                    # owner to resume the whole rebuild after the wait.
                    exc.full_rebuild = True
                    raise
                store.add(nodes)
            msg = "LLM index rebuilt successfully."
        else:
            scoped_documents = (
                documents.filter(id__in=root_document_ids(document_ids))
                if document_ids is not None
                else documents
            )
            existing = store.get_modified_times()
            changed = 0
            for document in iter_wrapper(
                QuerySetStream(scoped_documents, chunk_size=_INDEX_STREAM_CHUNK_SIZE),
            ):
                doc_id = str(document.id)
                if existing.get(doc_id) == document.modified.isoformat():
                    continue
                nodes = build_document_node(document, chunk_size=chunk_size)
                try:
                    _embed_nodes(nodes, embed_model, model_name=model_name)
                except EmbeddingRateLimitError as exc:
                    exc.full_rebuild = False
                    raise
                store.upsert_document(doc_id, nodes)
                changed += 1
            msg = (
                "LLM index updated successfully."
                if changed
                else "No changes detected in LLM index."
            )

        _with_exclusive_access("compaction", store.compact)
    return msg


def llm_index_add_or_update_document(document: Document):
    """
    Add or atomically replace a document's chunks in the index. Only root
    documents are indexed, so a version is indexed as its root document.
    """
    if document.root_document_id is not None:
        document = document.root_document
    config = AIConfig()
    new_nodes = build_document_node(
        document,
        chunk_size=config.llm_embedding_chunk_size,
    )
    if new_nodes:
        _embed_nodes(
            new_nodes,
            get_embedding_model(config),
            model_name=get_configured_model_name(config),
        )

    with write_store(embed_model_name=get_configured_model_name(config)) as store:
        migration_result = _check_and_run_migrations(store)
        if migration_result is MigrationCheckResult.REEMBED_REQUIRED:
            logger.warning(
                "Skipping incremental LLM index update for document %s: the "
                "index requires re-embedding first. Run 'document_llmindex "
                "rebuild' to resolve.",
                document.id,
            )
            return
        if migration_result is MigrationCheckResult.DEFERRED:
            logger.info(
                "Skipping incremental LLM index update for document %s: "
                "migration check deferred while index readers are active; "
                "will retry on the next write.",
                document.id,
            )
            return
        store.upsert_document(str(document.id), new_nodes)


def llm_index_migrate() -> None:
    """Apply any pending LLM index schema migrations, with no reindex.

    Intended to run unconditionally on every startup (see the
    init-llmindex-migrate container step and the bare-metal upgrade docs):
    has_pending_migration() short-circuits to a metadata-only read once the
    store is current, so a healthy install pays almost nothing here. Only
    ever applies structural migrations - a pending re-embed migration is
    left for the explicit, deliberate rebuild path (``document_llmindex
    update``/``rebuild``) to resolve, since re-embedding can be slow and,
    for a metered embedding backend, cost money.
    """
    if not AIConfig().llm_index_enabled:
        return
    with write_store() as store:
        migration_result = _check_and_run_migrations(store)
    if migration_result is MigrationCheckResult.REEMBED_REQUIRED:
        logger.warning(
            "LLM index requires re-embedding, which this automatic migration "
            "check will not do on its own - it can be slow and, for a "
            "metered embedding backend, cost money. Run "
            "'document_llmindex rebuild' manually when ready.",
        )
    elif migration_result is MigrationCheckResult.DEFERRED:
        logger.info(
            "LLM index migration check deferred while index readers are "
            "active; will retry next run.",
        )


def llm_index_compact() -> None:
    """Compact the index immediately, rebuilding the table to reclaim space."""
    with write_store() as store:
        _with_exclusive_access("compaction", lambda: store.compact(force=True))


def llm_index_remove_document(document: Document):
    """Remove a document's chunks from the LLM index."""
    with write_store() as store:
        migration_result = _check_and_run_migrations(store)
        if migration_result is MigrationCheckResult.REEMBED_REQUIRED:
            logger.warning(
                "Skipping removal of document %s from the LLM index: the "
                "index requires re-embedding first. Run 'document_llmindex "
                "rebuild' to resolve.",
                document.id,
            )
            return
        if migration_result is MigrationCheckResult.DEFERRED:
            logger.info(
                "Skipping removal of document %s from the LLM index: "
                "migration check deferred while index readers are active; "
                "will retry on the next write.",
                document.id,
            )
            return
        store.delete(str(document.id))


def truncate_content(
    content: str,
    *,
    chunk_size: int | None = None,
    context_size: int | None = None,
) -> str:
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.text_splitter import TokenTextSplitter

    if chunk_size is None or context_size is None:
        config = AIConfig()
        chunk_size = chunk_size or config.llm_embedding_chunk_size
        context_size = context_size or config.llm_context_size
    prompt_helper = get_rag_prompt_helper(
        chunk_size=chunk_size,
        context_size=context_size,
    )
    splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=get_rag_chunk_overlap(chunk_size),
    )
    content_chunks = splitter.split_text(content)
    truncated_chunks = prompt_helper.truncate(
        prompt=PromptTemplate(template="{content}"),
        text_chunks=content_chunks,
        padding=5,
    )
    return " ".join(truncated_chunks)


def truncate_embedding_query(content: str, *, chunk_size: int) -> str:
    from llama_index.core.text_splitter import TokenTextSplitter

    splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    content_chunks = splitter.split_text(content)
    return content_chunks[0] if content_chunks else ""


def normalize_document_ids(document_ids: Iterable[int | str] | None) -> set[str] | None:
    if document_ids is None:
        return None
    return {str(document_id) for document_id in document_ids}


def retrieve_similar_nodes(
    document: Document,
    top_k: int = 5,
    document_ids: Iterable[int | str] | None = None,
) -> list["NodeWithScore"]:
    """Run the vector-store retrieval once and return the raw scored nodes,
    permission-filtered by document_ids and with the source document excluded.
    Callers derive both RAG text context and taxonomy candidates from this
    single retrieval instead of querying the vector store twice per request.
    """
    allowed_document_ids = normalize_document_ids(document_ids)
    if allowed_document_ids is not None and not allowed_document_ids:
        return []

    if not llm_index_exists():
        queue_llm_index_update_if_needed(
            rebuild=False,
            reason="LLM index not found for similarity query.",
        )
        return []

    config = AIConfig()

    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.vector_stores.types import FilterCondition
    from llama_index.core.vector_stores.types import MetadataFilters

    filter_parts = []
    if allowed_document_ids is not None:
        filter_parts.extend(document_id_filters(allowed_document_ids).filters)
    if document.pk is not None:
        filter_parts.extend(_exclude_document_id_filter(document.pk).filters)

    filters = (
        MetadataFilters(filters=filter_parts, condition=FilterCondition.AND)
        if filter_parts
        else None
    )

    query_text = truncate_embedding_query(
        (document.title or "") + "\n" + (document.get_effective_content() or ""),
        chunk_size=config.llm_embedding_chunk_size,
    )
    # Hold the shared read lock for the whole retrieval so the connection is
    # never open across a compaction swap. The retrieve() call generates a
    # query embedding (a slow external request) and searches the vector store;
    # no Django ORM access happens during it, so release the pooled DB
    # connection for its duration. See #12976.
    with read_store() as store:
        index = load_or_build_index(config, store)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
            filters=filters,
        )
        with db_connection_released():
            results = retriever.retrieve(query_text)

    if allowed_document_ids is None:
        return results

    filtered = []
    for node in results:
        document_id = node.metadata.get("document_id")
        if document_id is None:  # pragma: no cover
            # Every node the indexing pipeline builds always sets
            # document_id; this guards a malformed/partial vec0 row that
            # shouldn't occur given the current schema.
            continue
        if str(document_id) not in allowed_document_ids:
            continue
        filtered.append(node)
    return filtered
