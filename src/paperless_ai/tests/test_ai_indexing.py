import logging
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_mock
from django.db import connection
from django.db import transaction
from django.test import TestCase
from django.test import override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
from filelock import Timeout
from llama_index.core.schema import MetadataMode

from documents.models import Correspondent
from documents.models import CustomField
from documents.models import CustomFieldInstance
from documents.models import Document
from documents.models import DocumentType
from documents.models import Note
from documents.models import PaperlessTask
from documents.signals import document_consumption_finished
from documents.signals import document_updated
from documents.tasks import llmindex_index
from documents.tasks import update_document_in_llm_index
from paperless.models import ApplicationConfiguration
from paperless_ai import indexing
from paperless_ai.exceptions import EmbeddingCapacityDeferred
from paperless_ai.exceptions import EmbeddingRateLimitError
from paperless_ai.tests.conftest import FakeEmbedding
from paperless_ai.vector_store import PaperlessSqliteVecVectorStore
from paperless_testing.factories import DocumentFactory
from paperless_testing.factories import PaperlessTaskFactory


def test_metered_embedding_batches_respect_provider_token_limit() -> None:
    texts = ["a" * 40_000, "b" * 40_000, "c" * 40_000]

    batches = list(indexing._embedding_batch_slices(texts, 100, metered=True))

    assert [len(batch) for batch in batches] == [1, 1, 1]
    assert all(
        indexing._estimate_embedding_tokens(batch) <= indexing._EMBEDDING_TPM_LIMIT
        for batch in batches
    )


def test_metered_embedding_batch_rejects_oversized_single_chunk() -> None:
    with pytest.raises(ValueError, match="single embedding chunk"):
        list(
            indexing._embedding_batch_slices(
                ["x" * (indexing._EMBEDDING_TPM_LIMIT * 3 * 3 + 1)],
                100,
                metered=True,
            ),
        )


def test_local_embedding_batches_keep_configured_batch_size() -> None:
    texts = ["one", "two", "three"]

    batches = list(indexing._embedding_batch_slices(texts, 2, metered=False))

    assert batches == [["one", "two"], ["three"]]


def test_embed_nodes_splits_metered_provider_batches_before_reservation(
    tmp_path: Path,
) -> None:
    class Node:
        def __init__(self, text: str) -> None:
            self.text = text
            self.embedding = None

        def get_content(self, *, metadata_mode) -> str:
            return self.text

    nodes = [Node("a" * 40_000), Node("b" * 40_000), Node("c" * 40_000)]
    batches: list[list[str]] = []
    model = MagicMock(embed_batch_size=100)
    model.get_text_embedding_batch.side_effect = lambda batch: (
        batches.append(batch) or [[0.1] for _ in batch]
    )

    reservations: list[int] = []

    def reserve(_model_name: str, batch: list[str]) -> None:
        tokens = indexing._estimate_embedding_tokens(batch)
        reservations.append(tokens)
        assert tokens <= indexing._EMBEDDING_TPM_LIMIT

    with (
        patch.object(indexing.settings, "DATA_DIR", tmp_path),
        patch("paperless_ai.indexing._embedding_requires_pacing", return_value=True),
        patch("paperless_ai.indexing._reserve_embedding_capacity", side_effect=reserve),
    ):
        indexing._embed_nodes(nodes, model, model_name="fake-openai-embedding")

    assert [len(batch) for batch in batches] == [1, 1, 1]
    assert len(reservations) == 3
    assert all(node.embedding == [0.1] for node in nodes)


def test_reservation_rejects_impossible_metered_batch(tmp_path: Path) -> None:
    with patch.object(indexing.settings, "DATA_DIR", tmp_path):
        with pytest.raises(ValueError, match="single embedding batch"):
            indexing._reserve_embedding_capacity(
                "fake-openai-embedding",
                ["x" * (indexing._EMBEDDING_TPM_LIMIT * 3 + 1)],
            )


def test_reservation_deferral_is_distinct_from_provider_rate_limit(
    tmp_path: Path,
) -> None:
    with patch.object(indexing.settings, "DATA_DIR", tmp_path):
        pacing = indexing._embedding_pacing_path("fake-openai-embedding")
        pacing.parent.mkdir(parents=True, exist_ok=True)
        pacing.write_text('{"last_request": 9999999999, "events": []}')
        with pytest.raises(EmbeddingCapacityDeferred):
            indexing._reserve_embedding_capacity("fake-openai-embedding", ["text"])


@pytest.mark.django_db
def test_repeated_local_capacity_deferrals_do_not_consume_provider_attempts(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Local pacing must remain retryable beyond the provider retry budget."""

    class RetrySent(Exception):
        pass

    retry_calls: list[dict[str, object]] = []
    clock = [100.0]

    document = DocumentFactory()
    index_update = mocker.patch(
        "documents.tasks.llm_index_add_or_update_document",
        side_effect=[EmbeddingCapacityDeferred(1)] * 4 + [None],
    )
    mocker.patch("documents.tasks.time.time", side_effect=lambda: clock[0])

    def retry(**kwargs: object) -> None:
        retry_calls.append(kwargs)
        raise RetrySent

    mocker.patch.object(update_document_in_llm_index, "retry", side_effect=retry)

    retry_kwargs: dict[str, object] = {
        "document_id": document.pk,
        "provider_attempts": 0,
    }
    for _ in range(4):
        with pytest.raises(RetrySent):
            update_document_in_llm_index.run(**retry_kwargs)
        retry_kwargs = retry_calls[-1]["kwargs"]
        clock[0] = float(retry_kwargs["not_before"]) + 1

    update_document_in_llm_index.run(**retry_kwargs)

    assert len(retry_calls) == 4
    assert [call["kwargs"]["provider_attempts"] for call in retry_calls] == [0] * 4
    assert index_update.call_count == 5


@pytest.mark.django_db
@override_settings(AI_ENABLED=True, LLM_EMBEDDING_BACKEND="huggingface")
def test_bulk_local_capacity_deferrals_do_not_consume_provider_attempts(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Bulk indexing keeps retrying local pacing without provider attempts."""

    class RetrySent(Exception):
        pass

    retry_calls: list[dict[str, object]] = []
    clock = [100.0]
    promoted = EmbeddingCapacityDeferred(1)
    promoted.full_rebuild = True
    index_update = mocker.patch(
        "paperless_ai.indexing.update_llm_index",
        side_effect=[
            promoted,
            EmbeddingCapacityDeferred(1),
            EmbeddingCapacityDeferred(1),
            EmbeddingCapacityDeferred(1),
            None,
        ],
    )
    mocker.patch("documents.tasks.time.time", side_effect=lambda: clock[0])

    def retry(**kwargs: object) -> None:
        retry_calls.append(kwargs)
        raise RetrySent

    mocker.patch.object(llmindex_index, "retry", side_effect=retry)
    retry_kwargs: dict[str, object] = {
        "rebuild": False,
        "document_ids": [1, 2],
        "provider_attempts": 0,
    }
    for _ in range(4):
        with pytest.raises(RetrySent):
            llmindex_index.run(**retry_kwargs)
        retry_kwargs = retry_calls[-1]["kwargs"]
        clock[0] = float(retry_kwargs["not_before"]) + 1

    llmindex_index.run(**retry_kwargs)
    assert len(retry_calls) == 4
    assert [call["kwargs"]["provider_attempts"] for call in retry_calls] == [0] * 4
    assert retry_calls[0]["kwargs"]["document_ids"] is None
    assert index_update.call_count == 5


def test_metered_checkpoint_retry_only_reembeds_remaining_batches(
    tmp_path: Path,
) -> None:
    class Node:
        def __init__(self, text: str) -> None:
            self.text = text
            self.embedding = None

        def get_content(self, *, metadata_mode) -> str:
            return self.text

    nodes = [Node("a" * 40_000), Node("b" * 40_000), Node("c" * 40_000)]
    provider_calls: list[list[str]] = []

    def embed(batch: list[str]) -> list[list[float]]:
        provider_calls.append(batch)
        if len(provider_calls) == 2:
            raise EmbeddingRateLimitError(60)
        return [[float(len(provider_calls))] for _ in batch]

    reservations = 0

    def reserve(_model_name: str, batch: list[str]) -> None:
        nonlocal reservations
        reservations += 1
        assert (
            indexing._estimate_embedding_tokens(batch) <= indexing._EMBEDDING_TPM_LIMIT
        )

    model = MagicMock(embed_batch_size=100)
    model.get_text_embedding_batch.side_effect = embed

    with (
        patch.object(indexing.settings, "DATA_DIR", tmp_path),
        patch("paperless_ai.indexing._embedding_requires_pacing", return_value=True),
        patch("paperless_ai.indexing._reserve_embedding_capacity", side_effect=reserve),
    ):
        with pytest.raises(EmbeddingRateLimitError):
            indexing._embed_nodes(nodes, model, model_name="fake-openai-embedding")
        indexing._embed_nodes(nodes, model, model_name="fake-openai-embedding")

    assert provider_calls == [
        [nodes[0].text],
        [nodes[1].text],
        [nodes[1].text],
        [nodes[2].text],
    ]
    assert reservations == 4
    # The first successful batch is restored from its checkpoint; only the
    # failed batch and the batches after it are submitted again.
    assert [node.embedding for node in nodes] == [[1.0], [3.0], [4.0]]


@pytest.mark.parametrize(
    ("module", "model_name", "allowlist", "expected"),
    [
        ("llama_index.embeddings.openai_like", "qwen-local", "qwen-local", False),
        (
            "llama_index.embeddings.openai_like",
            "gemini-embedding-001",
            "qwen-local",
            True,
        ),
        ("llama_index.embeddings.openai_like", "gemini-embedding-001", None, True),
        ("llama_index.embeddings.openai_like", None, "qwen-local", True),
        ("llama_index.embeddings.openai_like", "gemini-embedding-001", " , , ", True),
        ("llama_index.embeddings.ollama", "qwen-local", "qwen-local", False),
        ("llama_index.embeddings.huggingface", "qwen-local", "qwen-local", False),
    ],
)
def test_embedding_pacing_allowlist_is_exact_and_openai_only(
    monkeypatch,
    module: str,
    model_name: str | None,
    allowlist: str | None,
    *,
    expected: bool,
) -> None:
    if allowlist is None:
        monkeypatch.delenv("PAPERLESS_AI_UNMETERED_EMBEDDING_MODELS", raising=False)
    else:
        monkeypatch.setenv("PAPERLESS_AI_UNMETERED_EMBEDDING_MODELS", allowlist)
    attrs = {"__module__": module}
    if model_name is not None:
        attrs["model_name"] = model_name
    embed_model = type("FakeEmbedding", (), attrs)()

    assert indexing._embedding_requires_pacing(embed_model) is expected


def test_embed_nodes_uses_configured_alias_for_local_openai_compatible_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PAPERLESS_AI_UNMETERED_EMBEDDING_MODELS", "qwen-local")

    class Node:
        embedding = None

        def get_content(self, *, metadata_mode) -> str:
            return "x" * (indexing._EMBEDDING_TPM_LIMIT * 3 + 1)

    class OpenAILikeFake:
        __module__ = "llama_index.embeddings.openai_like"
        model_name = "gemini-embedding-001"
        embed_batch_size = 1

        def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] for _ in texts]

    model = OpenAILikeFake()
    nodes = [Node()]
    with (
        patch.object(indexing.settings, "DATA_DIR", tmp_path),
        patch("paperless_ai.indexing._reserve_embedding_capacity") as reserve,
    ):
        indexing._embed_nodes(nodes, model, model_name="qwen-local")

    reserve.assert_not_called()
    assert nodes[0].embedding == [0.1]


@pytest.fixture
def real_document(db: None) -> Document:
    return Document.objects.create(
        title="Test Document",
        content="This is some test content.",
        added=timezone.now(),
    )


@pytest.mark.django_db
def test_build_document_node(real_document: Document) -> None:
    nodes = indexing.build_document_node(real_document)
    assert len(nodes) > 0
    assert nodes[0].metadata["document_id"] == str(real_document.id)
    assert nodes[0].metadata["filename"] == real_document.filename
    assert nodes[0].metadata["storage_path"] == (
        real_document.storage_path.name if real_document.storage_path else None
    )
    assert (
        nodes[0].metadata["archive_serial_number"]
        == real_document.archive_serial_number
    )
    assert "filename" in nodes[0].excluded_embed_metadata_keys
    assert "filename" not in nodes[0].excluded_llm_metadata_keys


@pytest.mark.django_db
def test_build_document_node_sets_ref_doc_id(real_document: Document) -> None:
    """Every node produced by build_document_node must carry the paperless document id
    as its ref_doc_id so that the vector store's delete(str(doc.id)) works correctly."""
    nodes = indexing.build_document_node(real_document)
    assert len(nodes) > 0, "Expected at least one node"
    for node in nodes:
        assert node.ref_doc_id == str(real_document.id), (
            f"Expected ref_doc_id={real_document.id!r}, got {node.ref_doc_id!r}"
        )


@pytest.mark.django_db
def test_build_document_node_excludes_metadata_from_embedding(
    real_document: Document,
) -> None:
    """Metadata keys must not be prepended to the embedding text.

    build_llm_index_text already encodes all metadata in the body text, so
    including it again via llama_index's default MetadataMode.EMBED would
    double the token count and exceed embedding models with small context
    windows (e.g. nomic-embed-text via Ollama defaults to num_ctx=2048).
    """
    nodes = indexing.build_document_node(real_document)
    for node in nodes:
        embed_text = node.get_content(metadata_mode=MetadataMode.EMBED)
        for key in node.metadata:
            assert key not in embed_text, (
                f"Metadata key '{key}' should not appear in embedding text"
            )


@pytest.mark.django_db
def test_build_document_node_structured_fields_in_metadata(
    real_document: Document,
) -> None:
    """Structured fields must be in node.metadata so the LLM receives them via metadata prepend."""
    nodes = indexing.build_document_node(real_document)
    assert len(nodes) > 0
    for node in nodes:
        assert "title" in node.metadata
        assert "tags" in node.metadata
        assert "correspondent" in node.metadata
        assert "document_type" in node.metadata
        assert "created" in node.metadata
        assert "added" in node.metadata
        assert "modified" in node.metadata


@pytest.mark.django_db
def test_build_document_node_survives_concurrently_deleted_correspondent(
    real_document: Document,
) -> None:
    """Regression test for #13314.

    If a document's correspondent (or document type) is deleted after the
    in-memory Document instance was loaded but before build_document_node
    resolves the relation, accessing the FK must not raise - it should
    behave like an unset FK and produce None in the metadata instead of
    aborting the whole indexing pass.
    """
    correspondent = Correspondent.objects.create(name="Stale Correspondent")
    document_type = DocumentType.objects.create(name="Stale Type")
    real_document.correspondent = correspondent
    real_document.document_type = document_type
    real_document.save()

    # Re-fetch to get an instance whose correspondent/document_type relations
    # are unresolved (not yet cached), mirroring a task that loaded the
    # document before the concurrent deletion below.
    stale_document = Document.objects.get(pk=real_document.pk)

    correspondent.delete()
    document_type.delete()

    nodes = indexing.build_document_node(stale_document)
    assert len(nodes) > 0
    assert nodes[0].metadata["correspondent"] is None
    assert nodes[0].metadata["document_type"] is None


@pytest.mark.django_db
def test_build_document_node_excludes_document_id_from_llm_context(
    real_document: Document,
) -> None:
    """document_id is an internal key and must not appear in LLM context text."""
    nodes = indexing.build_document_node(real_document)
    assert len(nodes) > 0
    for node in nodes:
        assert "document_id" in node.excluded_llm_metadata_keys
        assert "document_id" not in node.get_content(metadata_mode=MetadataMode.LLM)


@pytest.mark.django_db
def test_build_document_node_uses_rag_chunk_settings(real_document: Document) -> None:
    app_config, _ = ApplicationConfiguration.objects.get_or_create()
    app_config.llm_embedding_chunk_size = 512
    app_config.save()

    with patch("llama_index.core.node_parser.SimpleNodeParser") as mock_parser:
        mock_parser.return_value.get_nodes_from_documents.return_value = []

        indexing.build_document_node(real_document)

        mock_parser.assert_called_once_with(chunk_size=512, chunk_overlap=200)


def test_get_rag_chunk_overlap_clamps_to_chunk_size() -> None:
    with patch("paperless_ai.indexing.RAG_CHUNK_OVERLAP", 128):
        assert indexing.get_rag_chunk_overlap(64) == 63


@pytest.mark.django_db
def test_get_rag_prompt_helper_uses_context_setting() -> None:
    app_config, _ = ApplicationConfiguration.objects.get_or_create()
    app_config.llm_context_size = 4096
    app_config.save()

    prompt_helper = indexing.get_rag_prompt_helper()

    assert prompt_helper.context_window == 4096


def test_truncate_embedding_query_returns_single_chunk() -> None:
    content = " ".join(f"word{i}" for i in range(200))

    result = indexing.truncate_embedding_query(content, chunk_size=32)

    assert result
    assert result != content
    assert "word199" not in result


@pytest.mark.django_db
def test_update_llm_index(
    temp_llm_index_dir: Path,
    real_document: Document,
    mock_embed_model: FakeEmbedding,
) -> None:
    mock_config = MagicMock()
    mock_config.llm_embedding_chunk_size = 512
    with (
        patch("documents.models.Document.objects.all") as mock_all,
        patch("paperless_ai.indexing.AIConfig", return_value=mock_config) as ai_config,
        patch("paperless_ai.indexing.build_document_node") as build_document_node,
    ):
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = True
        mock_queryset.__iter__.return_value = iter([real_document])
        mock_queryset.select_related.return_value = mock_queryset
        mock_queryset.prefetch_related.return_value = mock_queryset
        mock_all.return_value = mock_queryset
        build_document_node.return_value = []
        indexing.update_llm_index(rebuild=True)

        ai_config.assert_called_once()
        build_document_node.assert_called_once_with(real_document, chunk_size=512)


@pytest.mark.django_db
def test_update_llm_index_rebuilds_on_model_name_change(
    temp_llm_index_dir: Path,
    real_document: Document,
    mock_embed_model: FakeEmbedding,
) -> None:
    # Build initial index with model "model-a".
    with patch("documents.models.Document.objects.all") as mock_all:
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = True
        mock_queryset.__iter__.return_value = iter([real_document])
        mock_queryset.select_related.return_value = mock_queryset
        mock_queryset.prefetch_related.return_value = mock_queryset
        mock_all.return_value = mock_queryset
        with patch(
            "paperless_ai.indexing.get_configured_model_name",
            return_value="model-a",
        ):
            indexing.update_llm_index(rebuild=True)

    # Simulate config change to "model-b"; the incremental run must force a rebuild.
    with patch("documents.models.Document.objects.all") as mock_all:
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = True
        mock_queryset.__iter__.return_value = iter([real_document])
        mock_queryset.select_related.return_value = mock_queryset
        mock_queryset.prefetch_related.return_value = mock_queryset
        mock_all.return_value = mock_queryset
        with patch(
            "paperless_ai.indexing.get_configured_model_name",
            return_value="model-b",
        ):
            indexing.update_llm_index(rebuild=False)

    with indexing.get_vector_store() as store:
        # Schema metadata only updates when the table is dropped and recreated, never
        # on incremental writes - so "model-b" here proves a full rebuild happened.
        assert store.stored_model_name() == "model-b"


@pytest.mark.django_db
def test_update_llm_index_merges_exists_and_config_mismatch_reads(
    temp_llm_index_dir: Path,
    real_document: Document,
    mock_embed_model: FakeEmbedding,
) -> None:
    # Build an initial index so the second call's table_exists()/
    # config_mismatch() checks have something real to check against.
    with patch("documents.models.Document.objects.all") as mock_all:
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = True
        mock_queryset.__iter__.return_value = iter([real_document])
        mock_queryset.select_related.return_value = mock_queryset
        mock_queryset.prefetch_related.return_value = mock_queryset
        mock_all.return_value = mock_queryset
        indexing.update_llm_index(rebuild=True)

    with patch("documents.models.Document.objects.all") as mock_all:
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = True
        mock_queryset.__iter__.return_value = iter([real_document])
        mock_queryset.select_related.return_value = mock_queryset
        mock_queryset.prefetch_related.return_value = mock_queryset
        mock_all.return_value = mock_queryset
        with patch(
            "paperless_ai.indexing.read_store",
            wraps=indexing.read_store,
        ) as read_store_spy:
            indexing.update_llm_index(rebuild=False)

    # Documents exist, so the fast-exit check's `no_documents and ...`
    # short-circuits before ever calling llm_index_exists() - the only
    # read_store() call left in this path is the merged table_exists()/
    # config_mismatch() check. Before this task's fix, that merged check
    # was two separate read_store() calls (one inside llm_index_exists(),
    # one for config_mismatch() right after) - so this asserts 1, not 2.
    assert read_store_spy.call_count == 1


@pytest.mark.django_db
def test_update_llm_index_partial_update(
    temp_llm_index_dir: Path,
    real_document: Document,
    mock_embed_model: FakeEmbedding,
) -> None:
    doc2 = Document.objects.create(
        title="Test Document 2",
        content="This is some test content 2.",
        added=timezone.now(),
        checksum="1234567890abcdef",
    )
    # Initial index
    with patch("documents.models.Document.objects.all") as mock_all:
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = True
        mock_queryset.__iter__.return_value = iter([real_document, doc2])
        mock_queryset.select_related.return_value = mock_queryset
        mock_queryset.prefetch_related.return_value = mock_queryset
        mock_all.return_value = mock_queryset

        indexing.update_llm_index(rebuild=True)

    # modify document
    updated_document = real_document
    updated_document.modified = timezone.now()  # simulate modification

    # new doc
    doc3 = Document.objects.create(
        title="Test Document 3",
        content="This is some test content 3.",
        added=timezone.now(),
        checksum="abcdef1234567890",
    )

    with patch("documents.models.Document.objects.all") as mock_all:
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = True
        mock_queryset.__iter__.return_value = iter([updated_document, doc2, doc3])
        mock_all.return_value = mock_queryset

        indexing.update_llm_index(rebuild=False)

    with indexing.get_vector_store() as store:
        assert store.table_exists(), (
            "Expected the vector store table to exist after incremental update"
        )
        before = store.get_modified_times()

    # new doc, also touched by the scoped update below
    doc4 = DocumentFactory.create(title="Test Document 4", added=timezone.now())

    # A further edit, scoped via document_ids to doc3 + doc4 - doc2 must be
    # left exactly as it was, proving document_ids restricts the scan
    # instead of falling back to the whole library.
    doc3.modified = timezone.now()
    doc3.save()
    doc4.modified = timezone.now()
    doc4.save()

    # Give both scoped documents a note and a custom field: build_llm_index_text
    # reads both per document, so without the notes/custom_fields__field
    # prefetch on scoped_documents, each additional document adds 3 more
    # queries (N+1 regression) instead of the query count staying flat.
    custom_field = CustomField.objects.create(
        name="Priority",
        data_type=CustomField.FieldDataType.STRING,
    )
    for doc in (doc3, doc4):
        Note.objects.create(document=doc, note=f"a note on {doc.title}")
        CustomFieldInstance.objects.create(
            document=doc,
            field=custom_field,
            value_text="high",
        )

    with CaptureQueriesContext(connection) as ctx:
        result = indexing.update_llm_index(
            rebuild=False,
            document_ids=[doc3.pk, doc4.pk],
        )
    assert result == "LLM index updated successfully."
    # Notes/custom fields are prefetched in one batch query each (plus one
    # more for custom_fields__field), not re-queried per document - an N+1
    # regression here would scale with document count instead of staying flat
    # (7 with the prefetch vs. 10 without it, for these 2 documents).
    assert len(ctx.captured_queries) <= 8

    with indexing.get_vector_store() as store:
        after = store.get_modified_times()

    assert after[str(doc3.pk)] == doc3.modified.isoformat()
    assert after[str(doc2.pk)] == before[str(doc2.pk)]


@pytest.mark.django_db
class TestLlmIndexVersions:
    """The LLM index holds root documents only: a version is indexed as its root."""

    def test_add_or_update_document_indexes_a_version_as_its_root(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        """
        GIVEN:
            - A root document with a version
        WHEN:
            - The version is passed to llm_index_add_or_update_document
        THEN:
            - Only the root document is in the index
        """
        root = DocumentFactory(content="root content")
        version = DocumentFactory(root_document=root, version_index=1)

        indexing.llm_index_add_or_update_document(version)

        with indexing.get_vector_store() as store:
            indexed = store.get_modified_times()

        assert set(indexed) == {str(root.pk)}

    def test_rebuild_skips_versions(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        """
        GIVEN:
            - A root document with a version
        WHEN:
            - The LLM index is rebuilt
        THEN:
            - Only the root document is in the index
        """
        root = DocumentFactory()
        DocumentFactory(root_document=root, version_index=1)

        indexing.update_llm_index(rebuild=True)

        with indexing.get_vector_store() as store:
            indexed = store.get_modified_times()

        assert set(indexed) == {str(root.pk)}

    def test_rebuild_indexes_the_newest_versions_content(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """
        GIVEN:
            - A root document with a version
        WHEN:
            - The LLM index is rebuilt
        THEN:
            - The root's text for the index is the version's content, answered
              from the query rather than a query per document
        """
        root = DocumentFactory(content="stale text")
        DocumentFactory(root_document=root, version_index=1, content="latest text")
        spy = mocker.spy(indexing, "build_document_node")

        indexing.update_llm_index(rebuild=True)

        indexed = spy.call_args.args[0]
        assert indexed.pk == root.pk
        assert indexed.effective_content == "latest text"

    def test_incremental_update_by_version_id_refreshes_the_root(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        """
        GIVEN:
            - An indexed root document with a version whose root was modified since
        WHEN:
            - An incremental update is scoped to the version's id
        THEN:
            - The root's entry is refreshed and no entry exists for the version
        """
        root = DocumentFactory()
        version = DocumentFactory(root_document=root, version_index=1)
        indexing.update_llm_index(rebuild=True)
        Document.objects.filter(pk=root.pk).update(modified=timezone.now())
        root.refresh_from_db()

        indexing.update_llm_index(document_ids=[version.pk])

        with indexing.get_vector_store() as store:
            indexed = store.get_modified_times()

        assert indexed == {str(root.pk): root.modified.isoformat()}


@pytest.mark.django_db
def test_add_or_update_document_updates_existing_entry(
    temp_llm_index_dir: Path,
    real_document: Document,
    mock_embed_model: FakeEmbedding,
) -> None:
    indexing.update_llm_index(rebuild=True)
    indexing.llm_index_add_or_update_document(real_document)

    with indexing.get_vector_store() as store:
        assert store.table_exists(), (
            "Expected the vector store table to exist after add-or-update"
        )


@pytest.mark.django_db
def test_query_after_remove_does_not_raise_key_error(
    temp_llm_index_dir: Path,
    real_document: Document,
    mock_embed_model: FakeEmbedding,
) -> None:
    indexing.update_llm_index(rebuild=True)

    query_doc = Document.objects.create(
        title="Query",
        content="query content",
        added=timezone.now(),
    )

    indexing.llm_index_remove_document(real_document)

    result = indexing.retrieve_similar_nodes(query_doc, top_k=5)
    assert isinstance(result, list)


@pytest.mark.django_db
def test_update_llm_index_no_documents(
    temp_llm_index_dir: Path,
    mock_embed_model: FakeEmbedding,
) -> None:
    with patch("documents.models.Document.objects.all") as mock_all:
        mock_queryset = MagicMock()
        mock_queryset.exists.return_value = False
        mock_queryset.__iter__.return_value = iter([])
        mock_all.return_value = mock_queryset

        # check log message
        with patch("paperless_ai.indexing.logger") as mock_logger:
            indexing.update_llm_index(rebuild=True)
            mock_logger.warning.assert_called_once_with(
                "No documents found to index.",
            )


@pytest.mark.django_db
def test_update_no_documents_no_index_returns_early(
    temp_llm_index_dir: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """update with no documents and no existing index must return early."""
    mock_qs = MagicMock()
    mock_qs.exists.return_value = False
    mock_qs.__iter__ = MagicMock(return_value=iter([]))
    mocker.patch("paperless_ai.indexing.Document.objects.all", return_value=mock_qs)

    result = indexing.update_llm_index(rebuild=False)

    assert result == "No documents found to index."


@pytest.mark.django_db
def test_queue_llm_index_update_if_needed_enqueues_when_idle_or_skips_recent() -> None:
    # No existing tasks
    with patch("documents.tasks.llmindex_index") as mock_task:
        result = indexing.queue_llm_index_update_if_needed(
            rebuild=True,
            reason="test enqueue",
        )

    assert result is True
    mock_task.apply_async.assert_called_once_with(
        kwargs={"rebuild": True},
        headers={"trigger_source": "system"},
    )

    PaperlessTaskFactory(
        task_type=PaperlessTask.TaskType.LLM_INDEX,
        trigger_source=PaperlessTask.TriggerSource.SYSTEM,
        status=PaperlessTask.Status.STARTED,
    )

    # Existing running task
    with patch("documents.tasks.llmindex_index") as mock_task:
        result = indexing.queue_llm_index_update_if_needed(
            rebuild=False,
            reason="should skip",
        )

    assert result is False
    mock_task.apply_async.assert_not_called()


@override_settings(
    LLM_EMBEDDING_BACKEND="huggingface",
    LLM_EMBEDDING_CHUNK_SIZE=32,
    LLM_BACKEND="ollama",
)
def test_retrieve_similar_nodes_truncates_query_to_embedding_chunk_size(
    temp_llm_index_dir: Path,
    real_document: Document,
) -> None:
    real_document.content = " ".join(f"word{i}" for i in range(200))
    with (
        patch("paperless_ai.indexing.load_or_build_index") as mock_load_or_build_index,
        patch(
            "paperless_ai.indexing.llm_index_exists",
        ) as mock_vector_store_exists,
        patch("llama_index.core.retrievers.VectorIndexRetriever") as mock_retriever_cls,
        patch("paperless_ai.indexing.truncate_content") as mock_truncate_content,
    ):
        mock_vector_store_exists.return_value = True
        mock_load_or_build_index.return_value = MagicMock()
        mock_truncate_content.return_value = "wrong helper"

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        indexing.retrieve_similar_nodes(real_document, top_k=3)

        mock_truncate_content.assert_not_called()
        query_text = mock_retriever.retrieve.call_args.args[0]
        assert query_text
        assert "word199" not in query_text


@pytest.mark.django_db
@override_settings(
    LLM_EMBEDDING_BACKEND="huggingface",
    LLM_BACKEND="ollama",
)
def test_retrieve_similar_nodes_queries_with_newest_versions_content(
    temp_llm_index_dir: Path,
) -> None:
    """
    GIVEN:
        - A root document whose newest version carries different content
    WHEN:
        - Documents similar to the root are retrieved
    THEN:
        - The query text holds the newest version's content, not the root's own
    """
    root = DocumentFactory(title="Statement", content="stale root text")
    DocumentFactory(root_document=root, version_index=1, content="latest version text")

    with (
        patch("paperless_ai.indexing.load_or_build_index"),
        patch("paperless_ai.indexing.llm_index_exists", return_value=True),
        patch("llama_index.core.retrievers.VectorIndexRetriever") as mock_retriever_cls,
    ):
        mock_retriever_cls.return_value.retrieve.return_value = []

        indexing.retrieve_similar_nodes(root, top_k=3)

    query_text = mock_retriever_cls.return_value.retrieve.call_args.args[0]
    assert "latest version text" in query_text
    assert "stale root text" not in query_text


class TestUpdateLlmIndexEmptyDocumentSet:
    """update_llm_index must clear the vector store table when all documents are deleted.

    Without this, the stale vectors are never cleared and subsequent similarity
    searches return phantom hits for document IDs that no longer exist in the DB.
    """

    @pytest.mark.django_db
    def test_rebuild_clears_stale_index_when_no_documents_exist(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        """After deleting all documents, rebuild=True must produce a table with zero rows.

        Steps:
        1. Build an index with one document so the on-disk state is non-empty.
        2. Delete all documents from the DB.
        3. Call update_llm_index(rebuild=True).
        4. Open the LanceDB table directly and assert zero rows.
        """
        # Step 1: create a document and build a non-empty index
        Document.objects.create(
            title="Soon-to-be-deleted document",
            content="Some content that will become a phantom vector.",
            added=timezone.now(),
        )
        indexing.update_llm_index(rebuild=True)

        with indexing.get_vector_store() as store:
            assert store.table_exists(), (
                "Precondition failed: expected the vector store table to exist "
                "before deletion"
            )

        # Step 2: delete all documents
        Document.objects.all().delete()
        assert not Document.objects.exists()

        # Step 3: rebuild with no documents — drop_table is called so the table
        # is removed (no rows to re-insert, so it stays absent).
        indexing.update_llm_index(rebuild=True)

        # Step 4: the table must be absent (no rows) — phantom vectors gone
        with indexing.get_vector_store() as store2:
            assert not store2.table_exists(), (
                "Expected the vector store table to be absent after rebuilding "
                "with no documents"
            )


class TestDocumentUpdatedSignalTriggersLlmReindex:
    """document_updated must enqueue an LLM index update, just like document_consumption_finished."""

    @pytest.mark.django_db
    @override_settings(AI_ENABLED=True, LLM_EMBEDDING_BACKEND="huggingface")
    def test_document_updated_enqueues_llm_reindex(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Firing document_updated should call update_document_in_llm_index.apply_async."""
        mock_task = mocker.patch("documents.tasks.update_document_in_llm_index")

        doc = DocumentFactory()
        with TestCase.captureOnCommitCallbacks(execute=True):
            document_updated.send(sender=object, document=doc)

        mock_task.apply_async.assert_called_once_with(kwargs={"document_id": doc.pk})

    @pytest.mark.django_db
    @override_settings(AI_ENABLED=True, LLM_EMBEDDING_BACKEND="huggingface")
    def test_document_updated_does_not_enqueue_before_rollback(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_task = mocker.patch("documents.tasks.update_document_in_llm_index")
        doc = DocumentFactory()
        with TestCase.captureOnCommitCallbacks(execute=False) as callbacks:
            with pytest.raises(RuntimeError, match="rollback"):
                with transaction.atomic():
                    document_updated.send(sender=object, document=doc)
                    mock_task.apply_async.assert_not_called()
                    raise RuntimeError("rollback")
        assert callbacks == []
        mock_task.apply_async.assert_not_called()

    @pytest.mark.django_db
    @override_settings(AI_ENABLED=True, LLM_EMBEDDING_BACKEND="huggingface")
    def test_version_addition_consumption_enqueues_llm_index_once(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """When a new version is consumed, the root document must be enqueued exactly once."""
        mock_task = mocker.patch("documents.tasks.update_document_in_llm_index")

        root_doc = DocumentFactory()
        with TestCase.captureOnCommitCallbacks(execute=True):
            document_consumption_finished.send(
                sender=object,
                document=root_doc,
                logging_group=None,
                classifier=None,
                original_file=None,
            )
            document_updated.send(sender=object, document=root_doc, skip_ai_index=True)

        assert mock_task.apply_async.call_count == 1


@pytest.mark.django_db
class TestLlmIndexAddOrUpdateDocumentEmptyContent:
    """llm_index_add_or_update_document must handle empty node lists gracefully."""

    def test_returns_without_error_when_build_document_node_returns_empty(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: MagicMock,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """When build_document_node returns [], the function must return without error.

        The store's upsert_document treats an empty node list as a removal (no-op
        delete), so load_or_build_index must not be called.
        """
        mocker.patch(
            "paperless_ai.indexing.build_document_node",
            return_value=[],
        )
        mock_load = mocker.patch("paperless_ai.indexing.load_or_build_index")

        doc = MagicMock(spec=Document)
        doc.id = 42
        doc.root_document_id = None
        # Must not raise
        indexing.llm_index_add_or_update_document(doc)

        mock_load.assert_not_called()


@pytest.mark.django_db
def test_llm_index_compact_uses_force(
    temp_llm_index_dir: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """compact must use force=True to rebuild the table and reclaim space immediately."""
    mock_store = mocker.MagicMock()
    mocker.patch(
        "paperless_ai.indexing.write_store",
        return_value=mocker.MagicMock(
            __enter__=mocker.MagicMock(return_value=mock_store),
            __exit__=mocker.MagicMock(return_value=False),
        ),
    )

    indexing.llm_index_compact()

    mock_store.compact.assert_called_once_with(force=True)


@pytest.mark.django_db
class TestLlmIndexLocking:
    """Index mutation functions must go through write_store(), which holds the lock.

    Without locking, two concurrent Celery workers can open the same store,
    make independent modifications, and trigger CommitConflictError.
    """

    def test_add_or_update_document_uses_write_store(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_store = MagicMock()
        mock_store.has_pending_migration.return_value = False
        mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )
        mock_node = MagicMock()
        mock_node.get_content.return_value = "fake node text"
        mocker.patch(
            "paperless_ai.indexing.build_document_node",
            return_value=[mock_node],
        )

        doc = MagicMock(spec=Document)
        doc.id = 1
        indexing.llm_index_add_or_update_document(doc)

        mock_store.upsert_document.assert_called_once()

    def test_add_or_update_document_skips_write_when_reembed_pending(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """A pending re-embed migration must block the incremental write,
        not let it proceed against a schema that just changed underneath it.
        """
        mock_store = MagicMock()
        mock_store.has_pending_migration.return_value = True
        mock_store.check_and_run_migrations.return_value = True
        mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )
        mock_node = MagicMock()
        mock_node.get_content.return_value = "fake node text"
        mocker.patch(
            "paperless_ai.indexing.build_document_node",
            return_value=[mock_node],
        )

        doc = MagicMock(spec=Document)
        doc.id = 1
        indexing.llm_index_add_or_update_document(doc)

        mock_store.upsert_document.assert_not_called()

    def test_add_or_update_document_skips_write_when_migration_check_deferred(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """A migration check that times out waiting for readers to drain
        must be treated the same as a pending migration - proceeding to
        write would target a store still on its old schema. Regression
        test for the tri-state fix: a bare bool collapsed this outcome
        into the same falsy value as "already current".
        """
        mock_store = MagicMock()
        mock_store.has_pending_migration.return_value = True
        mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )
        mocker.patch(
            "paperless_ai.indexing._exclude_readers",
            side_effect=Timeout("test"),
        )
        mock_node = MagicMock()
        mock_node.get_content.return_value = "fake node text"
        mocker.patch(
            "paperless_ai.indexing.build_document_node",
            return_value=[mock_node],
        )

        doc = MagicMock(spec=Document)
        doc.id = 1
        indexing.llm_index_add_or_update_document(doc)

        mock_store.upsert_document.assert_not_called()

    def test_remove_document_uses_write_store(
        self,
        temp_llm_index_dir: Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_store = MagicMock()
        mock_store.has_pending_migration.return_value = False
        mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )

        doc = MagicMock(spec=Document)
        doc.id = 1
        indexing.llm_index_remove_document(doc)

        mock_store.delete.assert_called_once_with("1")

    def test_remove_document_skips_write_when_reembed_pending(
        self,
        temp_llm_index_dir: Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """A pending re-embed migration must block the delete too, for the
        same consistency reason as the incremental-update path.
        """
        mock_store = MagicMock()
        mock_store.has_pending_migration.return_value = True
        mock_store.check_and_run_migrations.return_value = True
        mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )

        doc = MagicMock(spec=Document)
        doc.id = 1
        indexing.llm_index_remove_document(doc)

        mock_store.delete.assert_not_called()

    def test_remove_document_skips_write_when_migration_check_deferred(
        self,
        temp_llm_index_dir: Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """A migration check deferred by a reader-lock timeout must block
        the delete too, for the same reason as the incremental-update path.
        """
        mock_store = MagicMock()
        mock_store.has_pending_migration.return_value = True
        mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )
        mocker.patch(
            "paperless_ai.indexing._exclude_readers",
            side_effect=Timeout("test"),
        )

        doc = MagicMock(spec=Document)
        doc.id = 1
        indexing.llm_index_remove_document(doc)

        mock_store.delete.assert_not_called()

    def test_update_llm_index_rebuild_uses_write_store(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mock_store = MagicMock()
        mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )
        mock_qs = MagicMock()
        mock_qs.exists.return_value = True
        mock_qs.__iter__ = MagicMock(return_value=iter([]))
        mocker.patch("paperless_ai.indexing.Document.objects.all", return_value=mock_qs)

        indexing.update_llm_index(rebuild=True)

        mock_store.drop_table.assert_called_once()

    def test_update_llm_index_skips_when_migration_check_deferred(
        self,
        temp_llm_index_dir: Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """A migration check deferred by a reader-lock timeout must short-
        circuit before the second write_store() block (document scanning,
        add/upsert, compaction) ever runs - that block would otherwise
        write against a store still on its old schema.
        """
        mock_store = MagicMock()
        mock_store.has_pending_migration.return_value = True
        write_store_mock = mocker.patch(
            "paperless_ai.indexing.write_store",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=mock_store),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )
        mocker.patch(
            "paperless_ai.indexing._exclude_readers",
            side_effect=Timeout("test"),
        )

        result = indexing.update_llm_index(rebuild=False)

        assert "deferred" in result
        write_store_mock.assert_called_once()


@pytest.mark.django_db
@pytest.mark.django_db
class TestVectorStoreIndexing:
    def test_get_vector_store_roundtrip(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        with indexing.get_vector_store() as store:
            assert isinstance(store, PaperlessSqliteVecVectorStore)

    def test_add_then_remove_document(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
        real_document: Document,
    ) -> None:
        indexing.llm_index_add_or_update_document(real_document)
        with indexing.get_vector_store() as store:
            assert store.table_exists()
            count_sql = "SELECT count(*) FROM documents"
            assert store.client.execute(count_sql).fetchone()[0] >= 1

            indexing.llm_index_remove_document(real_document)
            assert store.client.execute(count_sql).fetchone()[0] == 0

    def test_update_shrinks_chunks_without_orphans(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
        real_document: Document,
    ) -> None:
        real_document.content = "word " * 4000  # many chunks
        real_document.save()
        indexing.llm_index_add_or_update_document(real_document)
        count_sql = "SELECT count(*) FROM documents"
        with indexing.get_vector_store() as store:
            big = store.client.execute(count_sql).fetchone()[0]

            real_document.content = "short"  # one chunk
            real_document.save()
            indexing.llm_index_add_or_update_document(real_document)

            rows = store.client.execute(count_sql).fetchone()[0]
            assert rows < big
            assert rows >= 1


class TestLlmIndexMigrate:
    def test_noop_when_ai_disabled(self, mocker: pytest_mock.MockerFixture) -> None:
        """
        GIVEN:
            - AI/LLM index support is disabled in configuration
        WHEN:
            - llm_index_migrate() is called
        THEN:
            - No store is opened and no migration check runs
        """
        mocker.patch(
            "paperless_ai.indexing.AIConfig",
            return_value=mocker.Mock(llm_index_enabled=False),
        )
        write_store_mock = mocker.patch("paperless_ai.indexing.write_store")
        indexing.llm_index_migrate()
        write_store_mock.assert_not_called()

    def test_runs_pending_migration_when_enabled(
        self,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """
        GIVEN:
            - AI/LLM index support is enabled
        WHEN:
            - llm_index_migrate() is called
        THEN:
            - The store is opened for write and a migration check runs
        """
        mocker.patch(
            "paperless_ai.indexing.AIConfig",
            return_value=mocker.Mock(llm_index_enabled=True),
        )
        store_mock = mocker.MagicMock()
        store_mock.has_pending_migration.return_value = False
        write_store_cm = mocker.patch("paperless_ai.indexing.write_store")
        write_store_cm.return_value.__enter__.return_value = store_mock
        indexing.llm_index_migrate()
        store_mock.has_pending_migration.assert_called_once()

    def test_logs_warning_when_reembed_needed(
        self,
        mocker: pytest_mock.MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        GIVEN:
            - AI/LLM index support is enabled
            - A pending migration requires re-embedding
        WHEN:
            - llm_index_migrate() is called
        THEN:
            - A warning directs the operator to run a manual rebuild, since
              this automatic check must never re-embed on its own
        """
        mocker.patch(
            "paperless_ai.indexing.AIConfig",
            return_value=mocker.Mock(llm_index_enabled=True),
        )
        store_mock = mocker.MagicMock()
        store_mock.has_pending_migration.return_value = True
        store_mock.check_and_run_migrations.return_value = True
        write_store_cm = mocker.patch("paperless_ai.indexing.write_store")
        write_store_cm.return_value.__enter__.return_value = store_mock
        with caplog.at_level(logging.WARNING, logger="paperless_ai.indexing"):
            indexing.llm_index_migrate()
        assert "requires re-embedding" in caplog.text

    def test_logs_info_when_migration_check_deferred(
        self,
        mocker: pytest_mock.MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        GIVEN:
            - AI/LLM index support is enabled
            - A pending migration cannot run because readers are active
        WHEN:
            - llm_index_migrate() is called
        THEN:
            - An info line notes the deferral, not the re-embed warning
        """
        mocker.patch(
            "paperless_ai.indexing.AIConfig",
            return_value=mocker.Mock(llm_index_enabled=True),
        )
        store_mock = mocker.MagicMock()
        store_mock.has_pending_migration.return_value = True
        write_store_cm = mocker.patch("paperless_ai.indexing.write_store")
        write_store_cm.return_value.__enter__.return_value = store_mock
        mocker.patch(
            "paperless_ai.indexing._exclude_readers",
            side_effect=Timeout("test"),
        )
        with caplog.at_level(logging.INFO, logger="paperless_ai.indexing"):
            indexing.llm_index_migrate()
        assert "deferred" in caplog.text
        assert "requires re-embedding" not in caplog.text


@pytest.mark.django_db
def test_retrieve_similar_nodes_returns_raw_nodes_from_retriever(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """
    GIVEN:
        - A source document and a mocked retriever returning one node
    WHEN:
        - retrieve_similar_nodes() is called with no document_ids filter
    THEN:
        - The retriever's raw result is returned unchanged

    Source-document self-exclusion is a real vector-store MetadataFilters
    behavior this mocked retriever bypasses entirely - see
    TestRetrieveSimilarNodesAgainstRealIndex.test_excludes_self for that
    coverage against a real index.
    """
    source = DocumentFactory.create()
    other = DocumentFactory.create()
    fake_node = mocker.MagicMock()
    fake_node.metadata = {"document_id": str(other.pk)}
    mocker.patch("paperless_ai.indexing.llm_index_exists", return_value=True)
    mock_retriever_cls = mocker.patch(
        "llama_index.core.retrievers.VectorIndexRetriever",
    )
    mock_retriever_cls.return_value.retrieve.return_value = [fake_node]
    mocker.patch("paperless_ai.indexing.load_or_build_index")
    mocker.patch("paperless_ai.indexing.read_store")

    nodes = indexing.retrieve_similar_nodes(source, top_k=5)

    assert nodes == [fake_node]


@pytest.mark.django_db
def test_retrieve_similar_nodes_drops_result_outside_allow_list(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """
    GIVEN:
        - An allow-list naming only one document
        - A mocked retriever that returns a node for a DIFFERENT document
          (as if the vec0-level MetadataFilters had failed to apply)
    WHEN:
        - retrieve_similar_nodes() is called with that allow-list
    THEN:
        - The out-of-allow-list node is dropped by this function's own
          Python-level re-check, independent of whatever filtering the
          vector store itself applied - this is the defense-in-depth layer
          for a permission boundary, so it must work standalone.
    """
    source = DocumentFactory.create()
    allowed = DocumentFactory.create()
    not_allowed = DocumentFactory.create()
    allowed_node = mocker.MagicMock()
    allowed_node.metadata = {"document_id": str(allowed.pk)}
    disallowed_node = mocker.MagicMock()
    disallowed_node.metadata = {"document_id": str(not_allowed.pk)}
    mocker.patch("paperless_ai.indexing.llm_index_exists", return_value=True)
    mock_retriever_cls = mocker.patch(
        "llama_index.core.retrievers.VectorIndexRetriever",
    )
    mock_retriever_cls.return_value.retrieve.return_value = [
        allowed_node,
        disallowed_node,
    ]
    mocker.patch("paperless_ai.indexing.load_or_build_index")
    mocker.patch("paperless_ai.indexing.read_store")

    nodes = indexing.retrieve_similar_nodes(source, document_ids=[allowed.pk])

    assert nodes == [allowed_node]


@pytest.mark.django_db
def test_retrieve_similar_nodes_returns_empty_when_index_missing(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """
    GIVEN:
        - No LLM index exists yet
    WHEN:
        - retrieve_similar_nodes() is called
    THEN:
        - An empty list is returned and an index build is queued
    """
    source = DocumentFactory.create()
    mocker.patch("paperless_ai.indexing.llm_index_exists", return_value=False)
    mocker.patch("paperless_ai.indexing.queue_llm_index_update_if_needed")

    nodes = indexing.retrieve_similar_nodes(source)

    assert nodes == []


@pytest.mark.django_db
def test_retrieve_similar_nodes_empty_document_ids_short_circuits(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """
    GIVEN:
        - An empty document_ids allow-list
    WHEN:
        - retrieve_similar_nodes() is called
    THEN:
        - An empty list is returned without checking whether an index exists
    """
    source = DocumentFactory.create()
    spy = mocker.patch("paperless_ai.indexing.llm_index_exists")

    nodes = indexing.retrieve_similar_nodes(source, document_ids=[])

    assert nodes == []
    spy.assert_not_called()


@pytest.mark.django_db
class TestRetrieveSimilarNodesAgainstRealIndex:
    """End-to-end allow-list and self-exclusion coverage against a real
    on-disk index (the mocked-retriever tests above cannot see the metadata
    filters actually being applied by the vector store)."""

    def test_respects_allowed_ids(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        """
        GIVEN:
            - Three indexed documents and an allow-list naming only one of them
        WHEN:
            - retrieve_similar_nodes() is called with that allow-list
        THEN:
            - Only nodes for the allowed document are returned
        """
        a = DocumentFactory.create(content="alpha shared content here")
        b = DocumentFactory.create(content="beta shared content here")
        c = DocumentFactory.create(content="gamma shared content here")
        for doc in (a, b, c):
            indexing.llm_index_add_or_update_document(doc)

        nodes = indexing.retrieve_similar_nodes(a, document_ids=[b.id])

        assert all(int(node.metadata["document_id"]) == b.id for node in nodes)

    def test_excludes_self(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        """
        GIVEN:
            - The source document and one other document are both indexed
        WHEN:
            - retrieve_similar_nodes() is called for the source document
        THEN:
            - The source document's own nodes are excluded from the results
        """
        a = DocumentFactory.create(content="alpha shared content here")
        b = DocumentFactory.create(content="beta shared content here")
        for doc in (a, b):
            indexing.llm_index_add_or_update_document(doc)

        nodes = indexing.retrieve_similar_nodes(a, top_k=5)

        assert {int(node.metadata["document_id"]) for node in nodes} == {b.id}

    def test_excludes_self_with_multiple_chunks(
        self,
        temp_llm_index_dir: Path,
        mock_embed_model: FakeEmbedding,
    ) -> None:
        """
        GIVEN:
            - A source document long enough to be split into many chunks, so
              it could otherwise occupy several of the top-k slots itself
        WHEN:
            - retrieve_similar_nodes() is called for the source document
        THEN:
            - Every one of its own chunks is excluded from the results
        """
        a = DocumentFactory.create(content="word " * 4000)
        b = DocumentFactory.create(content="beta shared content here")
        for doc in (a, b):
            indexing.llm_index_add_or_update_document(doc)

        nodes = indexing.retrieve_similar_nodes(a, top_k=3)

        assert {int(node.metadata["document_id"]) for node in nodes} == {b.id}
