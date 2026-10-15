import json
from pathlib import Path

import pytest
from django.test import override_settings

from paperless_ai import indexing
from paperless_ai.exceptions import EmbeddingRateLimitError


@override_settings(DATA_DIR="/tmp/unused")
def test_embedding_pacing_persists_across_task_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.settings, "DATA_DIR", str(tmp_path))
    clock = [1000.0]
    monkeypatch.setattr(indexing.time, "time", lambda: clock[0])

    indexing._reserve_embedding_capacity("test-model", ["x" * 4000])
    # A fresh invocation sees the same persistent reservation and defers before
    # calling the provider; there is no process-local state to reset.
    clock[0] += indexing._EMBEDDING_MIN_INTERVAL_SECONDS - 0.1
    with pytest.raises(EmbeddingRateLimitError) as exc:
        indexing._reserve_embedding_capacity("test-model", ["x"])
    assert exc.value.retry_after >= 0.1

    clock[0] += 0.1
    indexing._reserve_embedding_capacity("test-model", ["x"])
    state = json.loads(next(Path(tmp_path).glob(".*.pacing.json")).read_text())
    assert len(state["events"]) == 2


@override_settings(DATA_DIR="/tmp/unused")
def test_embedding_pacing_defers_when_tpm_window_is_full(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing.settings, "DATA_DIR", str(tmp_path))
    clock = [2000.0]
    monkeypatch.setattr(indexing.time, "time", lambda: clock[0])
    payload = "x" * (indexing._EMBEDDING_TPM_LIMIT * 4)

    indexing._reserve_embedding_capacity("test-model", [payload])
    clock[0] += indexing._EMBEDDING_MIN_INTERVAL_SECONDS
    with pytest.raises(EmbeddingRateLimitError) as exc:
        indexing._reserve_embedding_capacity("test-model", ["x"])
    assert exc.value.retry_after >= indexing._EMBEDDING_TPM_WINDOW_SECONDS - indexing._EMBEDDING_MIN_INTERVAL_SECONDS

    clock[0] += indexing._EMBEDDING_TPM_WINDOW_SECONDS
    indexing._reserve_embedding_capacity("test-model", ["x"])
