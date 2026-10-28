from __future__ import annotations

import base64

import pytest

from echo_bio.artifacts.recall import ArtifactRecallService
from echo_bio.artifacts.store import (
    ArtifactCorruptionError,
    ArtifactIdentityError,
    ArtifactNotFoundError,
    LocalArtifactStore,
)


def test_content_addressed_store_round_trips_exact_bytes_and_deduplicates(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    content = "αβγ\nline two\n".encode()

    first = store.put_bytes(
        run_id="run-1",
        content=content,
        media_type="text/plain",
        encoding="utf-8",
        data_class="public",
    )
    second = store.put_bytes(
        run_id="run-1",
        content=content,
        media_type="text/plain",
        encoding="utf-8",
        data_class="public",
    )

    assert first == second
    assert store.get_bytes(first) == content
    assert store.get_text(first) == content.decode()
    assert store.list_run("run-1") == (first,)
    assert store.resolve(first.uri) == first


def test_recall_supports_exact_byte_and_line_paging(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_text(run_id="run-1", text="zero\none\ntwo\nthree\n", data_class="public")
    recall = ArtifactRecallService(store)

    byte_slice = recall.bytes(ref.uri, offset=5, length=7)
    line_slice = recall.lines(ref.uri, start_line=2, line_count=2)

    assert base64.b64decode(byte_slice.content_base64) == store.get_bytes(ref)[5:12]
    assert line_slice.content_text == "one\ntwo\n"
    assert line_slice.start == 2
    assert line_slice.total == 4
    assert not line_slice.complete


def test_same_content_is_namespaced_per_run(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    first = store.put_text(run_id="run-a", text="same", data_class="internal")
    second = store.put_text(run_id="run-b", text="same", data_class="internal")

    assert first.sha256 == second.sha256
    assert first.uri != second.uri
    assert store.resolve(first.uri).run_id == "run-a"
    assert store.resolve(second.uri).run_id == "run-b"


def test_corruption_and_wrong_hash_are_rejected(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_text(run_id="run-1", text="immutable", data_class="public")
    path = store._path_for_ref(ref)
    path.write_bytes(b"tampered")

    with pytest.raises(ArtifactCorruptionError, match="hash/size mismatch"):
        store.get_bytes(ref)
    wrong_uri = ref.uri.replace(ref.sha256, "0" * 64)
    with pytest.raises((ArtifactIdentityError, ArtifactNotFoundError)):
        store.resolve(wrong_uri)


def test_unsafe_run_namespace_is_rejected(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="unsafe run identifier"):
        store.put_text(run_id="../escape", text="no", data_class="public")
