from __future__ import annotations

from crabpath.index import VectorIndex


def test_vector_index_cosine() -> None:
    assert VectorIndex.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert abs(VectorIndex.cosine([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-12


def test_index_search_and_persistence(tmp_path) -> None:
    index = VectorIndex()
    index.upsert("a", [1.0, 0.0])
    index.upsert("b", [0.0, 1.0])

    assert index.search([0.9, 0.1], top_k=1)[0][0] == "a"
    index.remove("a")
    assert index.search([1.0, 0.0], top_k=2)[0][0] == "b"

    path = tmp_path / "index.json"
    index.save(str(path))
    loaded = VectorIndex.load(str(path))
    assert loaded.search([1.0, 0.0], top_k=1)[0][0] == "b"
