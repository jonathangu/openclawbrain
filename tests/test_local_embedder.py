from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
import types


def test_local_embedder_embed_is_normalized_and_sets_dim(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            for text in texts:
                scale = float(len(text) or 1)
                yield [3.0 * scale, 4.0 * scale, 0.0, 0.0]

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    module_name = "openclawbrain.local_embedder"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    module._MODEL_CACHE.clear()

    embedder = module.LocalEmbedder()
    vec = embedder.embed("hello")
    assert len(vec) == 4
    assert embedder.dim == 4
    assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-6)


def test_local_embedder_embed_batch_is_normalized(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            for idx, _text in enumerate(texts):
                yield [1.0 + idx, 0.0, 0.0]

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    module_name = "openclawbrain.local_embedder"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    module._MODEL_CACHE.clear()

    embedder = module.LocalEmbedder()
    vectors = embedder.embed_batch([("a", "alpha"), ("b", "beta")])
    assert set(vectors) == {"a", "b"}
    assert embedder.dim == 3
    assert math.isclose(sum(v * v for v in vectors["a"]), 1.0, rel_tol=1e-6)
    assert math.isclose(sum(v * v for v in vectors["b"]), 1.0, rel_tol=1e-6)


def test_local_embedder_uses_persistent_cache_dir(monkeypatch, tmp_path) -> None:
    seen: dict[str, str | None] = {}

    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
            seen["model_name"] = model_name
            seen["cache_dir"] = cache_dir

        def embed(self, texts: list[str]):
            for _text in texts:
                yield [1.0, 0.0, 0.0]

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    module_name = "openclawbrain.local_embedder"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    module._MODEL_CACHE.clear()

    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    embedder = module.LocalEmbedder()
    embedder.embed("hello")
    assert seen["cache_dir"] == str(Path.home() / ".cache" / "fastembed")

    custom_dir = tmp_path / "fastembed-cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(custom_dir))
    module._MODEL_CACHE.clear()
    seen.clear()
    embedder = module.LocalEmbedder()
    embedder.embed("hello again")
    assert seen["cache_dir"] == str(custom_dir)
