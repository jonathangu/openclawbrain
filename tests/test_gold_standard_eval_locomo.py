from __future__ import annotations

import subprocess
import sys

from benchmarks.gold_standard_eval.state_utils import resolve_embedder


def test_resolve_embedder_accepts_none_and_hash():
    embedder_default = resolve_embedder(None)
    assert embedder_default.mode in ("local", "hash")

    embedder_hash = resolve_embedder("hash")
    assert embedder_hash.mode == "hash"


def test_run_locomo_help():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.gold_standard_eval.run_locomo", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
