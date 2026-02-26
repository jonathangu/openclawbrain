from __future__ import annotations

import json
import sys
from pathlib import Path

from crabpath.cli import main


def _write_texts(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "note.md::0": "alpha",
                "note.md::1": "beta",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_embed_command_creates_index(tmp_path) -> None:
    texts_path = tmp_path / "texts.json"
    _write_texts(texts_path)
    output_path = tmp_path / "index.json"
    embed_script = tmp_path / "embed.py"
    embed_script.write_text(
        """import json
import sys

for line in sys.stdin:
    payload = json.loads(line)
    text = payload['text']
    print(json.dumps({'id': payload['id'], 'embedding': [float(len(text)), 0.0]}))
""",
        encoding="utf-8",
    )

    code = main(
        [
            "embed",
            "--texts",
            str(texts_path),
            "--output",
            str(output_path),
            "--command",
            f"{sys.executable} {embed_script}",
        ]
    )
    assert code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["note.md::0"] == [5.0, 0.0]
    assert payload["note.md::1"] == [4.0, 0.0]
