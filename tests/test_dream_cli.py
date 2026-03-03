from __future__ import annotations

from pathlib import Path

from openclawbrain import cli as cli_module
from openclawbrain.state_lock import state_write_lock


def test_dream_parser_accepts_subcommand() -> None:
    parser = cli_module._build_parser()
    args = parser.parse_args(["dream", "--state", "/tmp/state.json"])
    assert args.command == "dream"


def test_dream_once_skips_when_locked(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    traces_dir = tmp_path / "traces"

    def _unexpected_call(**_kwargs):
        raise AssertionError("run_async_route_pg should not run when lock is held and skip-if-locked is true")

    monkeypatch.setattr(cli_module, "run_async_route_pg", _unexpected_call)

    with state_write_lock(state_path, command_hint="test"):
        code = cli_module.main(
            [
                "dream",
                "--state",
                str(state_path),
                "--once",
                "--apply",
                "--skip-if-locked",
                "--traces-dir",
                str(traces_dir),
                "--json",
            ]
        )

    assert code == 0
