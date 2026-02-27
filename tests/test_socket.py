from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openclawbrain.socket_client import OCBClient


def _write_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "a.md").write_text("alpha memory", encoding="utf-8")
    (path / "b.md").write_text("beta knowledge", encoding="utf-8")


def _build_state(workspace: Path) -> Path:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    output = workspace.parent / "state"
    output.mkdir(parents=True, exist_ok=True)

    init = [
        sys.executable,
        "-m",
        "openclawbrain",
        "init",
        "--workspace",
        str(workspace),
        "--output",
        str(output),
    ]
    subprocess.run(init, check=True, capture_output=True, text=True, env=env)
    return output / "state.json"


def _start_server(state_path: Path, socket_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openclawbrain.socket_server",
            "--state",
            str(state_path),
            "--socket-path",
            str(socket_path),
            "--auto-save-interval",
            "2",
            "--verbose",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )


def _wait_for_socket(path: Path, proc: subprocess.Popen, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    # Capture stderr for diagnostics
    stderr_text = ""
    if proc.stderr:
        import fcntl

        fd = proc.stderr.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        try:
            data = proc.stderr.read(16384)
            stderr_text = data.decode("utf-8", errors="replace") if data else ""
        except Exception:
            pass
    raise AssertionError(
        f"socket did not appear: {path}\n"
        f"server rc={proc.poll()}\n"
        f"server stderr:\n{stderr_text}"
    )


def test_socket_server_query_and_introspection(tmp_path: Path) -> None:
    workspace = tmp_path / "main"
    _write_workspace(workspace)
    state_path = _build_state(workspace)

    # Use a short path for the socket to avoid AF_UNIX 104-char limit on macOS
    sock_dir = Path(tempfile.mkdtemp(prefix="ocb"))
    socket_path = sock_dir / "d.sock"

    proc = _start_server(state_path=state_path, socket_path=socket_path)
    try:
        _wait_for_socket(socket_path, proc)

        with OCBClient(str(socket_path)) as client:
            query_response = client.query("alpha", chat_id="chat-1", top_k=2)
            assert query_response["fired_nodes"]

            health_response = client.health()
            assert "nodes" in health_response
            assert health_response["nodes"] >= 1

            info_response = client.info()
            assert "embedder" in info_response
            assert info_response["nodes"] >= 1
    finally:
        if proc.poll() is None:
            try:
                with OCBClient(str(socket_path)) as shutdown_client:
                    shutdown_client.request("shutdown", {})
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        else:
            proc.kill()
            proc.wait(timeout=5)
        # Clean up socket dir
        import shutil
        shutil.rmtree(sock_dir, ignore_errors=True)
