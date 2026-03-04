from __future__ import annotations

from openclawbrain import socket_server


def test_socket_server_parses_route_stop_flags() -> None:
    args = socket_server._parse_args(
        [
            "--state",
            "/tmp/state.json",
            "--route-enable-stop",
            "false",
            "--route-stop-margin",
            "0.1",
        ]
    )

    assert args.route_enable_stop == "false"
    assert args.route_stop_margin == 0.1
