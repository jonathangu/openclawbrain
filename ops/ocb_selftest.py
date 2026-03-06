#!/usr/bin/env python3
"""ocb_selftest.py - quick smoke test for daemon/socket + journal update
Usage: python3 ops/ocb_selftest.py --state ~/.openclawbrain/main/state.json --socket /Users/guclaw/.openclawbrain/main/daemon.sock
"""
from pathlib import Path
import json
import socket
import time
import argparse

class SimpleOCBClient:
    def __init__(self, socket_path, timeout=10):
        self.sock_path = socket_path
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.sock_path)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def request(self, method, params):
        req = {"id": f"test-{int(time.time()*1000)}", "method": method, "params": params}
        self.sock.sendall((json.dumps(req) + "\n").encode('utf-8'))
        data = b''
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b'\n' in data:
                break
        line, _, _ = data.partition(b'\n')
        return json.loads(line.decode('utf-8'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True)
    p.add_argument('--socket', default=None)
    args = p.parse_args()

    state = Path(args.state).expanduser()
    if args.socket is None:
        sock = state.parent / 'daemon.sock'
    else:
        sock = Path(args.socket)
    if not sock.exists():
        raise SystemExit(f"socket not found: {sock}")

    client = SimpleOCBClient(str(sock))
    client.connect()
    try:
        query_text = f"ocb_selftest ping {int(time.time())}"
        resp = client.request('query', {'query': query_text, 'top_k': 1})
        print('daemon response OK, fired nodes:', resp.get('fired_nodes'))
    finally:
        client.close()

if __name__ == '__main__':
    main()
