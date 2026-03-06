#!/usr/bin/env python3
# Backfill script moved into repo; original code is the full backfill implementation.
# (Copied from assistant-generated runtime prototype.)

# -- BEGIN IMPLEMENTATION --

import argparse
import json
import socket
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Minimal NDJSON client for openclawbrain daemon socket (no external deps)

def parse_iso_ts(s: str) -> float:
    if not s:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0


def extract_text_from_message(msg_obj):
    content = msg_obj.get('content')
    if not isinstance(content, list):
        return None
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'text' and isinstance(item.get('text'), str):
            parts.append(item.get('text'))
    return ' '.join(parts).strip() if parts else None


class SimpleOCBClient:
    def __init__(self, socket_path, timeout=30):
        self.sock_path = socket_path
        self.timeout = timeout
        self.sock = None
        self.id_counter = 0

    def connect(self):
        if self.sock:
            return
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.sock_path)

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def request(self, method, params):
        self.connect()
        self.id_counter += 1
        req = {"id": f"req-{int(time.time()*1000)}-{self.id_counter}", "method": method, "params": params}
        line = json.dumps(req, separators=(',', ':')) + '\n'
        self.sock.sendall(line.encode('utf-8'))
        data = b''
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b'\n' in data:
                break
        line, _, _ = data.partition(b'\n')
        if not line:
            raise RuntimeError('no response from daemon')
        resp = json.loads(line.decode('utf-8'))
        if 'error' in resp and resp['error'] is not None:
            raise RuntimeError(f"daemon error {resp['error']}")
        return resp.get('result')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True)
    p.add_argument('--sessions-dir', default='/Users/guclaw/.openclaw/agents/main/sessions')
    p.add_argument('--since-hours', type=float, default=1000000)
    p.add_argument('--max-queries', type=int, default=2000)
    p.add_argument('--socket', default='/Users/guclaw/.openclawbrain/main/daemon.sock')
    p.add_argument('--posted-keys', default=None, help='Path to append-only JSONL of posted keys (dedupe across runs)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    sessions_dir = Path(args.sessions_dir)
    if not sessions_dir.exists():
        print('sessions dir not found', sessions_dir)
        return

    cutoff = time.time() - max(0.0, float(args.since_hours)) * 3600.0

    posted_keys_path = Path(args.posted_keys).expanduser() if args.posted_keys else (Path(args.state).expanduser().parent / 'scratch' / 'backfill_posted_keys.jsonl')
    posted_keys_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if posted_keys_path.exists():
        try:
            for raw in posted_keys_path.read_text(encoding='utf-8').splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                key = row.get('key')
                if isinstance(key, str) and key:
                    existing.add(key)
        except Exception:
            pass

    candidates = []
    files = sorted([p for p in sessions_dir.iterdir() if p.is_file() and p.suffix == '.jsonl'], key=lambda p: p.stat().st_mtime, reverse=True)
    seen = set()
    for fp in files:
        try:
            text = fp.read_text(encoding='utf-8')
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get('type') != 'message':
                continue
            msg = entry.get('message') or {}
            role = msg.get('role')
            if role != 'user':
                continue
            ts_raw = entry.get('timestamp') or msg.get('timestamp')
            ts = parse_iso_ts(str(ts_raw)) if ts_raw else 0.0
            if ts < cutoff:
                continue
            qtext = extract_text_from_message(msg)
            if not qtext:
                continue
            session_id = fp.stem
            stable_key_payload = f"{session_id}|{int(ts)}|{qtext.strip()}".encode('utf-8', errors='ignore')
            key = hashlib.sha256(stable_key_payload).hexdigest()
            if key in seen or key in existing:
                continue
            seen.add(key)
            candidates.append({'key': key, 'text': qtext, 'ts': ts, 'chat_id': f'session:{session_id}'})
            if len(candidates) >= args.max_queries:
                break
        if len(candidates) >= args.max_queries:
            break

    if not candidates:
        print('No recent user messages found to populate journal.')
        return

    print(f'Found {len(candidates)} candidates; socket={args.socket}; dry_run={args.dry_run}; posted_keys={posted_keys_path}')
    if args.dry_run:
        for c in candidates:
            print('[dry] ', c['key'][:12], c['chat_id'], datetime.fromtimestamp(c['ts'], timezone.utc).isoformat(), c['text'][:120])
        return

    client = SimpleOCBClient(args.socket)
    posted = 0
    errors = []
    try:
        client.connect()
        with posted_keys_path.open('a', encoding='utf-8') as out:
            for c in candidates:
                params = {
                    'query': c['text'],
                    'top_k': 5,
                    'chat_id': c['chat_id'],
                    'route_mode': 'edge+sim',
                    'max_prompt_context_chars': 12000,
                    'assert_learned': False,
                }
                try:
                    client.request('query', params)
                    posted += 1
                    out.write(json.dumps({
                        'key': c['key'],
                        'chat_id': c['chat_id'],
                        'ts_int': int(c['ts']),
                        'chars': len(c['text'] or ''),
                        'wrote_at': time.time(),
                    }) + '\n')
                    existing.add(c['key'])
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                time.sleep(0.05)
    finally:
        client.close()

    print(f'Posted {posted} queries to daemon (socket). Errors: {len(errors)}')
    if errors:
        for e in errors[:10]:
            print('- ', e)


if __name__ == '__main__':
    main()
