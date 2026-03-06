#!/usr/bin/env python3
"""Simple orchestrator: run backfill -> async-route-pg (dry-run) -> write traces_out path
Usage: python3 ops/harvest_pipeline.py --state ~/.openclawbrain/main/state.json --agent main
"""
import argparse
import subprocess
from pathlib import Path
import shlex
import time

parser = argparse.ArgumentParser()
parser.add_argument('--state', required=True)
parser.add_argument('--agent', default='main')
parser.add_argument('--max-queries', default='200')
parser.add_argument('--traces-out', default=None)
parser.add_argument('--labels-out', default=None)
args = parser.parse_args()

state = Path(args.state).expanduser()
scratch = state.parent / 'scratch'
scratch.mkdir(parents=True, exist_ok=True)
traces_out = Path(args.traces_out) if args.traces_out else scratch / f'traces_harvest_{int(time.time())}.jsonl'
labels_out = Path(args.labels_out) if args.labels_out else scratch / f'labels_harvest_{int(time.time())}.jsonl'

# 1) Backfill (safe, dry-run first)
print('Backfill: dry-run...')
cmd = f'python3 ops/backfill_journal.py --state {shlex.quote(str(state))} --sessions-dir ~/.openclaw/agents/{args.agent}/sessions --max-queries 200 --dry-run'
subprocess.run(cmd, shell=True, check=False)

print('Backfill: running...')
cmd = f'python3 ops/backfill_journal.py --state {shlex.quote(str(state))} --sessions-dir ~/.openclaw/agents/{args.agent}/sessions --max-queries {args.max_queries} --socket {shlex.quote(str(state.parent / "daemon.sock"))}'
subprocess.run(cmd, shell=True, check=True)

# 2) Run async-route-pg dry-run to produce traces
print('Running async-route-pg (dry-run) to generate traces...')
cmd = (
    f'OCB_STATE_LOCK_FORCE=1 OPENCLAWBRAIN_STATE_LOCK_FORCE=1 '
    f'openclawbrain async-route-pg --state {shlex.quote(str(state))} '
    f'--since-hours 168 --max-queries {args.max_queries} --sample-rate 1.0 --include-query-vector '
    f'--traces-out {shlex.quote(str(traces_out))} --labels-out {shlex.quote(str(labels_out))} --teacher none'
)
subprocess.run(cmd, shell=True, check=True)

print('Harvest done. traces_out=', traces_out, 'labels_out=', labels_out)
