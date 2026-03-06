# Backfill & Harvest runbook

This document explains how to run backfill -> harvest -> training tasks safely on a Mac mini running OpenClaw.

Quickstart

1. Dry-run:

```bash
python3 ops/backfill_journal.py --state ~/.openclawbrain/main/state.json --sessions-dir ~/.openclaw/agents/main/sessions --max-queries 200 --dry-run
```

2. Run backfill:

```bash
python3 ops/backfill_journal.py --state ~/.openclawbrain/main/state.json --sessions-dir ~/.openclaw/agents/main/sessions --max-queries 200 --socket ~/.openclawbrain/main/daemon.sock
```

3. Harvest traces (dry-run + run):

```bash
python3 ops/harvest_pipeline.py --state ~/.openclawbrain/main/state.json --agent main --max-queries 200
```

Notes

- Use `--posted-keys` to share posted keys path if you coordinate across machines.
- Teacher labeling requires `OPENAI_API_KEY` in environment. Use small batches and set a cost cap.
- All artifacts are written to `~/.openclawbrain/main/scratch/` for auditing.
