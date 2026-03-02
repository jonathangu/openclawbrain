#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV_DIR="$HOME/.openclaw/venvs/openclawbrain"
CONFIG="$HOME/.openclaw/openclaw.json"
ENV_FILE="$HOME/.openclaw/credentials/env/openclawbrain.env"
ROOT="$HOME/.openclawbrain"
EMBED_MODEL="BAAI/bge-large-en-v1.5"
TEACHER_MODEL="gpt-5-mini"
SINCE_HOURS="168"

usage() {
  cat <<'USAGE'
Usage: default_experience.sh

Runs the complete default brain-building pipeline for every agent listed in
~/.openclaw/openclaw.json.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "error: missing config: $CONFIG" >&2
  exit 1
fi

mkdir -p "$ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

PY_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
OCB_BIN="$VENV_DIR/bin/openclawbrain"

"$PY_BIN" -m pip install --upgrade pip wheel
"$PY_BIN" -m pip install -e "$REPO_ROOT[openai]"

if [[ ! -x "$OCB_BIN" ]]; then
  echo "error: openclawbrain CLI not found in venv: $OCB_BIN" >&2
  exit 1
fi

AGENT_ROWS="$($PY_BIN - <<'PY'
import json
import os
import sys

config = os.path.expanduser("~/.openclaw/openclaw.json")
with open(config, "r", encoding="utf-8") as f:
    data = json.load(f)

agents = data.get("agents", {}).get("list", [])
if not isinstance(agents, list):
    sys.exit(0)

for agent in agents:
    if not isinstance(agent, dict):
        continue
    agent_id = agent.get("id")
    workspace = agent.get("workspace")
    if not agent_id or not workspace:
        continue
    print(f"{agent_id}\t{workspace}")
PY
)"

if [[ -z "$AGENT_ROWS" ]]; then
  echo "error: no agents found in $CONFIG" >&2
  exit 1
fi

backup_if_exists() {
  local path="$1"
  local ts="$2"
  if [[ -e "$path" ]]; then
    local backup="${path}.bak.${ts}"
    cp -p "$path" "$backup"
  fi
}

FAILED=0

while IFS=$'\t' read -r AGENT_ID WORKSPACE_DIR; do
  if [[ -z "$AGENT_ID" || -z "$WORKSPACE_DIR" ]]; then
    continue
  fi

  TS="$(date '+%Y%m%d-%H%M%S')"
  AGENT_DIR="$ROOT/$AGENT_ID"
  STATE="$AGENT_DIR/state.json"
  SCRATCH="$AGENT_DIR/scratch"
  SESSIONS="$HOME/.openclaw/agents/$AGENT_ID/sessions"
  TRACE_OUT="$SCRATCH/route_traces.jsonl"
  ROUTE_MODEL_OUT="$AGENT_DIR/route_model.npz"
  STATE_BACKUP="$SCRATCH/state.pre-default-experience.${TS}.json"
  LOG="$SCRATCH/default-experience.${TS}.log"
  STATUS_BEFORE="$SCRATCH/default-experience.${TS}.status_before.json"
  STATUS_AFTER="$SCRATCH/default-experience.${TS}.status_after.json"
  MAINTAIN_JSON="$SCRATCH/default-experience.${TS}.maintain.json"
  ASYNC_ROUTE_JSON="$SCRATCH/default-experience.${TS}.async-route-pg.json"
  TRAIN_ROUTE_JSON="$SCRATCH/default-experience.${TS}.train-route-model.json"
  MANIFEST="$SCRATCH/default-experience.${TS}.manifest.json"

  mkdir -p "$SCRATCH"

  (
    exec > >(tee -a "$LOG") 2>&1
    echo "== OpenClawBrain default experience =="
    echo "agent: $AGENT_ID"
    echo "workspace: $WORKSPACE_DIR"
    echo "state: $STATE"
    echo "sessions: $SESSIONS"
    echo "timestamp: $(date -Is)"

    if [[ ! -f "$STATE" ]]; then
      echo "error: missing state file: $STATE" >&2
      exit 2
    fi
    if [[ ! -e "$SESSIONS" ]]; then
      echo "error: missing sessions path: $SESSIONS" >&2
      exit 2
    fi

    echo
    echo "== Status (before) =="
    "$OCB_BIN" status \
      --state "$STATE" \
      --json > "$STATUS_BEFORE"
    echo "saved: $STATUS_BEFORE"

    backup_if_exists "$TRACE_OUT" "$TS"
    backup_if_exists "$ROUTE_MODEL_OUT" "$TS"
    cp -p "$STATE" "$STATE_BACKUP"

    echo
    echo "== 1) Re-embed (local BGE-large) =="
    "$OCB_BIN" reembed \
      --state "$STATE" \
      --embedder local \
      --embed-model "$EMBED_MODEL"

    echo
    echo "== 2) Replay (full, include tool results) =="
    "$OCB_BIN" replay \
      --state "$STATE" \
      --sessions "$SESSIONS" \
      --mode full \
      --include-tool-results \
      --progress-every 250 \
      --checkpoint-every-seconds 60 \
      --workers 4 \
      --replay-workers 1

    echo
    echo "== 3) Maintain (structural tasks) =="
    "$OCB_BIN" maintain \
      --state "$STATE" \
      --tasks health,decay,scale,split,merge,prune,connect \
      --llm none \
      --embedder local \
      --json > "$MAINTAIN_JSON"
    echo "saved: $MAINTAIN_JSON"

    echo
    echo "== 4) Async route teacher labeling =="
    "$OCB_BIN" async-route-pg \
      --state "$STATE" \
      --teacher openai \
      --teacher-model "$TEACHER_MODEL" \
      --since-hours "$SINCE_HOURS" \
      --traces-out "$TRACE_OUT" \
      --apply \
      --json > "$ASYNC_ROUTE_JSON"
    echo "saved: $ASYNC_ROUTE_JSON"

    echo
    echo "== 5) Train route model =="
    "$OCB_BIN" train-route-model \
      --state "$STATE" \
      --traces-in "$TRACE_OUT" \
      --out "$ROUTE_MODEL_OUT" \
      --json > "$TRAIN_ROUTE_JSON"
    echo "saved: $TRAIN_ROUTE_JSON"

    echo
    echo "== Status (after) =="
    "$OCB_BIN" status \
      --state "$STATE" \
      --json > "$STATUS_AFTER"
    echo "saved: $STATUS_AFTER"

    "$PY_BIN" - <<PY
import json

manifest = {
    "agent_id": "$AGENT_ID",
    "state_path": "$STATE",
    "sessions_path": "$SESSIONS",
    "log_path": "$LOG",
    "embed_model": "$EMBED_MODEL",
    "teacher_model": "$TEACHER_MODEL",
    "since_hours": "$SINCE_HOURS",
    "artifacts": {
        "state_backup": "$STATE_BACKUP",
        "trace_out": "$TRACE_OUT",
        "route_model_out": "$ROUTE_MODEL_OUT",
        "maintain_json": "$MAINTAIN_JSON",
        "async_route_pg_json": "$ASYNC_ROUTE_JSON",
        "train_route_model_json": "$TRAIN_ROUTE_JSON",
        "status_before_json": "$STATUS_BEFORE",
        "status_after_json": "$STATUS_AFTER",
        "manifest_path": "$MANIFEST",
    },
}

with open("$STATUS_BEFORE", "r", encoding="utf-8") as f:
    manifest["status_before"] = json.load(f)
with open("$STATUS_AFTER", "r", encoding="utf-8") as f:
    manifest["status_after"] = json.load(f)

with open("$MANIFEST", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
PY
    echo "saved: $MANIFEST"

    echo
    echo "Done. Log: $LOG"
  ) || FAILED=1

done <<< "$AGENT_ROWS"

exit "$FAILED"
