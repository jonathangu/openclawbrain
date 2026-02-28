#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required"
  exit 1
fi

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 --brain-name <name> --state-path <path>"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --brain-name)
      BRAIN_NAME="${2:-}"
      shift 2
      ;;
    --state-path)
      STATE_PATH="${2:-}"
      shift 2
      ;;
    *)
      echo "Usage: $0 --brain-name <name> --state-path <path>"
      exit 1
      ;;
  esac
done

if [[ -z "${BRAIN_NAME:-}" || -z "${STATE_PATH:-}" ]]; then
  echo "Usage: $0 --brain-name <name> --state-path <path>"
  exit 1
fi

LAUNCHD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$HOME/Library/LaunchAgents"
PYTHON_PATH=$(python3 -c 'import sys; print(sys.executable)')
STATE_PATH="${STATE_PATH/#\~/$HOME}"
mkdir -p "$TARGET_DIR"

TEMPLATE="$LAUNCHD_DIR/com.openclawbrain.main.plist"
PLIST_NAME="com.openclawbrain.${BRAIN_NAME}.plist"
DST="$TARGET_DIR/$PLIST_NAME"

tmp_file="$(mktemp)"
python3 - <<'PY' "$TEMPLATE" "$tmp_file" "$OPENAI_API_KEY" "$PYTHON_PATH" "$BRAIN_NAME" "$STATE_PATH"
import pathlib
import sys

template, output_path, api_key, python_path, brain_name, state_path = sys.argv[1:]
text = pathlib.Path(template).read_text(encoding="utf-8")
text = text.replace("__REPLACE_ME__", api_key)
text = text.replace("__PYTHON_PATH__", python_path)
text = text.replace("__BRAIN_NAME__", brain_name)
text = text.replace("__STATE_PATH__", state_path)
pathlib.Path(output_path).write_text(text, encoding="utf-8")
PY
cp "$tmp_file" "$DST"
rm -f "$tmp_file"
launchctl load "$DST"
echo "loaded: $DST"

if launchctl list "com.openclawbrain.${BRAIN_NAME}" >/dev/null 2>&1; then
  echo "running: com.openclawbrain.${BRAIN_NAME}"
else
  echo "not running: com.openclawbrain.${BRAIN_NAME}"
  exit 1
fi
