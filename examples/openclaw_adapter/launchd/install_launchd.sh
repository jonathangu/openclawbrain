#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required"
  exit 1
fi

LAUNCHD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$TARGET_DIR"

PLISTS=(
  "com.openclawbrain.main.plist"
  "com.openclawbrain.pelican.plist"
  "com.openclawbrain.bountiful.plist"
)

for plist in "${PLISTS[@]}"; do
  src="$LAUNCHD_DIR/$plist"
  dst="$TARGET_DIR/$plist"

  tmp_file="$(mktemp)"
  sed \
    -e "s#__REPLACE_ME__#${OPENAI_API_KEY}#g" \
    -e "s#~#$HOME#g" \
    "$src" > "$tmp_file"
  cp "$tmp_file" "$dst"
  rm -f "$tmp_file"
  launchctl load "$dst"
  echo "loaded: $dst"

done

echo
for plist in "${PLISTS[@]}"; do
  label="${plist%.plist}"
  if launchctl list "$label" >/dev/null 2>&1; then
    echo "running: $label"
  else
    echo "not running: $label"
    exit 1
  fi
done
