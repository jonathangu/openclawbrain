#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

function readTruthyEnvFlag(name) {
  const value = process.env[name]?.trim().toLowerCase();
  return value === "1" || value === "true" || value === "yes" || value === "on";
}

function buildCompatibilityGuardMessage() {
  return [
    "[openclawbrain] STOP: this binary comes from the retired compatibility package @jonathangu/openclawbrain.",
    "Do not use it for modern install/repair/status flows.",
    "",
    "Canonical OpenClawBrain path:",
    "  openclawbrain install --openclaw-home <path>",
    "  openclaw gateway restart",
    "  openclawbrain status --openclaw-home <path> --detailed",
    "  openclawbrain proof --openclaw-home <path>",
    "",
    "If your shell still resolves `openclawbrain` to this compatibility package, remove the old global package/binary from PATH first.",
    "Maintainer-only escape hatch: set OPENCLAWBRAIN_COMPAT_ALLOW_LEGACY_CLI=1 to run the legacy compatibility CLI intentionally.",
  ].join("\n");
}

if (!readTruthyEnvFlag("OPENCLAWBRAIN_COMPAT_ALLOW_LEGACY_CLI")) {
  process.stderr.write(`${buildCompatibilityGuardMessage()}\n`);
  process.exit(1);
}

const here = dirname(fileURLToPath(import.meta.url));
const entry = join(here, "../src/brain-cli.ts");
const result = spawnSync(
  process.execPath,
  ["--import", "tsx/esm", entry, ...process.argv.slice(2)],
  { stdio: "inherit" },
);

process.exit(result.status ?? 1);
