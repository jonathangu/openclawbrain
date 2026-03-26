#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");
const packageJsonPath = resolve(repoRoot, "packages", "openclaw", "package.json");
const pluginManifestPath = resolve(repoRoot, "packages", "openclaw", "openclaw.plugin.json");

const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
const pluginManifest = JSON.parse(readFileSync(pluginManifestPath, "utf8"));

if (typeof packageJson.version !== "string" || packageJson.version.trim().length === 0) {
  throw new Error(`packages/openclaw/package.json is missing a valid version: ${JSON.stringify(packageJson.version)}`);
}

if (pluginManifest.version === packageJson.version) {
  process.exit(0);
}

pluginManifest.version = packageJson.version;
writeFileSync(pluginManifestPath, `${JSON.stringify(pluginManifest, null, 2)}\n`, "utf8");
