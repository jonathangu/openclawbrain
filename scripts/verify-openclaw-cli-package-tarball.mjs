#!/usr/bin/env node

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");
const packageDir = resolve(repoRoot, "packages", "cli");
const tempDir = mkdtempSync(join(tmpdir(), "openclaw-cli-pack-verify-"));
const npmCacheDir = join(tempDir, "npm-cache");

let tarballPath = null;

const npmPackEnv = {
  ...process.env,
  NPM_CONFIG_CACHE: process.env.NPM_CONFIG_CACHE ?? npmCacheDir,
  NPM_CONFIG_DRY_RUN: "false",
};

delete npmPackEnv.npm_config_dry_run;
delete npmPackEnv.npm_config_json;

try {
  const packOutput = execFileSync("npm", ["pack", "--json", packageDir], {
    cwd: tempDir,
    encoding: "utf8",
    env: npmPackEnv,
  });
  const [packResult] = JSON.parse(packOutput);

  assert(packResult, "npm pack did not return a tarball result");
  assert.equal(packResult.name, "@openclawbrain/cli");

  tarballPath = join(tempDir, packResult.filename);

  const tarballFiles = new Set(packResult.files.map((file) => file.path));
  const packageJson = JSON.parse(
    execFileSync("tar", ["-xOf", tarballPath, "package/package.json"], {
      encoding: "utf8",
    }),
  );

  assert.equal(packageJson.name, "@openclawbrain/cli");
  assert.equal(packageJson.openclaw, undefined, "cli package must not publish an OpenClaw plugin manifest");
  assert.equal(packageJson.bin?.openclawbrain, "dist/src/cli.js");
  assert.equal(packageJson.bin?.["openclawbrain-ops"], "dist/src/cli.js");

  for (const requiredFile of [
    "dist/src/index.js",
    "dist/src/index.d.ts",
    "dist/src/cli.js",
    "dist/src/daemon.js",
    "dist/src/import-export.js",
    "dist/src/attachment-policy-truth.js",
    "dist/src/attachment-truth.d.ts",
    "dist/src/attachment-truth.js",
    "dist/src/learning-spine.js",
    "dist/src/runtime-core.js",
    "dist/src/teacher-decision-match.js",
    "dist/src/traced-learning-bridge.js",
    "dist/extension/runtime-guard.js",
    "extension/index.ts",
    "extension/runtime-guard.ts",
    "dist/src/openclaw-home-layout.js",
    "dist/src/openclaw-hook-truth.js",
    "dist/src/openclaw-plugin-install.js",
    "dist/src/ollama-client.js",
    "dist/src/provider-config.js",
    "dist/src/resolve-activation-root.js",
    "dist/src/semantic-metadata.js",
    "dist/src/session-store.js",
    "dist/src/session-tail.js",
    "dist/src/local-session-passive-learning.js",
    "dist/src/teacher-labeler.js",
    "package.json",
  ]) {
    assert(tarballFiles.has(requiredFile), `cli tarball is missing ${requiredFile}`);
  }

  for (const forbiddenFile of ["openclaw.plugin.json"]) {
    assert(!tarballFiles.has(forbiddenFile), `cli tarball must not include ${forbiddenFile}`);
  }

  console.log(
    JSON.stringify(
      {
        ok: true,
        packageName: packageJson.name,
        version: packageJson.version,
        verifiedFiles: Array.from(tarballFiles)
          .filter((file) => [
            "dist/src/index.js",
            "dist/src/index.d.ts",
            "dist/src/cli.js",
            "dist/src/daemon.js",
            "dist/src/import-export.js",
            "dist/src/attachment-policy-truth.js",
            "dist/src/attachment-truth.d.ts",
            "dist/src/attachment-truth.js",
            "dist/src/learning-spine.js",
            "dist/src/runtime-core.js",
            "dist/src/teacher-decision-match.js",
            "dist/src/traced-learning-bridge.js",
            "dist/extension/runtime-guard.js",
            "extension/index.ts",
            "extension/runtime-guard.ts",
            "dist/src/openclaw-home-layout.js",
            "dist/src/openclaw-hook-truth.js",
            "dist/src/openclaw-plugin-install.js",
            "dist/src/ollama-client.js",
            "dist/src/provider-config.js",
            "dist/src/resolve-activation-root.js",
            "dist/src/semantic-metadata.js",
            "dist/src/session-store.js",
            "dist/src/session-tail.js",
            "dist/src/local-session-passive-learning.js",
            "dist/src/teacher-labeler.js",
            "package.json",
          ].includes(file))
          .sort(),
        tarballPath,
      },
      null,
      2,
    ),
  );
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}
