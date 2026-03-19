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
const packageDir = resolve(repoRoot, "packages", "openclaw");
const tempDir = mkdtempSync(join(tmpdir(), "openclaw-pack-verify-"));
const npmCacheDir = join(tempDir, "npm-cache");

let tarballPath = null;

const npmPackEnv = {
  ...process.env,
  NPM_CONFIG_CACHE: process.env.NPM_CONFIG_CACHE ?? npmCacheDir,
  NPM_CONFIG_DRY_RUN: "false",
};

delete npmPackEnv.npm_config_dry_run;
delete npmPackEnv.npm_config_json;

function extractJsonFromTarball(entryPath) {
  const contents = execFileSync("tar", ["-xOf", tarballPath, entryPath], {
    encoding: "utf8",
  });
  return JSON.parse(contents);
}

try {
  const packOutput = execFileSync("npm", ["pack", "--json", packageDir], {
    cwd: tempDir,
    encoding: "utf8",
    env: npmPackEnv,
  });
  const [packResult] = JSON.parse(packOutput);

  assert(packResult, "npm pack did not return a tarball result");
  assert.equal(packResult.name, "@openclawbrain/openclaw");

  tarballPath = join(tempDir, packResult.filename);

  const tarballFiles = new Set(packResult.files.map((file) => file.path));
  const packageJson = extractJsonFromTarball("package/package.json");
  const pluginManifest = extractJsonFromTarball("package/openclaw.plugin.json");

  assert.equal(packageJson.name, "@openclawbrain/openclaw");
  assert.equal(packageJson.bin, undefined, "plugin package must not publish CLI bins");

  const extensionEntries = packageJson.openclaw?.extensions;
  assert(Array.isArray(extensionEntries), "package.json missing openclaw.extensions array");
  assert.notEqual(extensionEntries.length, 0, "package.json openclaw.extensions must not be empty");
  assert(
    extensionEntries.includes("./dist/extension/index.js"),
    `package.json openclaw.extensions must include ./dist/extension/index.js; received ${JSON.stringify(extensionEntries)}`,
  );

  for (const requiredFile of [
    "dist/extension/index.js",
    "dist/extension/runtime-guard.js",
    "dist/src/index.js",
    "dist/src/runtime-core.js",
    "dist/src/attachment-truth.js",
    "openclaw.plugin.json",
    "package.json",
  ]) {
    assert(tarballFiles.has(requiredFile), `packed tarball is missing ${requiredFile}`);
  }

  for (const forbiddenFile of [
    "dist/src/cli.js",
    "dist/src/daemon.js",
    "dist/src/import-export.js",
    "dist/src/openclaw-home-layout.js",
    "dist/src/openclaw-hook-truth.js",
    "dist/src/openclaw-plugin-install.js",
    "dist/src/provider-config.js",
    "dist/src/resolve-activation-root.js",
    "dist/src/session-store.js",
    "dist/src/session-tail.js",
    "dist/src/local-session-passive-learning.js",
    "dist/src/shadow-extension-proof.js",
  ]) {
    assert(!tarballFiles.has(forbiddenFile), `plugin tarball must not include ${forbiddenFile}`);
  }

  assert.equal(pluginManifest.id, "openclawbrain");
  assert.equal(
    pluginManifest.version,
    packageJson.version,
    `openclaw.plugin.json version ${pluginManifest.version} must match package.json version ${packageJson.version}`,
  );

  console.log(
    JSON.stringify(
      {
        ok: true,
        packageName: packageJson.name,
        version: packageJson.version,
        extensionEntries,
        verifiedFiles: Array.from(tarballFiles)
          .filter((file) => [
            "dist/extension/index.js",
            "dist/extension/runtime-guard.js",
            "dist/src/index.js",
            "dist/src/runtime-core.js",
            "dist/src/attachment-truth.js",
            "openclaw.plugin.json",
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
