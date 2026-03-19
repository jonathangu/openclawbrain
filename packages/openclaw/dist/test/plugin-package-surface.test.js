import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageJson = JSON.parse(readFileSync(path.join(__dirname, "..", "..", "package.json"), "utf8"));
const indexSource = readFileSync(path.join(__dirname, "..", "src", "index.js"), "utf8");

test("plugin package only exposes runtime surface", () => {
  assert.equal(packageJson.name, "@openclawbrain/openclaw");
  assert.equal("bin" in packageJson, false);
  assert.match(indexSource, /compileRuntimeContext/);
  assert.match(indexSource, /recordOpenClawProfileRuntimeLoadProof/);
  assert.doesNotMatch(indexSource, /bootstrapRuntimeAttach/);
  assert.doesNotMatch(indexSource, /describeCurrentProfileBrainStatus/);
  assert.doesNotMatch(indexSource, /runDaemonCommand/);
});
