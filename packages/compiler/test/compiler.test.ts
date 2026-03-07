import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CONTRACT_IDS,
  FIXTURE_ARTIFACT_MANIFEST,
  FIXTURE_PACK_GRAPH,
  FIXTURE_PACK_VECTORS,
  FIXTURE_ROUTER_ARTIFACT,
  FIXTURE_RUNTIME_COMPILE_REQUEST
} from "@openclawbrain/contracts";
import { compileRuntime, determineRouteMode, loadPackForCompile } from "@openclawbrain/compiler";
import { PACK_LAYOUT, writePackFile } from "@openclawbrain/pack-format";

function materializeFixturePack(rootDir: string): void {
  writePackFile(rootDir, PACK_LAYOUT.graph, FIXTURE_PACK_GRAPH);
  writePackFile(rootDir, PACK_LAYOUT.vectors, FIXTURE_PACK_VECTORS);
  writePackFile(rootDir, PACK_LAYOUT.router, FIXTURE_ROUTER_ARTIFACT);
  writePackFile(rootDir, PACK_LAYOUT.manifest, FIXTURE_ARTIFACT_MANIFEST);
}

test("learned-required packs force learned mode and select scanner context", (t) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  const pack = loadPackForCompile(rootDir);

  assert.equal(determineRouteMode(pack, "heuristic"), "learned");

  const response = compileRuntime(pack, {
    ...FIXTURE_RUNTIME_COMPILE_REQUEST,
    contract: CONTRACT_IDS.runtimeCompile,
    userMessage: "Run the scanner with qwen checkpoints.",
    maxContextBlocks: 1,
    runtimeHints: ["feedback scanner"]
  });

  assert.equal(response.selectedContext[0]?.id, "ctx-feedback-scanner");
  assert.equal(response.diagnostics.modeEffective, "learned");
  assert.equal(response.diagnostics.usedLearnedRouteFn, true);
  assert.equal(response.diagnostics.routerIdentity, FIXTURE_ROUTER_ARTIFACT.routerIdentity);
});

test("compileRuntime falls back to priority order when nothing matches", (t) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  const response = compileRuntime(rootDir, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-fallback",
    userMessage: "zzzz qqqq",
    maxContextBlocks: 1,
    modeRequested: "heuristic",
    runtimeHints: []
  });

  assert.equal(response.packId, FIXTURE_ARTIFACT_MANIFEST.packId);
  assert.equal(response.selectedContext[0]?.id, "ctx-feedback-scanner");
  assert.match(response.diagnostics.notes[1] ?? "", /priority_fallback/);
});
