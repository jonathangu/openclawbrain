import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CONTRACT_IDS,
  FIXTURE_ARTIFACT_MANIFEST,
  FIXTURE_PACK_GRAPH,
  FIXTURE_PACK_VECTORS,
  FIXTURE_ROUTER_ARTIFACT,
  canonicalJson
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
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-fixture",
    userMessage: "Run the scanner with qwen checkpoints.",
    maxContextBlocks: 1,
    modeRequested: "heuristic",
    runtimeHints: ["feedback scanner"]
  });

  assert.equal(response.selectedContext[0]?.id, "ctx-feedback-scanner");
  assert.equal(response.diagnostics.modeEffective, "learned");
  assert.equal(response.diagnostics.usedLearnedRouteFn, true);
  assert.equal(response.diagnostics.routerIdentity, FIXTURE_ROUTER_ARTIFACT.routerIdentity);
  assert.equal(response.diagnostics.selectionStrategy, "pack_keyword_overlap_v1");
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
  assert.equal(response.diagnostics.compactionApplied, false);
  assert.equal(response.diagnostics.candidateCount, FIXTURE_PACK_GRAPH.blocks.length);
});

test("compileRuntime applies native structural compaction under a character budget", (t) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  const response = compileRuntime(rootDir, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-compact",
    userMessage: "feedback scanner manifest structural compaction context",
    maxContextBlocks: 3,
    maxContextChars: 180,
    modeRequested: "heuristic",
    compactionMode: "native",
    runtimeHints: ["pack-backed selection"]
  });

  assert.equal(response.diagnostics.compactionApplied, true);
  assert.equal(response.diagnostics.compactionMode, "native");
  assert.equal(response.diagnostics.selectedCharCount <= 180, true);
  assert.equal(response.selectedContext.some((block) => (block.compactedFrom?.length ?? 0) > 1), true);
  assert.match(response.diagnostics.notes.join(";"), /native_structural_compaction=applied/);
});

test("pack load rejects tampered graph payloads", (t) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-pack-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  writePackFile(rootDir, PACK_LAYOUT.graph, FIXTURE_PACK_GRAPH);
  writePackFile(rootDir, PACK_LAYOUT.vectors, FIXTURE_PACK_VECTORS);
  writePackFile(rootDir, PACK_LAYOUT.router, FIXTURE_ROUTER_ARTIFACT);
  writePackFile(rootDir, PACK_LAYOUT.manifest, FIXTURE_ARTIFACT_MANIFEST);

  writeFileSync(
    path.join(rootDir, PACK_LAYOUT.graph),
    canonicalJson({
      ...FIXTURE_PACK_GRAPH,
      blocks: FIXTURE_PACK_GRAPH.blocks.map((block, index) =>
        index === 0 ? { ...block, text: `${block.text} tampered` } : block
      )
    }),
    "utf8"
  );

  assert.throws(() => loadPackForCompile(rootDir), /graph checksum does not match manifest/);
});
