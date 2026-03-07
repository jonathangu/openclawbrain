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
  type ArtifactManifestV1,
  type PackGraphPayloadV1,
  type PackVectorsPayloadV1,
  type RouterArtifactV1,
  canonicalJson
} from "@openclawbrain/contracts";
import {
  compileRuntime,
  compileRuntimeFromActivation,
  determineRouteMode,
  loadPackForCompile,
  resolveActivationCompileTarget
} from "@openclawbrain/compiler";
import {
  PACK_LAYOUT,
  activatePack,
  computePayloadChecksum,
  promoteCandidatePack,
  stageCandidatePack,
  writePackFile
} from "@openclawbrain/pack-format";

function materializeFixturePack(rootDir: string): void {
  writePackFile(rootDir, PACK_LAYOUT.graph, FIXTURE_PACK_GRAPH);
  writePackFile(rootDir, PACK_LAYOUT.vectors, FIXTURE_PACK_VECTORS);
  writePackFile(rootDir, PACK_LAYOUT.router, FIXTURE_ROUTER_ARTIFACT);
  writePackFile(rootDir, PACK_LAYOUT.manifest, FIXTURE_ARTIFACT_MANIFEST);
}

function materializeNamedPack(rootDir: string, options: { packId: string; learnedRouting: boolean }): void {
  const graph: PackGraphPayloadV1 = {
    ...FIXTURE_PACK_GRAPH,
    packId: options.packId
  };
  const vectors: PackVectorsPayloadV1 = {
    ...FIXTURE_PACK_VECTORS,
    packId: options.packId
  };
  const router: RouterArtifactV1 | null = options.learnedRouting
    ? {
        ...FIXTURE_ROUTER_ARTIFACT,
        routerIdentity: `${options.packId}:route_fn`
      }
    : null;
  const manifest: ArtifactManifestV1 = {
    ...FIXTURE_ARTIFACT_MANIFEST,
    packId: options.packId,
    routePolicy: options.learnedRouting ? "requires_learned_routing" : "heuristic_allowed",
    runtimeAssets: {
      graphPath: PACK_LAYOUT.graph,
      vectorPath: PACK_LAYOUT.vectors,
      router: options.learnedRouting
        ? {
            kind: "artifact",
            identity: router?.routerIdentity ?? null,
            artifactPath: PACK_LAYOUT.router
          }
        : {
            kind: "none",
            identity: null,
            artifactPath: null
          }
    },
    payloadChecksums: {
      graph: computePayloadChecksum(graph),
      vector: computePayloadChecksum(vectors),
      router: router === null ? null : computePayloadChecksum(router)
    }
  };

  writePackFile(rootDir, PACK_LAYOUT.graph, graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, vectors);
  if (router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, manifest);
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

test("compileRuntime rejects stale activePackId expectations", (t) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  assert.throws(
    () =>
      compileRuntime(rootDir, {
        contract: CONTRACT_IDS.runtimeCompile,
        agentId: "agent-stale-pack",
        userMessage: "Run the scanner with qwen checkpoints.",
        maxContextBlocks: 1,
        modeRequested: "heuristic",
        activePackId: "pack-stale-runtime-view",
        runtimeHints: ["feedback scanner"]
      }),
    /activePackId pack-stale-runtime-view does not match loaded pack/
  );
});

test("compileRuntimeFromActivation serves the active pack and respects promotion", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-compile-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-pack-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-pack-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  materializeNamedPack(activeRoot, {
    packId: "pack-active-serving",
    learnedRouting: false
  });
  materializeNamedPack(candidateRoot, {
    packId: "pack-candidate-serving",
    learnedRouting: true
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T06:00:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T06:05:00.000Z");

  const activeResponse = compileRuntimeFromActivation(activationRoot, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-active-serving",
    userMessage: "feedback scanner manifest structural compaction context",
    maxContextBlocks: 3,
    maxContextChars: 180,
    modeRequested: "heuristic",
    activePackId: "pack-active-serving",
    compactionMode: "native",
    runtimeHints: ["pack-backed selection"]
  });

  assert.equal(activeResponse.packId, "pack-active-serving");
  assert.equal(activeResponse.target.packId, "pack-active-serving");
  assert.equal(activeResponse.response.packId, "pack-active-serving");
  assert.equal(activeResponse.diagnostics.modeEffective, "heuristic");
  assert.equal(activeResponse.diagnostics.compactionApplied, true);
  assert.equal(activeResponse.diagnostics.selectedCharCount <= 180, true);

  promoteCandidatePack(activationRoot, "2026-03-06T06:10:00.000Z");

  const promotedResponse = compileRuntimeFromActivation(activationRoot, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-promoted-serving",
    userMessage: "feedback scanner manifest structural compaction context",
    maxContextBlocks: 3,
    maxContextChars: 180,
    modeRequested: "heuristic",
    activePackId: "pack-candidate-serving",
    compactionMode: "native",
    runtimeHints: ["pack-backed selection"]
  });

  assert.equal(promotedResponse.packId, "pack-candidate-serving");
  assert.equal(promotedResponse.target.packId, "pack-candidate-serving");
  assert.equal(promotedResponse.response.packId, "pack-candidate-serving");
  assert.equal(promotedResponse.diagnostics.modeEffective, "learned");
  assert.equal(promotedResponse.diagnostics.usedLearnedRouteFn, true);
  assert.equal(promotedResponse.diagnostics.compactionApplied, true);
});

test("compileRuntimeFromActivation returns target metadata and accepts expectedTarget", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-bridge-activation-"));
  const packRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-bridge-pack-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(packRoot, { recursive: true, force: true }));

  materializeFixturePack(packRoot);
  activatePack(activationRoot, packRoot, "2026-03-06T06:20:00.000Z");

  const result = compileRuntimeFromActivation(
    activationRoot,
    {
      contract: CONTRACT_IDS.runtimeCompile,
      agentId: "agent-bridge-result",
      userMessage: "Compile scanner context.",
      maxContextBlocks: 1,
      modeRequested: "heuristic",
      runtimeHints: ["feedback scanner"]
    },
    {
      expectedTarget: {
        packId: FIXTURE_ARTIFACT_MANIFEST.packId,
        workspaceSnapshot: FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot,
        eventExportDigest: FIXTURE_ARTIFACT_MANIFEST.provenance.eventExports?.exportDigest ?? null
      }
    }
  );

  assert.equal(result.packId, FIXTURE_ARTIFACT_MANIFEST.packId);
  assert.equal(result.response.packId, FIXTURE_ARTIFACT_MANIFEST.packId);
  assert.equal(result.target.packId, FIXTURE_ARTIFACT_MANIFEST.packId);
  assert.equal(result.target.workspaceSnapshot, FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot);
  assert.equal(result.target.eventExportDigest, FIXTURE_ARTIFACT_MANIFEST.provenance.eventExports?.exportDigest ?? null);

  assert.throws(
    () =>
      compileRuntimeFromActivation(
        activationRoot,
        {
          contract: CONTRACT_IDS.runtimeCompile,
          agentId: "agent-bridge-stale",
          userMessage: "Compile scanner context.",
          maxContextBlocks: 1,
          modeRequested: "heuristic"
        },
        {
          expectedTarget: {
            workspaceSnapshot: "workspace-stale@snapshot"
          }
        }
      ),
    /Activation compile target mismatch: runtime compile target workspaceSnapshot/
  );
});

test("compileRuntimeFromActivation rejects conflicting expectation aliases", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-conflict-activation-"));
  const packRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-conflict-pack-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(packRoot, { recursive: true, force: true }));

  materializeFixturePack(packRoot);
  activatePack(activationRoot, packRoot, "2026-03-06T06:21:00.000Z");

  assert.throws(
    () =>
      compileRuntimeFromActivation(
        activationRoot,
        {
          contract: CONTRACT_IDS.runtimeCompile,
          agentId: "agent-bridge-conflict",
          userMessage: "Compile scanner context.",
          maxContextBlocks: 1,
          modeRequested: "heuristic"
        },
        {
          expectation: {
            packId: FIXTURE_ARTIFACT_MANIFEST.packId
          },
          expectedTarget: {
            packId: "pack-stale"
          }
        }
      ),
    /Conflicting compile expectations: packId differ between expectation and expectedTarget/
  );
});

test("compileRuntimeFromActivation fails fast when no active pack is present", () => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-empty-activation-"));

  try {
    assert.throws(
      () =>
        compileRuntimeFromActivation(activationRoot, {
          contract: CONTRACT_IDS.runtimeCompile,
          agentId: "agent-empty-activation",
          userMessage: "Compile scanner context.",
          maxContextBlocks: 1,
          modeRequested: "heuristic"
        }),
      /Activation slot active is empty/
    );
  } finally {
    rmSync(activationRoot, { recursive: true, force: true });
  }
});

test("resolveActivationCompileTarget validates manifest expectations before compile", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-expect-activation-"));
  const packRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-expect-pack-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(packRoot, { recursive: true, force: true }));

  materializeFixturePack(packRoot);
  activatePack(activationRoot, packRoot, "2026-03-06T06:20:00.000Z");

  const resolved = resolveActivationCompileTarget(activationRoot, {
    expectation: {
      packId: FIXTURE_ARTIFACT_MANIFEST.packId,
      routePolicy: FIXTURE_ARTIFACT_MANIFEST.routePolicy,
      routerIdentity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
      workspaceSnapshot: FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot,
      workspaceRevision: FIXTURE_ARTIFACT_MANIFEST.provenance.workspace.revision,
      eventRange: {
        start: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.start,
        end: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.end,
        count: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.count
      },
      eventExportDigest: FIXTURE_ARTIFACT_MANIFEST.provenance.eventExports?.exportDigest ?? null,
      builtAt: FIXTURE_ARTIFACT_MANIFEST.provenance.builtAt
    }
  });

  assert.equal(resolved.slot, "active");
  assert.equal(resolved.target.packId, FIXTURE_ARTIFACT_MANIFEST.packId);

  assert.throws(
    () =>
      resolveActivationCompileTarget(activationRoot, {
        expectation: {
          workspaceSnapshot: "workspace-stale@snapshot"
        }
      }),
    /Activation compile target mismatch: runtime compile target workspaceSnapshot/
  );
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
