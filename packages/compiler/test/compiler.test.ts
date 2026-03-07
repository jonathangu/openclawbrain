import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";

import {
  CONTRACT_IDS,
  FIXTURE_ARTIFACT_MANIFEST,
  FIXTURE_PACK_GRAPH,
  FIXTURE_PACK_VECTORS,
  FIXTURE_ROUTER_ARTIFACT,
  buildNormalizedEventExport,
  canonicalJson,
  computeRouterFreshnessChecksum,
  computeRouterQueryChecksum,
  computeRouterWeightsChecksum,
  createFeedbackEvent,
  createInteractionEvent,
  type ArtifactManifestV1,
  type FeedbackEventKind,
  type InteractionEventKind,
  type NormalizedEventExportV1,
  type PackGraphPayloadV1,
  type PackVectorsPayloadV1,
  type RouterArtifactV1
} from "@openclawbrain/contracts";
import {
  compileRuntime,
  compileRuntimeFromActivation,
  determineRouteMode,
  loadPackForCompile,
  rankContextBlocks,
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
  const router: RouterArtifactV1 | null =
    options.learnedRouting
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
    },
    modelFingerprints: options.learnedRouting
      ? ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", router?.routerIdentity ?? "router:missing"]
      : ["BAAI/bge-large-en-v1.5"]
  };

  writePackFile(rootDir, PACK_LAYOUT.graph, graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, vectors);
  if (router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, manifest);
}

interface BuildFixtureExportOptions {
  sessionId: string;
  sequenceStart: number;
  createdAt: string;
  streamSuffix: string;
  interactionKind: InteractionEventKind;
  feedbackKind: FeedbackEventKind;
  feedbackContent: string;
}

interface MaterializeActivationPackOptions {
  packId: string;
  learnedRouting: boolean;
  normalizedEventExport: NormalizedEventExportV1;
  snapshotId: string;
  revision: string | null;
  builtAt: string;
}

function buildFixtureNormalizedEventExport(options: BuildFixtureExportOptions): NormalizedEventExportV1 {
  const interaction = createInteractionEvent({
    eventId: `${options.sessionId}:interaction:${options.sequenceStart}`,
    agentId: "agent-compile-activation",
    sessionId: options.sessionId,
    channel: "cli",
    sequence: options.sequenceStart,
    kind: options.interactionKind,
    createdAt: options.createdAt,
    source: {
      runtimeOwner: "openclaw",
      stream: `openclaw/runtime/${options.streamSuffix}`
    },
    messageId: `msg-${options.sessionId}`
  });

  const feedback = createFeedbackEvent({
    eventId: `${options.sessionId}:feedback:${options.sequenceStart + 1}`,
    agentId: interaction.agentId,
    sessionId: interaction.sessionId,
    channel: interaction.channel,
    sequence: options.sequenceStart + 1,
    kind: options.feedbackKind,
    createdAt: new Date(Date.parse(options.createdAt) + 60_000).toISOString(),
    source: interaction.source,
    content: options.feedbackContent,
    relatedInteractionId: interaction.eventId
  });

  return buildNormalizedEventExport({
    interactionEvents: [interaction],
    feedbackEvents: [feedback]
  });
}

function materializeActivationPack(rootDir: string, options: MaterializeActivationPackOptions): void {
  const feedbackEvent = options.normalizedEventExport.feedbackEvents[0];
  if (feedbackEvent === undefined) {
    throw new Error("activation fixture requires a feedback event");
  }

  const graph: PackGraphPayloadV1 = {
    packId: options.packId,
    blocks: [
      {
        id: `${options.packId}:ctx:promotion-evidence`,
        source: `${feedbackEvent.source.stream}:${feedbackEvent.kind}`,
        text: feedbackEvent.content,
        keywords: ["activation", "promotion", "evidence", "candidate", "runtime"],
        priority: 5,
        tokenCount: 14,
        learning: {
          role: "feedback",
          humanLabels: 1,
          selfLabels: 0,
          decayHalfLifeDays: 30,
          hebbianPulse: 5
        }
      },
      {
        id: `${options.packId}:ctx:workspace-summary`,
        source: `workspace:${options.snapshotId}`,
        text: `Workspace ${options.snapshotId} is pinned to event export ${options.normalizedEventExport.provenance.exportDigest}.`,
        keywords: ["workspace", "snapshot", "event", "export", "digest"],
        priority: 3,
        tokenCount: 12,
        learning: {
          role: "workspace",
          humanLabels: 0,
          selfLabels: 0,
          decayHalfLifeDays: null,
          hebbianPulse: 1
        }
      }
    ]
  };

  const promotionBlock = graph.blocks[0];
  const workspaceBlock = graph.blocks[1];
  if (promotionBlock === undefined || workspaceBlock === undefined) {
    throw new Error("activation fixture requires both promotion and workspace blocks");
  }

  const vectors: PackVectorsPayloadV1 = {
    packId: options.packId,
    entries: [
      {
        blockId: promotionBlock.id,
        keywords: ["activation", "promotion", "evidence"],
        boost: 3,
        weights: {
          activation: 6,
          promotion: 6,
          evidence: 5,
          candidate: 4
        }
      },
      {
        blockId: workspaceBlock.id,
        keywords: ["workspace", "snapshot", "digest"],
        boost: 1,
        weights: {
          workspace: 4,
          snapshot: 4,
          digest: 4
        }
      }
    ]
  };

  const router: RouterArtifactV1 | null =
    options.learnedRouting
      ? {
          ...FIXTURE_ROUTER_ARTIFACT,
          routerIdentity: `${options.packId}:route_fn`,
          trainedAt: options.builtAt,
          requiresLearnedRouting: true,
          training: {
            ...FIXTURE_ROUTER_ARTIFACT.training,
            eventExportDigest: options.normalizedEventExport.provenance.exportDigest,
            freshnessChecksum: computeRouterFreshnessChecksum({
              trainedAt: options.builtAt,
              status: FIXTURE_ROUTER_ARTIFACT.training.status,
              eventExportDigest: options.normalizedEventExport.provenance.exportDigest,
              routeTraceCount: FIXTURE_ROUTER_ARTIFACT.training.routeTraceCount,
              supervisionCount: FIXTURE_ROUTER_ARTIFACT.training.supervisionCount,
              updateCount: FIXTURE_ROUTER_ARTIFACT.training.updateCount
            })
          }
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
    },
    modelFingerprints: options.learnedRouting
      ? ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", router?.routerIdentity ?? "router:missing"]
      : ["BAAI/bge-large-en-v1.5"],
    provenance: {
      ...FIXTURE_ARTIFACT_MANIFEST.provenance,
      workspace: {
        ...FIXTURE_ARTIFACT_MANIFEST.provenance.workspace,
        workspaceId: "workspace-compile",
        snapshotId: options.snapshotId,
        capturedAt: options.builtAt,
        rootDir: "/workspace/compile",
        branch: "main",
        revision: options.revision,
        dirty: false,
        manifestDigest: null,
        labels: ["compile", options.learnedRouting ? "candidate" : "active"],
        files: ["graph.json", "vectors.json", "manifest.json"]
      },
      workspaceSnapshot: options.snapshotId,
      eventRange: options.normalizedEventExport.range,
      eventExports: options.normalizedEventExport.provenance,
      learningSurface: options.normalizedEventExport.provenance.learningSurface,
      builtAt: options.builtAt,
      offlineArtifacts: ["compiler-smoke", "activation-bridge"]
    }
  };

  writePackFile(rootDir, PACK_LAYOUT.graph, graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, vectors);
  if (router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, manifest);
}

function setupActivatedCandidateFixture(t: TestContext): {
  activationRoot: string;
  candidateExport: NormalizedEventExportV1;
  candidatePackId: string;
  candidateRouterIdentity: string;
  candidateBuiltAt: string;
  candidateWorkspaceSnapshot: string;
  candidateWorkspaceRevision: string;
} {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activeExport = buildFixtureNormalizedEventExport({
    sessionId: "session-compile-active",
    sequenceStart: 31,
    createdAt: "2026-03-06T05:31:00.000Z",
    streamSuffix: "compile-active",
    interactionKind: "memory_compiled",
    feedbackKind: "approval",
    feedbackContent: "Keep the baseline active pack stable while a promoted candidate is prepared."
  });
  const candidateExport = buildFixtureNormalizedEventExport({
    sessionId: "session-compile-candidate",
    sequenceStart: 41,
    createdAt: "2026-03-06T05:41:00.000Z",
    streamSuffix: "compile-candidate",
    interactionKind: "operator_override",
    feedbackKind: "teaching",
    feedbackContent: "Activation promotion evidence should reach runtime compile after the candidate is promoted."
  });

  materializeActivationPack(activeRoot, {
    packId: "pack-compile-active",
    learnedRouting: false,
    normalizedEventExport: activeExport,
    snapshotId: "workspace-compile@snapshot-active",
    revision: "compile-active-rev",
    builtAt: "2026-03-06T05:36:00.000Z"
  });

  materializeActivationPack(candidateRoot, {
    packId: "pack-compile-candidate",
    learnedRouting: true,
    normalizedEventExport: candidateExport,
    snapshotId: "workspace-compile@snapshot-candidate",
    revision: "compile-candidate-rev",
    builtAt: "2026-03-06T05:46:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T05:50:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T05:55:00.000Z");
  promoteCandidatePack(activationRoot, "2026-03-06T06:00:00.000Z");

  return {
    activationRoot,
    candidateExport,
    candidatePackId: "pack-compile-candidate",
    candidateRouterIdentity: "pack-compile-candidate:route_fn",
    candidateBuiltAt: "2026-03-06T05:46:00.000Z",
    candidateWorkspaceSnapshot: "workspace-compile@snapshot-candidate",
    candidateWorkspaceRevision: "compile-candidate-rev"
  };
}

test("learned-required packs force learned mode and select scanner context", (t: TestContext) => {
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
  assert.equal(response.diagnostics.selectionStrategy, "pack_route_fn_selection_v1");
  assert.match(response.diagnostics.notes.join(";"), /learned_required_enforced=requested_heuristic->learned/);
});

test("compileRuntime uses learned router policy updates and emits refresh diagnostics", (t: TestContext) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-router-delta-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const router: RouterArtifactV1 = {
    routerIdentity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
    strategy: "learned_route_fn_v1",
    trainedAt: "2026-03-07T12:00:00.000Z",
    requiresLearnedRouting: true,
    traces: [
      {
        traceId: "trace-compile-router-delta",
        sourceEventId: "evt-compile-router-delta",
        sourceContract: CONTRACT_IDS.feedbackEvents,
        sourceKind: "teaching",
        supervisionKind: "human_feedback",
        targetBlockIds: ["ctx-structural-ops"],
        reward: 4,
        queryTokens: ["ambiguous"],
        queryVector: {
          ambiguous: 2
        }
      }
    ],
    policyUpdates: [
      {
        blockId: "ctx-structural-ops",
        delta: 9,
        evidenceCount: 1,
        rewardSum: 4,
        tokenWeights: {
          ambiguous: 7
        },
        traceIds: ["trace-compile-router-delta"]
      }
    ],
    training: {
      status: "updated",
      eventExportDigest: FIXTURE_ARTIFACT_MANIFEST.provenance.eventExports?.exportDigest ?? null,
      routeTraceCount: 1,
      supervisionCount: 1,
      updateCount: 1,
      queryChecksum: computeRouterQueryChecksum([
        {
          traceId: "trace-compile-router-delta",
          sourceEventId: "evt-compile-router-delta",
          sourceContract: CONTRACT_IDS.feedbackEvents,
          sourceKind: "teaching",
          supervisionKind: "human_feedback",
          targetBlockIds: ["ctx-structural-ops"],
          reward: 4,
          queryTokens: ["ambiguous"],
          queryVector: {
            ambiguous: 2
          }
        }
      ]),
      weightsChecksum: computeRouterWeightsChecksum([
        {
          blockId: "ctx-structural-ops",
          delta: 9,
          evidenceCount: 1,
          rewardSum: 4,
          tokenWeights: {
            ambiguous: 7
          },
          traceIds: ["trace-compile-router-delta"]
        }
      ]),
      freshnessChecksum: computeRouterFreshnessChecksum({
        trainedAt: "2026-03-07T12:00:00.000Z",
        status: "updated",
        eventExportDigest: FIXTURE_ARTIFACT_MANIFEST.provenance.eventExports?.exportDigest ?? null,
        routeTraceCount: 1,
        supervisionCount: 1,
        updateCount: 1
      }),
      noOpReason: null
    }
  };
  const manifest: ArtifactManifestV1 = {
    ...FIXTURE_ARTIFACT_MANIFEST,
    payloadChecksums: {
      ...FIXTURE_ARTIFACT_MANIFEST.payloadChecksums,
      router: computePayloadChecksum(router)
    }
  };

  writePackFile(rootDir, PACK_LAYOUT.graph, FIXTURE_PACK_GRAPH);
  writePackFile(rootDir, PACK_LAYOUT.vectors, FIXTURE_PACK_VECTORS);
  writePackFile(rootDir, PACK_LAYOUT.router, router);
  writePackFile(rootDir, PACK_LAYOUT.manifest, manifest);

  const response = compileRuntime(rootDir, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-router-delta",
    userMessage: "ambiguous",
    maxContextBlocks: 1,
    modeRequested: "learned",
    runtimeHints: []
  });

  assert.equal(response.selectedContext[0]?.id, "ctx-structural-ops");
  assert.match(response.diagnostics.notes.join(";"), /router_refresh_status=updated/);
  assert.match(response.diagnostics.notes.join(";"), /router_update_count=1/);
  assert.match(response.diagnostics.notes.join(";"), /router_top_deltas=ctx-structural-ops:9/);
});

test("rankContextBlocks normalizes underscored keyword weights for runtime hints", (t: TestContext) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  const pack = loadPackForCompile(rootDir);
  const ranked = rankContextBlocks(pack, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-keyword-normalize",
    userMessage: "Keep the runtime always on.",
    maxContextBlocks: 1,
    modeRequested: "heuristic",
    runtimeHints: ["always on"]
  });

  const scannerBlock = ranked.find((entry) => entry.blockId === "ctx-feedback-scanner");
  assert.equal(scannerBlock?.matchedTokens.includes("always"), true);
  assert.equal(scannerBlock?.matchedTokens.includes("on"), true);
});

test("compileRuntime falls back to priority order when nothing matches", (t: TestContext) => {
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

test("compileRuntime applies native structural compaction under a character budget", (t: TestContext) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  const response = compileRuntime(rootDir, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-compact",
    userMessage: "qwen manifest split",
    maxContextBlocks: 3,
    maxContextChars: 180,
    modeRequested: "heuristic",
    compactionMode: "native",
    runtimeHints: []
  });

  assert.equal(response.diagnostics.compactionApplied, true);
  assert.equal(response.diagnostics.compactionMode, "native");
  assert.equal(response.diagnostics.selectedCharCount <= 180, true);
  assert.equal(response.selectedContext.some((block) => (block.compactedFrom?.length ?? 0) > 1), true);
  assert.match(response.diagnostics.notes.join(";"), /native_structural_compaction=applied/);
});

test("compileRuntime fills fallback tiers after matched selection", (t: TestContext) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  const response = compileRuntime(rootDir, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-tier-fill",
    userMessage: "Run the scanner with qwen checkpoints.",
    maxContextBlocks: 2,
    modeRequested: "heuristic",
    compactionMode: "none",
    runtimeHints: []
  });

  assert.deepEqual(response.selectedContext.map((block) => block.id), ["ctx-feedback-scanner", "ctx-runtime-compile"]);
  assert.match(response.diagnostics.notes.join(";"), /selection_tiers=token_match\+priority_fallback/);
});

test("compileRuntime prunes overlapping compacted and raw context blocks", (t: TestContext) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-compile-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));
  materializeFixturePack(rootDir);

  const response = compileRuntime(rootDir, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-overlap-prune",
    userMessage: "feedback scanner manifest structural compaction context",
    maxContextBlocks: 3,
    modeRequested: "heuristic",
    compactionMode: "none",
    runtimeHints: ["pack-backed selection"]
  });

  assert.deepEqual(response.selectedContext.map((block) => block.id), ["ctx-context-compact"]);
  assert.match(response.diagnostics.notes.join(";"), /selection_compaction_deduped=3/);
});

test("compileRuntime rejects stale activePackId expectations", (t: TestContext) => {
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

test("compileRuntimeFromActivation serves the active pack and respects promotion", (t: TestContext) => {
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

  const activeCompile = compileRuntimeFromActivation(activationRoot, {
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

  assert.equal(activeCompile.slot, "active");
  assert.equal(activeCompile.response.packId, "pack-active-serving");
  assert.equal(activeCompile.target.packId, "pack-active-serving");
  assert.equal(activeCompile.response.diagnostics.modeEffective, "heuristic");
  assert.equal(activeCompile.response.diagnostics.selectedCharCount <= 180, true);

  promoteCandidatePack(activationRoot, "2026-03-06T06:10:00.000Z");

  const promotedCompile = compileRuntimeFromActivation(activationRoot, {
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

  assert.equal(promotedCompile.slot, "active");
  assert.equal(promotedCompile.response.packId, "pack-candidate-serving");
  assert.equal(promotedCompile.target.packId, "pack-candidate-serving");
  assert.equal(promotedCompile.response.diagnostics.modeEffective, "learned");
  assert.equal(promotedCompile.response.diagnostics.usedLearnedRouteFn, true);
});

test("compileRuntimeFromActivation surfaces stale-route warnings when a fresher candidate is staged", (t: TestContext) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-freshness-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-freshness-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-freshness-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activeExport = buildFixtureNormalizedEventExport({
    sessionId: "session-freshness-active",
    sequenceStart: 21,
    createdAt: "2026-03-06T04:21:00.000Z",
    streamSuffix: "freshness-active",
    interactionKind: "memory_compiled",
    feedbackKind: "approval",
    feedbackContent: "Serve the current active route until a newer candidate is safe to promote."
  });
  const candidateExport = buildFixtureNormalizedEventExport({
    sessionId: "session-freshness-candidate",
    sequenceStart: 41,
    createdAt: "2026-03-06T05:41:00.000Z",
    streamSuffix: "freshness-candidate",
    interactionKind: "operator_override",
    feedbackKind: "teaching",
    feedbackContent: "A fresher candidate should warn operators before promotion."
  });

  materializeActivationPack(activeRoot, {
    packId: "pack-freshness-active",
    learnedRouting: false,
    normalizedEventExport: activeExport,
    snapshotId: "workspace-freshness@snapshot-active",
    revision: "freshness-active-rev",
    builtAt: "2026-03-06T04:26:00.000Z"
  });
  materializeActivationPack(candidateRoot, {
    packId: "pack-freshness-candidate",
    learnedRouting: true,
    normalizedEventExport: candidateExport,
    snapshotId: "workspace-freshness@snapshot-candidate",
    revision: "freshness-candidate-rev",
    builtAt: "2026-03-06T05:46:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T05:50:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T05:55:00.000Z");

  const result = compileRuntimeFromActivation(activationRoot, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-freshness-warning",
    userMessage: "Compile the active route while a fresher candidate is waiting.",
    maxContextBlocks: 2,
    maxContextChars: 320,
    modeRequested: "heuristic",
    runtimeHints: ["active", "candidate", "freshness"],
    compactionMode: "native"
  });

  assert.equal(result.slot, "active");
  assert.equal(result.target.packId, "pack-freshness-active");
  assert.match(
    result.response.diagnostics.notes.join(";"),
    /stale_route_warning=active pack pack-freshness-active is behind promotion-ready candidate pack-freshness-candidate/
  );
});

test("compileRuntimeFromActivation treats snapshot and export drift as freshness even when event ranges tie", (t: TestContext) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-freshness-tie-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-freshness-tie-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-freshness-tie-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activeExport = buildFixtureNormalizedEventExport({
    sessionId: "session-freshness-tie-active",
    sequenceStart: 91,
    createdAt: "2026-03-06T08:01:00.000Z",
    streamSuffix: "freshness-tie-active",
    interactionKind: "memory_compiled",
    feedbackKind: "approval",
    feedbackContent: "Serve the current active pack until an equivalent-range candidate proves fresher provenance."
  });
  const candidateExport = buildFixtureNormalizedEventExport({
    sessionId: "session-freshness-tie-candidate",
    sequenceStart: 91,
    createdAt: "2026-03-06T08:01:00.000Z",
    streamSuffix: "freshness-tie-candidate",
    interactionKind: "operator_override",
    feedbackKind: "teaching",
    feedbackContent: "The candidate keeps the same event range but advances workspace and export freshness."
  });

  materializeActivationPack(activeRoot, {
    packId: "pack-freshness-tie-active",
    learnedRouting: false,
    normalizedEventExport: activeExport,
    snapshotId: "workspace-freshness-tie@snapshot-active",
    revision: "freshness-tie-active-rev",
    builtAt: "2026-03-06T08:06:00.000Z"
  });
  materializeActivationPack(candidateRoot, {
    packId: "pack-freshness-tie-candidate",
    learnedRouting: true,
    normalizedEventExport: candidateExport,
    snapshotId: "workspace-freshness-tie@snapshot-candidate",
    revision: "freshness-tie-candidate-rev",
    builtAt: "2026-03-06T08:06:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T08:10:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T08:15:00.000Z");

  const result = compileRuntimeFromActivation(activationRoot, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-freshness-tie-warning",
    userMessage: "Keep serving the active route while the candidate proves fresher provenance.",
    maxContextBlocks: 2,
    maxContextChars: 320,
    modeRequested: "heuristic",
    runtimeHints: ["active", "candidate", "freshness", "snapshot", "export"],
    compactionMode: "native"
  });

  assert.equal(result.slot, "active");
  assert.equal(result.target.packId, "pack-freshness-tie-active");
  assert.match(
    result.response.diagnostics.notes.join(";"),
    /stale_route_warning=active pack pack-freshness-tie-active is behind promotion-ready candidate pack-freshness-tie-candidate/
  );
});

test("compileRuntimeFromActivation evaluates the staged candidate only when promotion safety holds", (t: TestContext) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-candidate-eval-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-candidate-eval-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-candidate-eval-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activeExport = buildFixtureNormalizedEventExport({
    sessionId: "session-candidate-eval-active",
    sequenceStart: 101,
    createdAt: "2026-03-06T09:01:00.000Z",
    streamSuffix: "candidate-eval-active",
    interactionKind: "memory_compiled",
    feedbackKind: "approval",
    feedbackContent: "Keep the active pack stable until the staged candidate passes evaluation."
  });
  const candidateExport = buildFixtureNormalizedEventExport({
    sessionId: "session-candidate-eval-candidate",
    sequenceStart: 111,
    createdAt: "2026-03-06T09:11:00.000Z",
    streamSuffix: "candidate-eval-candidate",
    interactionKind: "operator_override",
    feedbackKind: "teaching",
    feedbackContent: "Evaluate the staged candidate only after promotion safety is satisfied."
  });

  materializeActivationPack(activeRoot, {
    packId: "pack-candidate-eval-active",
    learnedRouting: false,
    normalizedEventExport: activeExport,
    snapshotId: "workspace-candidate-eval@snapshot-active",
    revision: "candidate-eval-active-rev",
    builtAt: "2026-03-06T09:06:00.000Z"
  });
  materializeActivationPack(candidateRoot, {
    packId: "pack-candidate-eval-candidate",
    learnedRouting: true,
    normalizedEventExport: candidateExport,
    snapshotId: "workspace-candidate-eval@snapshot-candidate",
    revision: "candidate-eval-candidate-rev",
    builtAt: "2026-03-06T09:16:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T09:20:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T09:25:00.000Z");

  const result = compileRuntimeFromActivation(
    activationRoot,
    {
      contract: CONTRACT_IDS.runtimeCompile,
      agentId: "agent-candidate-eval",
      userMessage: "Evaluate the staged candidate pack before promotion.",
      maxContextBlocks: 2,
      maxContextChars: 360,
      modeRequested: "heuristic",
      runtimeHints: ["candidate", "evaluation", "promotion", "safety"],
      compactionMode: "native"
    },
    {
      slot: "candidate",
      expectedTarget: {
        packId: "pack-candidate-eval-candidate",
        routePolicy: "requires_learned_routing",
        routerIdentity: "pack-candidate-eval-candidate:route_fn",
        workspaceSnapshot: "workspace-candidate-eval@snapshot-candidate",
        workspaceRevision: "candidate-eval-candidate-rev",
        eventRange: {
          start: candidateExport.range.start,
          end: candidateExport.range.end,
          count: candidateExport.range.count
        },
        eventExportDigest: candidateExport.provenance.exportDigest,
        builtAt: "2026-03-06T09:16:00.000Z"
      }
    }
  );

  assert.equal(result.slot, "candidate");
  assert.equal(result.target.packId, "pack-candidate-eval-candidate");
  assert.equal(result.response.packId, "pack-candidate-eval-candidate");
  assert.equal(result.response.diagnostics.modeEffective, "learned");
  assert.equal(result.response.diagnostics.usedLearnedRouteFn, true);
  assert.match(result.response.diagnostics.notes.join(";"), /activation_slot=candidate/);
  assert.match(result.response.diagnostics.notes.join(";"), /target_pack_id=pack-candidate-eval-candidate/);
});

test("resolveActivationCompileTarget blocks candidate evaluation when promotion safety fails", (t: TestContext) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-candidate-gate-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-candidate-gate-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-candidate-gate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activeExport = buildFixtureNormalizedEventExport({
    sessionId: "session-candidate-gate-active",
    sequenceStart: 61,
    createdAt: "2026-03-06T06:01:00.000Z",
    streamSuffix: "candidate-gate-active",
    interactionKind: "memory_compiled",
    feedbackKind: "approval",
    feedbackContent: "Keep the active route newer than any stale candidate."
  });
  const staleCandidateExport = buildFixtureNormalizedEventExport({
    sessionId: "session-candidate-gate-stale",
    sequenceStart: 41,
    createdAt: "2026-03-06T05:01:00.000Z",
    streamSuffix: "candidate-gate-stale",
    interactionKind: "operator_override",
    feedbackKind: "teaching",
    feedbackContent: "This staged candidate is stale and must be rejected before evaluation."
  });

  materializeActivationPack(activeRoot, {
    packId: "pack-candidate-gate-active",
    learnedRouting: false,
    normalizedEventExport: activeExport,
    snapshotId: "workspace-candidate-gate@snapshot-active",
    revision: "candidate-gate-active-rev",
    builtAt: "2026-03-06T06:06:00.000Z"
  });
  materializeActivationPack(candidateRoot, {
    packId: "pack-candidate-gate-stale",
    learnedRouting: true,
    normalizedEventExport: staleCandidateExport,
    snapshotId: "workspace-candidate-gate@snapshot-stale",
    revision: "candidate-gate-stale-rev",
    builtAt: "2026-03-06T05:06:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T06:10:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T06:15:00.000Z");

  assert.throws(
    () =>
      resolveActivationCompileTarget(activationRoot, {
        slot: "candidate"
      }),
    /Candidate compile blocked: candidate pack builtAt must not precede active pack builtAt during promotion; candidate eventRange\.end must be >= active eventRange\.end during promotion/
  );
});

test("compileRuntimeFromActivation surfaces candidate rejection findings on the active route", (t: TestContext) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-candidate-rejection-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-candidate-rejection-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-rejection-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activeExport = buildFixtureNormalizedEventExport({
    sessionId: "session-candidate-rejection-active",
    sequenceStart: 71,
    createdAt: "2026-03-06T07:01:00.000Z",
    streamSuffix: "candidate-rejection-active",
    interactionKind: "memory_compiled",
    feedbackKind: "approval",
    feedbackContent: "The active route should keep serving while a broken candidate is rejected."
  });
  const candidateExport = buildFixtureNormalizedEventExport({
    sessionId: "session-candidate-rejection-candidate",
    sequenceStart: 81,
    createdAt: "2026-03-06T07:21:00.000Z",
    streamSuffix: "candidate-rejection-candidate",
    interactionKind: "operator_override",
    feedbackKind: "teaching",
    feedbackContent: "This candidate will drift after staging and should be rejected loudly."
  });

  materializeActivationPack(activeRoot, {
    packId: "pack-candidate-rejection-active",
    learnedRouting: false,
    normalizedEventExport: activeExport,
    snapshotId: "workspace-candidate-rejection@snapshot-active",
    revision: "candidate-rejection-active-rev",
    builtAt: "2026-03-06T07:06:00.000Z"
  });
  materializeActivationPack(candidateRoot, {
    packId: "pack-candidate-rejection-candidate",
    learnedRouting: true,
    normalizedEventExport: candidateExport,
    snapshotId: "workspace-candidate-rejection@snapshot-candidate",
    revision: "candidate-rejection-candidate-rev",
    builtAt: "2026-03-06T07:26:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T07:30:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T07:35:00.000Z");

  const candidatePack = loadPackForCompile(candidateRoot);
  writePackFile(candidateRoot, PACK_LAYOUT.manifest, {
    ...candidatePack.manifest,
    modelFingerprints: [...candidatePack.manifest.modelFingerprints, "mutated-after-staging"]
  });

  const result = compileRuntimeFromActivation(activationRoot, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-candidate-rejection-note",
    userMessage: "Compile the active route while the candidate is being rejected.",
    maxContextBlocks: 2,
    maxContextChars: 320,
    modeRequested: "heuristic",
    runtimeHints: ["active", "candidate", "rejected"],
    compactionMode: "native"
  });

  assert.equal(result.slot, "active");
  assert.equal(result.target.packId, "pack-candidate-rejection-active");
  assert.match(result.response.diagnostics.notes.join(";"), /candidate_rejected=pack-candidate-rejection-candidate:.*manifestDigest/);
});

test("compileRuntimeFromActivation fails clearly when an active learned route artifact goes missing", (t: TestContext) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-missing-route-activation-"));
  const packRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-missing-route-pack-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(packRoot, { recursive: true, force: true }));

  materializeNamedPack(packRoot, {
    packId: "pack-missing-route-artifact",
    learnedRouting: true
  });
  activatePack(activationRoot, packRoot, "2026-03-06T07:40:00.000Z");
  rmSync(path.join(packRoot, "router", "model.json"), { force: true });

  assert.throws(
    () =>
      compileRuntimeFromActivation(activationRoot, {
        contract: CONTRACT_IDS.runtimeCompile,
        agentId: "agent-missing-route-artifact",
        userMessage: "Compile scanner context.",
        maxContextBlocks: 1,
        modeRequested: "heuristic"
      }),
    /router payload not found/
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

test("resolveActivationCompileTarget validates manifest expectations before compile", (t: TestContext) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-expect-activation-"));
  const packRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-expect-pack-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(packRoot, { recursive: true, force: true }));

  materializeFixturePack(packRoot);
  activatePack(activationRoot, packRoot, "2026-03-06T06:20:00.000Z");

  const resolved = resolveActivationCompileTarget(activationRoot, {
    expectedTarget: {
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
        expectedTarget: {
          workspaceSnapshot: "workspace-stale@snapshot"
        }
      }),
    /Activation compile target mismatch: runtime compile target workspaceSnapshot/
  );
});

test("resolveActivationCompileTarget rejects the removed legacy expectation alias", (t: TestContext) => {
  const fixture = setupActivatedCandidateFixture(t);
  const legacyOptions = {
    expectation: {
      packId: fixture.candidatePackId,
      routePolicy: "requires_learned_routing",
      routerIdentity: fixture.candidateRouterIdentity,
      workspaceSnapshot: fixture.candidateWorkspaceSnapshot,
      workspaceRevision: fixture.candidateWorkspaceRevision,
      eventRange: {
        start: fixture.candidateExport.range.start,
        end: fixture.candidateExport.range.end,
        count: fixture.candidateExport.range.count
      },
      eventExportDigest: fixture.candidateExport.provenance.exportDigest,
      builtAt: fixture.candidateBuiltAt
    }
  } as unknown as Parameters<typeof resolveActivationCompileTarget>[1];

  assert.throws(
    () => resolveActivationCompileTarget(fixture.activationRoot, legacyOptions),
    /Activation compile options expectation has been removed; use expectedTarget/
  );
});

test("compileRuntimeFromActivation compiles the promoted pack when expected provenance matches", (t: TestContext) => {
  const fixture = setupActivatedCandidateFixture(t);

  const result = compileRuntimeFromActivation(
    fixture.activationRoot,
    {
      contract: CONTRACT_IDS.runtimeCompile,
      agentId: "agent-activation-bridge",
      userMessage: "Compile activation promotion evidence from the promoted candidate pack.",
      maxContextBlocks: 2,
      maxContextChars: 360,
      modeRequested: "heuristic",
      runtimeHints: ["activation", "promotion", "evidence"],
      compactionMode: "native"
    },
    {
      expectedTarget: {
        packId: fixture.candidatePackId,
        routePolicy: "requires_learned_routing",
        routerIdentity: fixture.candidateRouterIdentity,
        workspaceSnapshot: fixture.candidateWorkspaceSnapshot,
        workspaceRevision: fixture.candidateWorkspaceRevision,
        eventRange: {
          start: fixture.candidateExport.range.start,
          end: fixture.candidateExport.range.end,
          count: fixture.candidateExport.range.count
        },
        eventExportDigest: fixture.candidateExport.provenance.exportDigest,
        builtAt: fixture.candidateBuiltAt
      }
    }
  );

  assert.equal(result.target.packId, fixture.candidatePackId);
  assert.equal(result.target.eventExportDigest, fixture.candidateExport.provenance.exportDigest);
  assert.equal(result.target.workspaceSnapshot, fixture.candidateWorkspaceSnapshot);
  assert.equal(result.response.packId, fixture.candidatePackId);
  assert.equal(result.response.diagnostics.modeEffective, "learned");
  assert.equal(result.response.diagnostics.usedLearnedRouteFn, true);
  assert.equal(result.response.selectedContext.length > 0, true);
  assert.equal(result.response.selectedContext.some((block) => /activation promotion evidence/i.test(block.text)), true);
  assert.match(result.response.diagnostics.notes.join(";"), /activation_slot=active/);
  assert.match(result.response.diagnostics.notes.join(";"), new RegExp(`target_pack_id=${fixture.candidatePackId}`));
  assert.match(result.response.diagnostics.notes.join(";"), /target_route_policy=requires_learned_routing/);
  assert.match(
    result.response.diagnostics.notes.join(";"),
    new RegExp(
      `target_event_range=${fixture.candidateExport.range.start}-${fixture.candidateExport.range.end}#${fixture.candidateExport.range.count}`
    )
  );
  assert.match(
    result.response.diagnostics.notes.join(";"),
    new RegExp(`target_event_export_digest=${fixture.candidateExport.provenance.exportDigest}`)
  );
  assert.match(
    result.response.diagnostics.notes.join(";"),
    new RegExp(`target_workspace_snapshot=${fixture.candidateWorkspaceSnapshot}`)
  );
  assert.match(
    result.response.diagnostics.notes.join(";"),
    new RegExp(`target_workspace_revision=${fixture.candidateWorkspaceRevision}`)
  );
  assert.match(result.response.diagnostics.notes.join(";"), new RegExp(`target_built_at=${fixture.candidateBuiltAt}`));
  assert.match(
    result.response.diagnostics.notes.join(";"),
    new RegExp(`target_router_identity=${fixture.candidateRouterIdentity}`)
  );
});

test("compileRuntimeFromActivation rejects stale provenance expectations before compiling", (t: TestContext) => {
  const fixture = setupActivatedCandidateFixture(t);

  assert.throws(
    () =>
      compileRuntimeFromActivation(
        fixture.activationRoot,
        {
          contract: CONTRACT_IDS.runtimeCompile,
          agentId: "agent-activation-stale",
          userMessage: "Compile the promoted candidate.",
          maxContextBlocks: 1,
          modeRequested: "heuristic",
          runtimeHints: ["activation"]
        },
        {
          expectedTarget: {
            workspaceSnapshot: "workspace-compile@snapshot-stale",
            eventExportDigest: "sha256-stale"
          }
        }
      ),
    /Activation compile target mismatch: runtime compile target workspaceSnapshot .* does not match expected workspace-compile@snapshot-stale; runtime compile target eventExportDigest .* does not match expected sha256-stale/
  );
});

test("pack load rejects tampered graph payloads", (t: TestContext) => {
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
      blocks: FIXTURE_PACK_GRAPH.blocks.map((block, index) => (index === 0 ? { ...block, text: `${block.text} tampered` } : block))
    }),
    "utf8"
  );

  assert.throws(() => loadPackForCompile(rootDir), /graph checksum does not match manifest/);
});
