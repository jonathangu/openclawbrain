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
import { compileRuntime, compileRuntimeFromActivation, determineRouteMode, loadPackForCompile } from "@openclawbrain/compiler";
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

  const router: RouterArtifactV1 | null = options.learnedRouting
    ? {
        ...FIXTURE_ROUTER_ARTIFACT,
        routerIdentity: `${options.packId}:route_fn`,
        trainedAt: options.builtAt,
        requiresLearnedRouting: true
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
  assert.equal(response.diagnostics.selectionStrategy, "pack_keyword_overlap_v1");
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
  assert.equal(
    result.response.selectedContext.some((block) => /activation promotion evidence/i.test(block.text)),
    true
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
    /activation compile target mismatch: expected workspaceSnapshot workspace-compile@snapshot-stale but found workspace-compile@snapshot-candidate; expected eventExportDigest sha256-stale but found sha256-/
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
      blocks: FIXTURE_PACK_GRAPH.blocks.map((block, index) =>
        index === 0 ? { ...block, text: `${block.text} tampered` } : block
      )
    }),
    "utf8"
  );

  assert.throws(() => loadPackForCompile(rootDir), /graph checksum does not match manifest/);
});
