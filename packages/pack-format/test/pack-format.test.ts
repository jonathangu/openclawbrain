import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  buildNormalizedEventExport,
  canonicalJson,
  CONTRACT_IDS,
  createFeedbackEvent,
  createInteractionEvent,
  FIXTURE_ARTIFACT_MANIFEST,
  FIXTURE_PACK_GRAPH,
  FIXTURE_PACK_VECTORS,
  FIXTURE_ROUTER_ARTIFACT
} from "@openclawbrain/contracts";
import {
  activatePack,
  computePayloadChecksum,
  describeActivationObservability,
  describeActivationTarget,
  describePackCompileTarget,
  inspectActivationState,
  loadActivationPointers,
  loadPack,
  loadPackFromActivation,
  PACK_LAYOUT,
  promoteCandidatePack,
  rollbackActivePack,
  stageCandidatePack,
  validatePackActivationReadiness,
  writePackFile
} from "@openclawbrain/pack-format";

function materializeTestPack(
  rootDir: string,
  options: {
    packId: string;
    learnedRouting: boolean;
    eventStart: number;
    builtAt?: string;
  }
) {
  const graph = {
    ...FIXTURE_PACK_GRAPH,
    packId: options.packId,
    blocks: FIXTURE_PACK_GRAPH.blocks.map((block) => ({
      ...block,
      id: `${options.packId}:${block.id}`
    }))
  };
  const vectors = {
    ...FIXTURE_PACK_VECTORS,
    packId: options.packId,
    entries: FIXTURE_PACK_VECTORS.entries.map((entry, index) => ({
      ...entry,
      blockId: graph.blocks[index]?.id ?? entry.blockId
    }))
  };
  const router = options.learnedRouting
    ? {
        ...FIXTURE_ROUTER_ARTIFACT,
        routerIdentity: `${options.packId}:route_fn`
      }
    : null;
  const eventExport = buildNormalizedEventExport({
    interactionEvents: [
      createInteractionEvent({
        eventId: `${options.packId}:interaction`,
        agentId: "pack-test-agent",
        sessionId: `${options.packId}:session`,
        channel: "cli",
        sequence: options.eventStart,
        kind: "memory_compiled",
        createdAt: `2026-03-06T00:${String(options.eventStart).padStart(2, "0")}:00.000Z`,
        source: {
          runtimeOwner: "openclaw",
          stream: `openclaw/runtime/${options.packId}`
        },
        packId: options.packId
      })
    ],
    feedbackEvents: [
      createFeedbackEvent({
        eventId: `${options.packId}:feedback`,
        agentId: "pack-test-agent",
        sessionId: `${options.packId}:session`,
        channel: "cli",
        sequence: options.eventStart + 1,
        kind: "teaching",
        createdAt: `2026-03-06T00:${String(options.eventStart + 1).padStart(2, "0")}:00.000Z`,
        source: {
          runtimeOwner: "openclaw",
          stream: `openclaw/runtime/${options.packId}`
        },
        content: `Teach activation metadata for ${options.packId}.`,
        relatedInteractionId: `${options.packId}:interaction`
      })
    ]
  });

  const manifest = {
    ...FIXTURE_ARTIFACT_MANIFEST,
    contract: CONTRACT_IDS.artifactManifest,
    packId: options.packId,
    routePolicy: options.learnedRouting ? "requires_learned_routing" : "heuristic_allowed",
    runtimeAssets: {
      graphPath: PACK_LAYOUT.graph,
      vectorPath: PACK_LAYOUT.vectors,
      router: options.learnedRouting
        ? {
            kind: "artifact" as const,
            identity: router?.routerIdentity ?? null,
            artifactPath: PACK_LAYOUT.router
          }
        : {
            kind: "none" as const,
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
      builtAt: options.builtAt ?? `2026-03-06T00:${String(options.eventStart).padStart(2, "0")}:30.000Z`,
      eventRange: eventExport.range,
      eventExports: eventExport.provenance,
      learningSurface: eventExport.provenance.learningSurface
    }
  };

  writePackFile(rootDir, PACK_LAYOUT.graph, graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, vectors);
  if (router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, manifest);

  return loadPack(rootDir);
}

test("activation observability reports learned route freshness, graph dynamics, and promotion freshness", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-observability-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-observability-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-observability-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  materializeTestPack(activeRoot, {
    packId: "pack-observability-active",
    learnedRouting: false,
    eventStart: 41,
    builtAt: "2026-03-06T04:01:00.000Z"
  });
  const candidate = materializeTestPack(candidateRoot, {
    packId: "pack-observability-candidate",
    learnedRouting: true,
    eventStart: 51,
    builtAt: "2026-03-06T04:11:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T04:15:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T04:20:00.000Z");

  const staged = describeActivationObservability(activationRoot, "active", {
    updatedAt: "2026-03-06T04:21:00.000Z"
  });

  assert.equal(staged.learnedRouteFn.required, false);
  assert.equal(staged.learnedRouteFn.available, false);
  assert.equal(staged.graphDynamics.packId, "pack-observability-active");
  assert.equal(staged.graphDynamics.eventRange?.end, 42);
  assert.equal(staged.promotionFreshness.promotionAllowed, true);
  assert.equal(staged.promotionFreshness.activeBehindPromotionReadyCandidate, true);
  assert.deepEqual(staged.promotionFreshness.candidateAheadBy, {
    builtAt: true,
    eventRangeEnd: true,
    eventRangeCount: false,
    workspaceSnapshot: false,
    workspaceRevision: false,
    eventExportDigest: true
  });

  promoteCandidatePack(activationRoot, "2026-03-06T04:25:00.000Z");

  const promoted = describeActivationObservability(activationRoot, "active", {
    requireActivationReady: true,
    updatedAt: "2026-03-06T04:26:00.000Z"
  });

  assert.equal(promoted.target?.packId, "pack-observability-candidate");
  assert.equal(promoted.learnedRouteFn.packId, "pack-observability-candidate");
  assert.equal(promoted.learnedRouteFn.required, true);
  assert.equal(promoted.learnedRouteFn.available, true);
  assert.equal(promoted.learnedRouteFn.routerAssetKind, "artifact");
  assert.equal(promoted.learnedRouteFn.routerIdentity, `${candidate.manifest.packId}:route_fn`);
  assert.equal(promoted.learnedRouteFn.routeFnVersion, "learned_route_fn_v1");
  assert.equal(promoted.learnedRouteFn.routerChecksum, candidate.manifest.payloadChecksums.router);
  assert.equal(promoted.learnedRouteFn.routerTrainedAt, candidate.router?.trainedAt ?? null);
  assert.equal(promoted.graphDynamics.packId, candidate.manifest.packId);
  assert.equal(promoted.graphDynamics.graphChecksum, candidate.manifest.payloadChecksums.graph);
  assert.deepEqual(promoted.graphDynamics.structuralOps, candidate.manifest.graphDynamics.structuralOps);
  assert.equal(promoted.graphDynamics.builtAt, candidate.manifest.provenance.builtAt);
  assert.equal(promoted.promotionFreshness.candidatePackId, null);
  assert.equal(promoted.promotionFreshness.rollbackAllowed, true);
});

test("pack descriptors keep packs immutable and addressable", (t) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-pack-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  writePackFile(rootDir, PACK_LAYOUT.graph, FIXTURE_PACK_GRAPH);
  writePackFile(rootDir, PACK_LAYOUT.vectors, FIXTURE_PACK_VECTORS);
  writePackFile(rootDir, PACK_LAYOUT.router, FIXTURE_ROUTER_ARTIFACT);
  writePackFile(rootDir, PACK_LAYOUT.manifest, FIXTURE_ARTIFACT_MANIFEST);

  const descriptor = loadPack(rootDir);
  const compileTarget = describePackCompileTarget(descriptor);

  assert.equal(descriptor.manifestPath, path.join(rootDir, "manifest.json"));
  assert.equal(descriptor.router?.routerIdentity, FIXTURE_ROUTER_ARTIFACT.routerIdentity);
  assert.equal(descriptor.graph.blocks[0]?.id, "ctx-feedback-scanner");
  assert.equal(compileTarget.packId, FIXTURE_ARTIFACT_MANIFEST.packId);
  assert.equal(compileTarget.workspaceSnapshot, FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot);
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

  assert.throws(() => loadPack(rootDir), /graph checksum does not match manifest/);
});

test("pack load rejects manifest asset paths that escape the pack root", (t) => {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-pack-"));
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  writePackFile(rootDir, PACK_LAYOUT.graph, FIXTURE_PACK_GRAPH);
  writePackFile(rootDir, PACK_LAYOUT.vectors, FIXTURE_PACK_VECTORS);
  writePackFile(rootDir, PACK_LAYOUT.router, FIXTURE_ROUTER_ARTIFACT);
  writePackFile(rootDir, PACK_LAYOUT.manifest, {
    ...FIXTURE_ARTIFACT_MANIFEST,
    runtimeAssets: {
      ...FIXTURE_ARTIFACT_MANIFEST.runtimeAssets,
      graphPath: "../graph.json"
    }
  });

  assert.throws(() => loadPack(rootDir), /graphPath must not escape the pack root/);
});

test("promotion flips active and previous pointers while rollback restores the prior active pack", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activePack = materializeTestPack(activeRoot, {
    packId: "pack-active-test",
    learnedRouting: false,
    eventStart: 10
  });
  const candidatePack = materializeTestPack(candidateRoot, {
    packId: "pack-candidate-test",
    learnedRouting: true,
    eventStart: 20
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T01:00:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T01:05:00.000Z");

  const staged = inspectActivationState(activationRoot, "2026-03-06T01:06:00.000Z");
  assert.equal(staged.active?.packId, activePack.manifest.packId);
  assert.equal(staged.candidate?.packId, candidatePack.manifest.packId);
  assert.equal(staged.previous, null);
  assert.equal(staged.candidate?.activationReady, true);
  assert.deepEqual(staged.active?.eventRange, {
    start: 10,
    end: 11,
    count: 2
  });
  assert.deepEqual(staged.candidate?.eventRange, {
    start: 20,
    end: 21,
    count: 2
  });
  assert.equal(staged.candidate?.eventExportDigest, candidatePack.manifest.provenance.eventExports?.exportDigest ?? null);
  assert.equal(staged.promotion.allowed, true);
  assert.equal(staged.rollback.allowed, false);

  promoteCandidatePack(activationRoot, "2026-03-06T01:10:00.000Z");

  const promoted = loadActivationPointers(activationRoot).pointers;
  assert.equal(promoted.active?.packId, candidatePack.manifest.packId);
  assert.equal(promoted.previous?.packId, activePack.manifest.packId);
  assert.equal(promoted.candidate, null);
  assert.deepEqual(promoted.active?.eventRange, {
    start: 20,
    end: 21,
    count: 2
  });
  assert.equal(promoted.active?.eventExportDigest, candidatePack.manifest.provenance.eventExports?.exportDigest ?? null);
  assert.deepEqual(promoted.previous?.eventRange, {
    start: 10,
    end: 11,
    count: 2
  });

  rollbackActivePack(activationRoot, "2026-03-06T01:15:00.000Z");

  const rolledBack = loadActivationPointers(activationRoot).pointers;
  const activeAfterRollback = loadPackFromActivation(activationRoot, "active", { requireActivationReady: true });
  const activeTarget = describeActivationTarget(activationRoot, "active", { requireActivationReady: true });
  assert.equal(rolledBack.active?.packId, activePack.manifest.packId);
  assert.equal(rolledBack.candidate?.packId, candidatePack.manifest.packId);
  assert.equal(rolledBack.previous, null);
  assert.deepEqual(rolledBack.active?.eventRange, {
    start: 10,
    end: 11,
    count: 2
  });
  assert.deepEqual(rolledBack.candidate?.eventRange, {
    start: 20,
    end: 21,
    count: 2
  });
  assert.equal(activeAfterRollback?.manifest.packId, activePack.manifest.packId);
  assert.equal(activeTarget?.packId, activePack.manifest.packId);
});

test("learned-routing packs with stub router metadata can be staged but not activated", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  materializeTestPack(activeRoot, {
    packId: "pack-active-test",
    learnedRouting: false,
    eventStart: 30
  });
  const candidatePack = materializeTestPack(candidateRoot, {
    packId: "pack-candidate-stub-test",
    learnedRouting: true,
    eventStart: 40
  });

  writePackFile(candidateRoot, PACK_LAYOUT.manifest, {
    ...candidatePack.manifest,
    runtimeAssets: {
      ...candidatePack.manifest.runtimeAssets,
      router: {
        ...candidatePack.manifest.runtimeAssets.router,
        kind: "stub"
      }
    }
  });

  const readinessErrors = validatePackActivationReadiness(candidateRoot);
  assert.deepEqual(readinessErrors, ["learned-routing packs require runtimeAssets.router.kind=artifact for activation"]);

  activatePack(activationRoot, activeRoot, "2026-03-06T02:00:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T02:05:00.000Z");

  const inspection = inspectActivationState(activationRoot, "2026-03-06T02:06:00.000Z");
  assert.equal(inspection.candidate?.activationReady, false);
  assert.match(
    inspection.candidate?.findings.join("; ") ?? "",
    /runtimeAssets\.router\.kind=artifact/
  );
  assert.equal(inspection.promotion.allowed, false);
  assert.match(inspection.promotion.findings.join("; "), /runtimeAssets\.router\.kind=artifact/);
  assert.throws(() => promoteCandidatePack(activationRoot, "2026-03-06T02:10:00.000Z"), /Promotion blocked/);
  assert.throws(() => activatePack(activationRoot, candidateRoot, "2026-03-06T02:15:00.000Z"), /Pack is not activation-ready/);
});

test("activation inspection surfaces missing learned route artifacts on the active slot", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));

  materializeTestPack(activeRoot, {
    packId: "pack-active-missing-route",
    learnedRouting: true,
    eventStart: 50
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T02:20:00.000Z");
  rmSync(path.join(activeRoot, "router", "model.json"), { force: true });

  const inspection = inspectActivationState(activationRoot, "2026-03-06T02:21:00.000Z");
  assert.equal(inspection.active?.activationReady, false);
  assert.match(inspection.active?.findings.join("; ") ?? "", /router payload not found/);
  assert.throws(
    () => loadPackFromActivation(activationRoot, "active", { requireActivationReady: true }),
    /router payload not found/
  );
});

test("promotion blocks stale candidate provenance from displacing a newer active pack", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  materializeTestPack(activeRoot, {
    packId: "pack-active-newer",
    learnedRouting: false,
    eventStart: 20,
    builtAt: "2026-03-06T04:00:00.000Z"
  });
  materializeTestPack(candidateRoot, {
    packId: "pack-candidate-stale",
    learnedRouting: true,
    eventStart: 10,
    builtAt: "2026-03-06T03:00:00.000Z"
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T04:05:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T04:10:00.000Z");

  const inspection = inspectActivationState(activationRoot, "2026-03-06T04:11:00.000Z");

  assert.equal(inspection.promotion.allowed, false);
  assert.match(inspection.promotion.findings.join("; "), /candidate pack builtAt must not precede active pack builtAt/);
  assert.match(inspection.promotion.findings.join("; "), /candidate eventRange\.end must be >= active eventRange\.end/);
  assert.throws(() => promoteCandidatePack(activationRoot, "2026-03-06T04:15:00.000Z"), /Promotion blocked/);
});

test("promotion blocks candidates whose manifests drift after staging", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  materializeTestPack(activeRoot, {
    packId: "pack-active-stable",
    learnedRouting: false,
    eventStart: 10
  });
  const candidatePack = materializeTestPack(candidateRoot, {
    packId: "pack-candidate-drifted",
    learnedRouting: true,
    eventStart: 20
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T06:00:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T06:05:00.000Z");

  writePackFile(candidateRoot, PACK_LAYOUT.manifest, {
    ...candidatePack.manifest,
    modelFingerprints: [...candidatePack.manifest.modelFingerprints, "mutated-after-staging"]
  });

  const inspection = inspectActivationState(activationRoot, "2026-03-06T06:06:00.000Z");
  assert.equal(inspection.candidate?.activationReady, false);
  assert.match(inspection.candidate?.findings.join("; ") ?? "", /manifestDigest/);
  assert.equal(inspection.promotion.allowed, false);
  assert.match(inspection.promotion.findings.join("; "), /manifestDigest/);
  assert.throws(() => promoteCandidatePack(activationRoot, "2026-03-06T06:10:00.000Z"), /manifestDigest/);
});

test("rollback blocks previous packs whose manifests drift after promotion", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activePack = materializeTestPack(activeRoot, {
    packId: "pack-active-rollback-source",
    learnedRouting: false,
    eventStart: 10
  });
  materializeTestPack(candidateRoot, {
    packId: "pack-candidate-rollback-target",
    learnedRouting: true,
    eventStart: 20
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T07:00:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T07:05:00.000Z");
  promoteCandidatePack(activationRoot, "2026-03-06T07:10:00.000Z");

  writePackFile(activeRoot, PACK_LAYOUT.manifest, {
    ...activePack.manifest,
    provenance: {
      ...activePack.manifest.provenance,
      offlineArtifacts: [...activePack.manifest.provenance.offlineArtifacts, "mutated-before-rollback"]
    }
  });

  const inspection = inspectActivationState(activationRoot, "2026-03-06T07:11:00.000Z");
  assert.equal(inspection.rollback.allowed, false);
  assert.match(inspection.previous?.findings.join("; ") ?? "", /manifestDigest/);
  assert.match(inspection.rollback.findings.join("; "), /manifestDigest/);
  assert.throws(() => rollbackActivePack(activationRoot, "2026-03-06T07:15:00.000Z"), /manifestDigest/);
});


test("staging blocks when the retained active pointer drifts from its pinned manifest", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));

  const activePack = materializeTestPack(activeRoot, {
    packId: "pack-active-staging-pinned",
    learnedRouting: false,
    eventStart: 10
  });
  materializeTestPack(candidateRoot, {
    packId: "pack-candidate-staging-next",
    learnedRouting: true,
    eventStart: 20
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T08:00:00.000Z");

  writePackFile(activeRoot, PACK_LAYOUT.manifest, {
    ...activePack.manifest,
    modelFingerprints: [...activePack.manifest.modelFingerprints, "mutated-before-staging"]
  });

  assert.throws(
    () => stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T08:05:00.000Z"),
    /active pointer cannot be retained: Invalid activation pointer: .*manifestDigest/
  );
});

test("staging blocks when the retained previous pointer drifts from its pinned manifest", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));
  const candidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));
  const nextCandidateRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-candidate-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));
  t.after(() => rmSync(candidateRoot, { recursive: true, force: true }));
  t.after(() => rmSync(nextCandidateRoot, { recursive: true, force: true }));

  const activePack = materializeTestPack(activeRoot, {
    packId: "pack-previous-staging-source",
    learnedRouting: false,
    eventStart: 10
  });
  materializeTestPack(candidateRoot, {
    packId: "pack-active-staging-current",
    learnedRouting: true,
    eventStart: 20
  });
  materializeTestPack(nextCandidateRoot, {
    packId: "pack-candidate-staging-future",
    learnedRouting: true,
    eventStart: 30
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T09:00:00.000Z");
  stageCandidatePack(activationRoot, candidateRoot, "2026-03-06T09:05:00.000Z");
  promoteCandidatePack(activationRoot, "2026-03-06T09:10:00.000Z");

  writePackFile(activeRoot, PACK_LAYOUT.manifest, {
    ...activePack.manifest,
    provenance: {
      ...activePack.manifest.provenance,
      offlineArtifacts: [...activePack.manifest.provenance.offlineArtifacts, "mutated-before-restaging"]
    }
  });

  assert.throws(
    () => stageCandidatePack(activationRoot, nextCandidateRoot, "2026-03-06T09:15:00.000Z"),
    /previous pointer cannot be retained: Invalid activation pointer: .*manifestDigest/
  );
});

test("activation blocks reusing an active packId after its pinned manifest drifts", (t) => {
  const activationRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-activation-"));
  const activeRoot = mkdtempSync(path.join(tmpdir(), "openclawbrain-ts-active-"));

  t.after(() => rmSync(activationRoot, { recursive: true, force: true }));
  t.after(() => rmSync(activeRoot, { recursive: true, force: true }));

  const activePack = materializeTestPack(activeRoot, {
    packId: "pack-active-manifest-pinned",
    learnedRouting: true,
    eventStart: 10
  });

  activatePack(activationRoot, activeRoot, "2026-03-06T10:00:00.000Z");

  writePackFile(activeRoot, PACK_LAYOUT.manifest, {
    ...activePack.manifest,
    modelFingerprints: [...activePack.manifest.modelFingerprints, "mutated-before-reactivation"]
  });

  assert.throws(
    () => activatePack(activationRoot, activeRoot, "2026-03-06T10:05:00.000Z"),
    /active pointer for packId pack-active-manifest-pinned is already pinned to a different manifest: .*manifestDigest/
  );
});
