import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  buildNormalizedEventExport,
  CONTRACT_IDS,
  FIXTURE_ACTIVATION_POINTERS,
  FIXTURE_ARTIFACT_MANIFEST,
  FIXTURE_FEEDBACK_EVENT,
  FIXTURE_FEEDBACK_EVENTS,
  FIXTURE_INTERACTION_EVENT,
  FIXTURE_INTERACTION_EVENTS,
  FIXTURE_NORMALIZED_EVENT_EXPORT,
  FIXTURE_TEACHER_SUPERVISION_ARTIFACT,
  FIXTURE_WORKSPACE_METADATA,
  FIXTURE_PACK_GRAPH,
  FIXTURE_PACK_VECTORS,
  FIXTURE_ROUTER_ARTIFACT,
  FIXTURE_RUNTIME_COMPILE_REQUEST,
  FIXTURE_RUNTIME_COMPILE_RESPONSE,
  validateActivationPointers,
  validateArtifactManifest,
  validateFeedbackEvent,
  validateInteractionEvent,
  validateLearningSurface,
  validateNormalizedEventExport,
  validatePackGraphPayload,
  validatePackVectorsPayload,
  validateRouterArtifact,
  validateTeacherSupervisionArtifact,
  validateRuntimeCompileExpectation,
  validateRuntimeCompileRequest,
  validateRuntimeCompileResponse,
  validateRuntimeCompileTargetExpectation,
  validateWorkspaceMetadata
} from "@openclawbrain/contracts";

test("canonical fixtures validate end-to-end", () => {
  assert.deepEqual(validateActivationPointers(FIXTURE_ACTIVATION_POINTERS), []);
  assert.deepEqual(validateRuntimeCompileRequest(FIXTURE_RUNTIME_COMPILE_REQUEST), []);
  assert.deepEqual(validateRuntimeCompileResponse(FIXTURE_RUNTIME_COMPILE_RESPONSE), []);
  assert.deepEqual(validateInteractionEvent(FIXTURE_INTERACTION_EVENT), []);
  assert.deepEqual(validateFeedbackEvent(FIXTURE_FEEDBACK_EVENT), []);
  assert.deepEqual(validateLearningSurface(FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.learningSurface), []);
  assert.deepEqual(validateNormalizedEventExport(FIXTURE_NORMALIZED_EVENT_EXPORT), []);
  assert.deepEqual(validateTeacherSupervisionArtifact(FIXTURE_TEACHER_SUPERVISION_ARTIFACT), []);
  assert.deepEqual(validateWorkspaceMetadata(FIXTURE_WORKSPACE_METADATA), []);
  assert.deepEqual(validateArtifactManifest(FIXTURE_ARTIFACT_MANIFEST), []);
  assert.deepEqual(validatePackGraphPayload(FIXTURE_PACK_GRAPH, FIXTURE_ARTIFACT_MANIFEST.packId), []);
  assert.deepEqual(validatePackVectorsPayload(FIXTURE_PACK_VECTORS, FIXTURE_PACK_GRAPH), []);
  assert.deepEqual(validateRouterArtifact(FIXTURE_ROUTER_ARTIFACT, FIXTURE_ARTIFACT_MANIFEST), []);
});

test("teacher supervision artifacts validate freshness and dedup metadata", () => {
  assert.deepEqual(validateTeacherSupervisionArtifact(FIXTURE_TEACHER_SUPERVISION_ARTIFACT), []);

  assert.deepEqual(
    validateTeacherSupervisionArtifact({
      ...FIXTURE_TEACHER_SUPERVISION_ARTIFACT,
      artifactId: "",
      freshness: {
        ...FIXTURE_TEACHER_SUPERVISION_ARTIFACT.freshness,
        ageMs: -1
      }
    }),
    ["teacher supervision artifactId is required", "teacher supervision freshness.ageMs must be non-negative"]
  );
});

test("canonical fixture sources point at current repo docs", () => {
  const sources = FIXTURE_PACK_GRAPH.blocks.map((block) => block.source);
  const attachQuickstartDoc = fileURLToPath(new URL("../../../../docs/openclaw-attach-quickstart.md", import.meta.url));
  const contractsDoc = fileURLToPath(new URL("../../../../docs/contracts-v1.md", import.meta.url));
  const convergenceDoc = fileURLToPath(new URL("../../../../docs/learning-first-convergence.md", import.meta.url));

  assert.match(sources.join("\n"), /docs\/openclaw-attach-quickstart\.md/);
  assert.match(sources.join("\n"), /docs\/contracts-v1\.md/);
  assert.match(sources.join("\n"), /docs\/learning-first-convergence\.md/);
  assert.equal(existsSync(attachQuickstartDoc), true);
  assert.equal(existsSync(contractsDoc), true);
  assert.equal(existsSync(convergenceDoc), true);
  assert.equal(sources.some((source) => source.includes("openclawbrain-openclaw-rearchitecture")), false);
  assert.equal(sources.includes("memory/2026-03-05-openclawbrain-vnext-roadmap.md"), false);
});

test("runtime compile golden response stays in sync with the TS fixture", () => {
  const golden = JSON.parse(
    readFileSync(fileURLToPath(new URL("../../../../contracts/runtime_compile/v1/golden-response.json", import.meta.url)), "utf8")
  ) as typeof FIXTURE_RUNTIME_COMPILE_RESPONSE;

  assert.deepEqual(validateRuntimeCompileResponse(golden), []);
  assert.deepEqual(golden, FIXTURE_RUNTIME_COMPILE_RESPONSE);
});

test("normalized event export metadata is deterministically derived from interaction and feedback events", () => {
  const rebuilt = buildNormalizedEventExport({
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS
  });

  assert.deepEqual(rebuilt, FIXTURE_NORMALIZED_EVENT_EXPORT);
  assert.equal(rebuilt.range.start, FIXTURE_INTERACTION_EVENTS[0]?.sequence);
  assert.equal(rebuilt.range.end, FIXTURE_FEEDBACK_EVENTS[1]?.sequence);
  assert.equal(rebuilt.provenance.interactionCount, FIXTURE_INTERACTION_EVENTS.length);
  assert.equal(rebuilt.provenance.feedbackCount, FIXTURE_FEEDBACK_EVENTS.length);
  assert.deepEqual(rebuilt.provenance.contracts, [CONTRACT_IDS.interactionEvents, CONTRACT_IDS.feedbackEvents]);
  assert.equal(rebuilt.provenance.learningSurface.bootProfile, "fast_boot_defaults");
  assert.equal(rebuilt.provenance.learningSurface.learningCadence, "passive_background");
  assert.equal(rebuilt.provenance.learningSurface.labelHarvest.humanLabels, FIXTURE_FEEDBACK_EVENTS.length);
  assert.equal(rebuilt.provenance.learningSurface.labelHarvest.selfLabels, 1);
  assert.match(rebuilt.provenance.learningSurface.scanSurfaces.join("\n"), /memory_compiled/);
});

test("learned routing responses must explicitly mark route_fn usage", () => {
  const errors = validateRuntimeCompileResponse({
    contract: CONTRACT_IDS.runtimeCompile,
    packId: "pack-active",
    selectedContext: [],
    diagnostics: {
      modeRequested: "learned",
      modeEffective: "learned",
      usedLearnedRouteFn: false,
      routerIdentity: "router-stub",
      candidateCount: 0,
      selectedCount: 0,
      selectedCharCount: 0,
      selectedTokenCount: 0,
      selectionStrategy: "pack_route_fn_selection_v1",
      selectionDigest: "sha256-empty",
      compactionMode: "native",
      compactionApplied: false,
      notes: []
    }
  });

  assert.deepEqual(errors, ["learned mode requires usedLearnedRouteFn=true"]);
});

test("runtime compile responses reject overlapping compacted coverage", () => {
  const selectedContext = [
    {
      id: "ctx-summary",
      source: "pack/runtime-summary",
      text: "Summary coverage for ctx-a and ctx-b.",
      tokenCount: 6,
      compactedFrom: ["ctx-a", "ctx-b"]
    },
    {
      id: "ctx-a",
      source: "memory/runtime-detail",
      text: "Detailed runtime context for ctx-a.",
      tokenCount: 5
    }
  ];

  const errors = validateRuntimeCompileResponse({
    ...FIXTURE_RUNTIME_COMPILE_RESPONSE,
    selectedContext,
    diagnostics: {
      ...FIXTURE_RUNTIME_COMPILE_RESPONSE.diagnostics,
      selectedCount: selectedContext.length,
      selectedCharCount: selectedContext.reduce((sum, block) => sum + block.text.length, 0),
      selectedTokenCount: selectedContext.reduce((sum, block) => sum + (block.tokenCount ?? 0), 0),
      selectionDigest: "sha256-overlap"
    }
  });

  assert.deepEqual(errors, ["selectedContext[1] overlaps block ctx-summary via ctx-a"]);
});

test("runtime compile expectations validate shape and target compatibility", () => {
  const target = {
    packId: FIXTURE_ARTIFACT_MANIFEST.packId,
    routePolicy: FIXTURE_ARTIFACT_MANIFEST.routePolicy,
    routerIdentity: FIXTURE_ARTIFACT_MANIFEST.runtimeAssets.router.identity,
    workspaceSnapshot: FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot,
    workspaceRevision: FIXTURE_ARTIFACT_MANIFEST.provenance.workspace.revision,
    eventRange: {
      start: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.start,
      end: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.end,
      count: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.count
    },
    eventExportDigest: FIXTURE_ARTIFACT_MANIFEST.provenance.eventExports?.exportDigest ?? null,
    builtAt: FIXTURE_ARTIFACT_MANIFEST.provenance.builtAt
  };

  assert.deepEqual(
    validateRuntimeCompileExpectation({
      packId: FIXTURE_ARTIFACT_MANIFEST.packId,
      workspaceSnapshot: FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot,
      builtAt: FIXTURE_ARTIFACT_MANIFEST.provenance.builtAt
    }),
    []
  );

  assert.deepEqual(validateRuntimeCompileExpectation({ packId: "", builtAt: "not-an-iso-date" }), [
    "runtime compile expectation packId must be non-empty when set",
    "runtime compile expectation builtAt must be an ISO timestamp when set"
  ]);

  assert.deepEqual(
    validateRuntimeCompileTargetExpectation(target, {
      packId: FIXTURE_ARTIFACT_MANIFEST.packId,
      workspaceSnapshot: FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot,
      eventRange: {
        start: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.start,
        end: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.end,
        count: FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.count
      }
    }),
    []
  );

  assert.deepEqual(
    validateRuntimeCompileTargetExpectation(target, {
      workspaceSnapshot: "workspace-stale@snapshot",
      eventRange: {
        start: 1,
        end: 2,
        count: 3
      }
    }),
    [
      `runtime compile target workspaceSnapshot ${FIXTURE_ARTIFACT_MANIFEST.provenance.workspaceSnapshot} does not match expected workspace-stale@snapshot`,
      `runtime compile target eventRange.start ${FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.start} does not match expected 1`,
      `runtime compile target eventRange.end ${FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.end} does not match expected 2`,
      `runtime compile target eventRange.count ${FIXTURE_ARTIFACT_MANIFEST.provenance.eventRange.count} does not match expected 3`
    ]
  );
});

test("runtime compile requests reject empty activePackId", () => {
  const errors = validateRuntimeCompileRequest({
    ...FIXTURE_RUNTIME_COMPILE_REQUEST,
    activePackId: ""
  });

  assert.deepEqual(errors, ["activePackId must be non-empty when set"]);
});

test("activation pointers reject duplicate pack ids across slots", () => {
  const candidate = FIXTURE_ACTIVATION_POINTERS.candidate;
  if (candidate === null) {
    throw new Error("candidate fixture is required for this test");
  }

  const errors = validateActivationPointers({
    contract: CONTRACT_IDS.activationPointers,
    active: FIXTURE_ACTIVATION_POINTERS.active,
    candidate: {
      slot: candidate.slot,
      packRootDir: candidate.packRootDir,
      manifestPath: candidate.manifestPath,
      manifestDigest: candidate.manifestDigest,
      routePolicy: candidate.routePolicy,
      routerIdentity: candidate.routerIdentity,
      workspaceSnapshot: candidate.workspaceSnapshot,
      workspaceRevision: candidate.workspaceRevision,
      eventRange: candidate.eventRange,
      eventExportDigest: candidate.eventExportDigest,
      builtAt: candidate.builtAt,
      updatedAt: candidate.updatedAt,
      packId: FIXTURE_ACTIVATION_POINTERS.active?.packId ?? "pack-active"
    },
    previous: null
  });

  assert.deepEqual(errors, ["activation pointers must not reuse packId across slots: pack-active"]);
});

test("activation pointers require manifest digests", () => {
  const active = FIXTURE_ACTIVATION_POINTERS.active;
  if (active === null) {
    throw new Error("active fixture is required for this test");
  }

  const errors = validateActivationPointers({
    contract: CONTRACT_IDS.activationPointers,
    active: {
      ...active,
      manifestDigest: "manifest-without-checksum-prefix"
    },
    candidate: FIXTURE_ACTIVATION_POINTERS.candidate,
    previous: FIXTURE_ACTIVATION_POINTERS.previous
  });

  assert.deepEqual(errors, ["activation pointer manifestDigest must be a sha256 digest"]);
});
