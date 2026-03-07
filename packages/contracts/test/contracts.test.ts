import assert from "node:assert/strict";
import test from "node:test";

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
  validateRuntimeCompileRequest,
  validateRuntimeCompileResponse,
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
  assert.deepEqual(validateWorkspaceMetadata(FIXTURE_WORKSPACE_METADATA), []);
  assert.deepEqual(validateArtifactManifest(FIXTURE_ARTIFACT_MANIFEST), []);
  assert.deepEqual(validatePackGraphPayload(FIXTURE_PACK_GRAPH, FIXTURE_ARTIFACT_MANIFEST.packId), []);
  assert.deepEqual(validatePackVectorsPayload(FIXTURE_PACK_VECTORS, FIXTURE_PACK_GRAPH), []);
  assert.deepEqual(validateRouterArtifact(FIXTURE_ROUTER_ARTIFACT, FIXTURE_ARTIFACT_MANIFEST), []);
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
      selectionStrategy: "pack_keyword_overlap_v1",
      selectionDigest: "sha256-empty",
      compactionMode: "native",
      compactionApplied: false,
      notes: []
    }
  });

  assert.deepEqual(errors, ["learned mode requires usedLearnedRouteFn=true"]);
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
