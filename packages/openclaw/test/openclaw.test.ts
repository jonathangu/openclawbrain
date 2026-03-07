import assert from "node:assert/strict";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CONTRACT_IDS,
  FIXTURE_FEEDBACK_EVENTS,
  FIXTURE_INTERACTION_EVENTS,
  type NormalizedEventExportV1
} from "@openclawbrain/contracts";
import { materializeAlwaysOnLearningCandidatePack, materializeCandidatePack } from "@openclawbrain/learner";
import { activatePack } from "@openclawbrain/pack-format";
import {
  buildNormalizedRuntimeEventExport,
  buildCanonicalSupervision,
  classifyFeedbackKind,
  compileRuntimeContext,
  createAsyncTeacherLiveLoop,
  loadRuntimeEventExportBundle,
  resolveActivePackForCompile,
  runContinuousProductLoopTurn,
  runRuntimeTurn
} from "@openclawbrain/openclaw";

function materializeActivePack(rootDir: string, activationRoot: string): { packId: string } {
  const pack = materializeCandidatePack(rootDir, {
    packLabel: "runtime-openclaw",
    workspace: {
      workspaceId: "workspace-openclaw",
      snapshotId: "workspace-openclaw@snapshot-1",
      capturedAt: "2026-03-07T17:00:00.000Z",
      rootDir: "/workspace/openclawbrain",
      branch: "main",
      revision: "runtime-openclaw-rev",
      labels: ["openclaw", "runtime"]
    },
    eventRange: {
      start: 101,
      end: 104
    },
    eventExports: {
      interactionEvents: FIXTURE_INTERACTION_EVENTS,
      feedbackEvents: FIXTURE_FEEDBACK_EVENTS
    },
    learnedRouting: true
  });

  activatePack(activationRoot, rootDir, "2026-03-07T17:00:00.000Z");

  return {
    packId: pack.manifest.packId
  };
}

function mkdtemp(prefix: string): string {
  const rootDir = path.join(tmpdir(), `${prefix}${Date.now()}-${Math.random().toString(16).slice(2)}`);
  mkdirSync(rootDir, { recursive: true });
  return rootDir;
}

function expectNormalizedEventExport(result: ReturnType<typeof runRuntimeTurn>["eventExport"]): NormalizedEventExportV1 {
  assert.equal(result.ok, true);
  return result.normalizedEventExport;
}

test("compileRuntimeContext consumes the active pack through activation pointers", (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-compile-");
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const { packId } = materializeActivePack(activePackRoot, activationRoot);
  const target = resolveActivePackForCompile(activationRoot);
  assert.equal(target.activePointer.packId, packId);
  assert.equal(target.inspection.packId, packId);

  const result = compileRuntimeContext({
    activationRoot,
    agentId: "runtime-test",
    message: "feedback scanner route gating",
    maxContextBlocks: 2,
    runtimeHints: ["feedback scanner"]
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }

  assert.equal(result.activePackId, packId);
  assert.equal(result.compileResponse.contract, CONTRACT_IDS.runtimeCompile);
  assert.equal(result.compileResponse.selectedContext.length > 0, true);
  assert.match(result.compileResponse.diagnostics.notes.join("; "), /activation_slot=active/);
  assert.match(result.compileResponse.diagnostics.notes.join("; "), new RegExp(`target_pack_id=${packId}`));
  assert.match(result.compileResponse.diagnostics.notes.join("; "), /target_route_policy=requires_learned_routing/);
  assert.match(result.compileResponse.diagnostics.notes.join("; "), new RegExp(`target_router_identity=${packId}:route_fn`));
  assert.match(result.compileResponse.diagnostics.notes.join("; "), /brain_boundary=promoted_pack_compile_only/);
  assert.match(result.compileResponse.diagnostics.notes.join("; "), /OpenClaw remains the runtime owner/);
  assert.match(result.brainContext, /^\[BRAIN_CONTEXT v1\]/);
  assert.match(result.brainContext, /PACK_ID:/);
  assert.match(result.brainContext, /SOURCE:/);
});

test("compileRuntimeContext fails open when no active pack is available", () => {
  const activationRoot = path.join(tmpdir(), `openclawbrain-openclaw-missing-${Date.now()}`);
  const result = compileRuntimeContext({
    activationRoot,
    message: "hello world"
  });

  assert.equal(result.ok, false);
  assert.equal(result.fallbackToStaticContext, true);
  assert.equal(result.hardRequirementViolated, false);
  assert.equal(result.brainContext, "");
  assert.match(result.error, /No active pack pointer found/);
});

test("compileRuntimeContext hard-fails learned-required packs when the active route artifact disappears", (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-hard-fail-");
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const { packId } = materializeActivePack(activePackRoot, activationRoot);
  rmSync(path.join(activePackRoot, "router", "model.json"), { force: true });

  const result = compileRuntimeContext({
    activationRoot,
    agentId: "runtime-hard-fail",
    message: "feedback scanner route gating"
  });

  assert.equal(result.ok, false);
  assert.equal(result.fallbackToStaticContext, false);
  assert.equal(result.hardRequirementViolated, true);
  assert.match(result.error, new RegExp(`Learned-routing hotpath hard requirement violated for active pack ${packId}`));
  assert.match(result.error, /router payload not found/);
});

test("buildNormalizedRuntimeEventExport emits compile, delivery, and feedback events", (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-export-");
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  materializeActivePack(activePackRoot, activationRoot);
  const compileResult = compileRuntimeContext({
    activationRoot,
    agentId: "runtime-test",
    message: "feedback scanner route gating",
    runtimeHints: ["feedback scanner"]
  });
  assert.equal(compileResult.ok, true);

  const normalizedEventExport = buildNormalizedRuntimeEventExport(
    {
      agentId: "runtime-test",
      sessionId: "session-live-1",
      channel: "whatsapp",
      userMessage: "feedback scanner route gating",
      sequenceStart: 501,
      compile: {
        createdAt: "2026-03-07T15:00:00.000Z"
      },
      delivery: {
        createdAt: "2026-03-07T15:03:00.000Z",
        messageId: "msg-live-1"
      },
      feedback: [
        {
          createdAt: "2026-03-07T15:02:00.000Z",
          content: "Use the unified feedback scanner before enabling default loop scans."
        }
      ]
    },
    compileResult
  );

  assert.equal(normalizedEventExport.interactionEvents.length, 2);
  assert.equal(normalizedEventExport.feedbackEvents.length, 1);
  assert.equal(normalizedEventExport.interactionEvents[0]?.kind, "memory_compiled");
  assert.equal(normalizedEventExport.interactionEvents[1]?.kind, "message_delivered");
  assert.equal(normalizedEventExport.feedbackEvents[0]?.kind, "teaching");
  assert.equal(normalizedEventExport.feedbackEvents[0]?.relatedInteractionId, normalizedEventExport.interactionEvents[0]?.eventId);
  assert.equal(normalizedEventExport.range.start, 501);
  assert.equal(normalizedEventExport.range.end, 503);
});

test("runRuntimeTurn fails open on compile while still exporting delivery and feedback events", (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-turn-");
  const activationRoot = path.join(rootDir, "missing-activation");
  const exportRoot = path.join(rootDir, "event-export");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const result = runRuntimeTurn(
    {
      agentId: "runtime-test",
      sessionId: "session-live-2",
      channel: "whatsapp",
      userMessage: "hello world",
      sequenceStart: 700,
      compile: {
        createdAt: "2026-03-07T16:00:00.000Z"
      },
      delivery: {
        createdAt: "2026-03-07T16:02:00.000Z",
        messageId: "msg-live-2"
      },
      feedback: [
        {
          createdAt: "2026-03-07T16:01:00.000Z",
          content: "No — do this instead."
        }
      ],
      export: {
        rootDir: exportRoot,
        exportName: "session-live-2-export",
        exportedAt: "2026-03-07T16:03:00.000Z"
      }
    },
    {
      activationRoot
    }
  );

  assert.equal(result.ok, false);
  assert.equal(result.fallbackToStaticContext, true);
  assert.equal(result.hardRequirementViolated, false);
  const normalizedEventExport = expectNormalizedEventExport(result.eventExport);
  assert.equal(normalizedEventExport.interactionEvents.length, 1);
  assert.equal(normalizedEventExport.feedbackEvents.length, 1);
  assert.equal(normalizedEventExport.interactionEvents[0]?.kind, "message_delivered");
  assert.equal(normalizedEventExport.feedbackEvents[0]?.kind, "correction");

  const bundle = loadRuntimeEventExportBundle(exportRoot);
  assert.equal(bundle.manifest.exportName, "session-live-2-export");
  assert.equal(bundle.normalizedEventExport.provenance.interactionCount, 1);
  assert.equal(bundle.normalizedEventExport.provenance.feedbackCount, 1);
});

test("runRuntimeTurn keeps compile output when event-export bundle writing fails", (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-turn-");
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  const exportRoot = path.join(rootDir, "occupied-root");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  materializeActivePack(activePackRoot, activationRoot);
  writeFileSync(exportRoot, "occupied");

  const result = runRuntimeTurn(
    {
      agentId: "runtime-test",
      sessionId: "session-live-3",
      channel: "whatsapp",
      userMessage: "feedback scanner route gating",
      compile: {
        createdAt: "2026-03-07T17:01:00.000Z"
      },
      delivery: {
        createdAt: "2026-03-07T17:02:00.000Z",
        messageId: "msg-live-3"
      },
      export: {
        rootDir: exportRoot,
        exportName: "session-live-3-export",
        exportedAt: "2026-03-07T17:03:00.000Z"
      }
    },
    {
      activationRoot
    }
  );

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }

  assert.equal(result.eventExport.ok, false);
  assert.equal(result.warnings.length, 1);
  assert.match(result.eventExport.error, /EEXIST|not a directory/i);
  assert.match(result.brainContext, /^\[BRAIN_CONTEXT v1\]/);
});

test("runRuntimeTurn throws when a learned-required active route artifact is missing", (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-turn-hard-fail-");
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const { packId } = materializeActivePack(activePackRoot, activationRoot);
  rmSync(path.join(activePackRoot, "router", "model.json"), { force: true });

  assert.throws(
    () =>
      runRuntimeTurn(
        {
          agentId: "runtime-test",
          sessionId: "session-hard-fail",
          channel: "whatsapp",
          userMessage: "feedback scanner route gating"
        },
        {
          activationRoot
        }
      ),
    new RegExp(`Learned-routing hotpath hard requirement violated for active pack ${packId}`)
  );
});

test("runContinuousProductLoopTurn promotes fresher learned packs and later compiles use the newer route artifact", (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-product-loop-");
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  const loopRoot = path.join(rootDir, "product-loop");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const { packId: seedPackId } = materializeActivePack(activePackRoot, activationRoot);

  const first = runContinuousProductLoopTurn({
    activationRoot,
    loopRoot,
    packLabel: "post-attach-loop",
    workspace: {
      workspaceId: "workspace-openclaw-loop",
      snapshotId: "workspace-openclaw-loop@snapshot-1",
      capturedAt: "2026-03-07T18:00:30.000Z",
      rootDir: "/workspace/openclawbrain",
      branch: "main",
      revision: "runtime-loop-rev-1",
      labels: ["openclaw", "runtime", "loop"]
    },
    turn: {
      agentId: "runtime-loop",
      sessionId: "session-product-loop",
      channel: "whatsapp",
      userMessage: "Compile freshness evidence before promotion.",
      runtimeHints: ["freshness", "promotion", "evidence"],
      sequenceStart: 801,
      compile: {
        createdAt: "2026-03-07T18:00:00.000Z"
      },
      delivery: {
        createdAt: "2026-03-07T18:01:00.000Z",
        messageId: "msg-loop-1"
      },
      feedback: [
        {
          createdAt: "2026-03-07T18:02:00.000Z",
          content: "Prefer the fresher learned route artifact after promotion when compiling freshness evidence."
        }
      ]
    },
    candidateBuiltAt: "2026-03-07T18:03:00.000Z",
    stageUpdatedAt: "2026-03-07T18:04:00.000Z",
    promoteUpdatedAt: "2026-03-07T18:05:00.000Z"
  });

  assert.equal(first.compileActiveVersion, 1);
  assert.equal(first.compileActivePackId, seedPackId);
  assert.equal(first.turn.eventExport.ok, true);
  if (!first.turn.eventExport.ok || !first.turn.eventExport.wroteBundle) {
    throw new Error("first product-loop turn should write a real event-export bundle");
  }

  const firstBundle = loadRuntimeEventExportBundle(first.turn.eventExport.rootDir);
  const firstSupervision = buildCanonicalSupervision(firstBundle.normalizedEventExport);
  assert.deepEqual(first.supervision, firstSupervision);
  assert.equal(first.supervision?.humanLabelCount, 1);
  assert.equal(first.supervision?.feedbackCounts.teachings, 1);
  assert.equal(first.learning.promoted, true);
  assert.equal(first.learning.promotionAllowed, true);
  assert.equal(first.state.activePackVersion, 2);
  assert.equal(first.state.packLineage.length, 2);
  assert.notEqual(first.state.currentActivePack?.packId, seedPackId);
  assert.notEqual(first.state.currentActivePack?.routerIdentity, `${seedPackId}:route_fn`);

  const promotedAfterFirst = first.state.currentActivePack;
  if (promotedAfterFirst === null) {
    throw new Error("first product-loop turn should promote a fresher active pack");
  }

  const second = runContinuousProductLoopTurn({
    activationRoot,
    loopRoot,
    packLabel: "post-attach-loop",
    workspace: {
      workspaceId: "workspace-openclaw-loop",
      snapshotId: "workspace-openclaw-loop@snapshot-2",
      capturedAt: "2026-03-07T18:10:30.000Z",
      rootDir: "/workspace/openclawbrain",
      branch: "main",
      revision: "runtime-loop-rev-2",
      labels: ["openclaw", "runtime", "loop", "follow-up"]
    },
    state: first.state,
    turn: {
      agentId: "runtime-loop",
      sessionId: "session-product-loop",
      channel: "whatsapp",
      userMessage: "Compile the fresher learned route artifact after promotion.",
      runtimeHints: ["freshness", "promotion", "learned", "route", "artifact"],
      sequenceStart: 804,
      compile: {
        createdAt: "2026-03-07T18:10:00.000Z"
      },
      delivery: {
        createdAt: "2026-03-07T18:11:00.000Z",
        messageId: "msg-loop-2"
      },
      feedback: [
        {
          createdAt: "2026-03-07T18:12:00.000Z",
          content: "Looks good. Keep the continuous post-attach product loop live."
        }
      ]
    },
    candidateBuiltAt: "2026-03-07T18:13:00.000Z",
    stageUpdatedAt: "2026-03-07T18:14:00.000Z",
    promoteUpdatedAt: "2026-03-07T18:15:00.000Z"
  });

  assert.equal(second.compileActiveVersion, 2);
  assert.equal(second.compileActivePackId, promotedAfterFirst.packId);
  assert.equal(second.turn.ok, true);
  if (!second.turn.ok) {
    return;
  }

  assert.equal(second.turn.activePackId, promotedAfterFirst.packId);
  assert.equal(second.turn.compileResponse.diagnostics.usedLearnedRouteFn, true);
  assert.equal(second.turn.compileResponse.diagnostics.routerIdentity, promotedAfterFirst.routerIdentity);
  assert.equal(
    second.turn.compileResponse.selectedContext.some((block) =>
      block.text.includes("Prefer the fresher learned route artifact after promotion when compiling freshness evidence.")
    ),
    true
  );
  assert.equal(second.learning.promoted, true);
  assert.equal(second.state.activePackVersion, 3);
  assert.equal(second.state.packLineage.length, 3);
});

test("classifyFeedbackKind normalizes common operator cues", () => {
  assert.equal(classifyFeedbackKind("No — do this instead."), "correction");
  assert.equal(classifyFeedbackKind("ship it, looks good"), "approval");
  assert.equal(classifyFeedbackKind("do not send this message"), "suppression");
});

test("async teacher live loop surfaces duplicate/no-op and freshness diagnostics", async () => {
  const loop = createAsyncTeacherLiveLoop({
    packLabel: "teacher-loop-runtime",
    workspace: {
      workspaceId: "workspace-teacher-loop",
      snapshotId: "workspace-teacher-loop@snapshot-1",
      capturedAt: "2026-03-07T18:00:00.000Z",
      rootDir: "/workspace/teacher-loop",
      revision: "teacher-loop-rev"
    },
    learnedRouting: false,
    maxQueuedExports: 1,
    staleAfterMs: 120_000,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });
  const exportOne = buildNormalizedRuntimeEventExport(
    {
      agentId: "runtime-teacher",
      sessionId: "session-teacher-1",
      channel: "whatsapp",
      userMessage: "teacher queue fresh diagnostics",
      sequenceStart: 801,
      compile: {
        createdAt: "2026-03-07T18:00:00.000Z"
      },
      feedback: [
        {
          createdAt: "2026-03-07T18:00:30.000Z",
          content: "Use the freshest correction before falling back to older teacher state."
        }
      ]
    },
    {
      ok: false,
      fallbackToStaticContext: true,
      hardRequirementViolated: false,
      activationRoot: "/missing",
      error: "No active pack pointer found",
      brainContext: ""
    }
  );

  const accepted = loop.enqueueNormalizedEventExport(exportOne, {
    observedAt: "2026-03-07T18:01:00.000Z"
  });
  const duplicate = loop.enqueueNormalizedEventExport(exportOne, {
    observedAt: "2026-03-07T18:01:15.000Z"
  });
  const snapshot = await loop.flush();

  assert.equal(accepted.accepted, true);
  assert.equal(duplicate.accepted, false);
  assert.equal(duplicate.reason, "duplicate_export");
  assert.equal(snapshot.queue.depth, 0);
  assert.equal(snapshot.teacher.artifactCount, 1);
  assert.equal(snapshot.teacher.latestFreshness, "fresh");
  assert.equal(snapshot.diagnostics.latestFreshness, "fresh");
  assert.match(snapshot.diagnostics.notes.join(";"), /teacher_freshness=fresh/);
  assert.match(snapshot.diagnostics.notes.join(";"), /teacher_noop=duplicate_export/);
  assert.match(duplicate.notes.join(";"), /teacher_noop=duplicate_export/);
});

test("async teacher live loop smoke proves teacher output reaches the learner candidate-pack path", async (t) => {
  const rootDir = mkdtemp("openclawbrain-openclaw-teacher-smoke-");
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const loop = createAsyncTeacherLiveLoop({
    packLabel: "teacher-smoke",
    workspace: {
      workspaceId: "workspace-teacher-smoke",
      snapshotId: "workspace-teacher-smoke@snapshot-1",
      capturedAt: "2026-03-07T19:00:00.000Z",
      rootDir: "/workspace/teacher-smoke",
      revision: "teacher-smoke-rev"
    },
    learnedRouting: true,
    staleAfterMs: 300_000,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });
  const exportOne = buildNormalizedRuntimeEventExport(
    {
      agentId: "runtime-teacher",
      sessionId: "session-teacher-2",
      channel: "whatsapp",
      userMessage: "teacher output must reach the learner path",
      sequenceStart: 901,
      compile: {
        createdAt: "2026-03-07T19:00:00.000Z"
      },
      delivery: {
        createdAt: "2026-03-07T19:00:45.000Z",
        messageId: "msg-teacher-2"
      },
      feedback: [
        {
          createdAt: "2026-03-07T19:00:30.000Z",
          content: "Prioritize this fresh supervision when building the next candidate pack."
        }
      ]
    },
    {
      ok: false,
      fallbackToStaticContext: true,
      hardRequirementViolated: false,
      activationRoot: "/missing",
      error: "No active pack pointer found",
      brainContext: ""
    }
  );

  loop.enqueueNormalizedEventExport(exportOne, {
    observedAt: "2026-03-07T19:01:00.000Z"
  });
  const snapshot = await loop.flush();

  assert.notEqual(snapshot.learner.lastMaterialization, null);
  const descriptor = materializeAlwaysOnLearningCandidatePack(rootDir, snapshot.learner.lastMaterialization!);

  assert.equal(snapshot.teacher.artifactCount, 1);
  assert.equal(snapshot.learner.lastMaterialization?.candidate.summary.routePolicy, "requires_learned_routing");
  assert.match(descriptor.graph.blocks.map((block) => block.text).join("\n"), /Teacher teaching/);
  assert.match(
    descriptor.graph.blocks.map((block) => block.text).join("\n"),
    /Prioritize this fresh supervision when building the next candidate pack/
  );
  assert.equal(descriptor.graph.blocks.some((block) => block.learning.role === "teacher_supervision"), true);
});
