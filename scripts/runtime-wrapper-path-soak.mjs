#!/usr/bin/env node

import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { describeCompileFallbackUsage } from "../packages/compiler/dist/src/index.js";
import { FIXTURE_FEEDBACK_EVENTS, FIXTURE_INTERACTION_EVENTS } from "../packages/contracts/dist/src/index.js";
import { describeNormalizedEventExportObservability } from "../packages/event-export/dist/src/index.js";
import {
  materializeAlwaysOnLearningCandidatePack,
  materializeCandidatePack
} from "../packages/learner/dist/src/index.js";
import {
  activatePack,
  describeActivationObservability,
  inspectActivationState,
  loadPack,
  promoteCandidatePack,
  stageCandidatePack
} from "../packages/pack-format/dist/src/index.js";
import {
  compileRuntimeContext,
  createAsyncTeacherLiveLoop,
  loadRuntimeEventExportBundle,
  runRuntimeTurn
} from "../packages/openclaw/dist/src/index.js";

function logStep(message) {
  console.log(`[runtime-wrapper-path:smoke] ${message}`);
}

function makeRootDir() {
  return path.join(tmpdir(), `openclawbrain-runtime-wrapper-path-${Date.now()}-${Math.random().toString(16).slice(2)}`);
}

function noteValue(notes, prefix) {
  const note = notes.find((entry) => entry.startsWith(prefix));
  return note === undefined ? null : note.slice(prefix.length);
}

function expectCompileSuccess(result) {
  assert.equal(result.ok, true);

  if (!result.ok) {
    throw new Error(`expected compile success, received failure: ${result.error}`);
  }

  return result;
}

function materializeSeedActivePack(activePackRoot, activationRoot) {
  const pack = materializeCandidatePack(activePackRoot, {
    packLabel: "runtime-wrapper-soak-seed",
    workspace: {
      workspaceId: "workspace-wrapper-soak",
      snapshotId: "workspace-wrapper-soak@snapshot-1",
      capturedAt: "2026-03-07T17:00:00.000Z",
      rootDir: "/workspace/openclawbrain",
      branch: "main",
      revision: "runtime-wrapper-soak-rev-1",
      labels: ["openclaw", "wrapper", "soak", "seed"]
    },
    eventRange: {
      start: 101,
      end: 104
    },
    eventExports: {
      interactionEvents: FIXTURE_INTERACTION_EVENTS,
      feedbackEvents: FIXTURE_FEEDBACK_EVENTS
    },
    learnedRouting: true,
    builtAt: "2026-03-07T17:05:00.000Z"
  });

  activatePack(activationRoot, activePackRoot, "2026-03-07T17:06:00.000Z");
  return pack;
}

function summarizeGraphEvolution(rootDir, graphChecksum) {
  const pack = loadPack(rootDir);
  const evolution = pack.graph.evolution ?? null;

  return {
    packId: pack.manifest.packId,
    graphChecksum,
    blockCount: pack.graph.blocks.length,
    strongestBlockId: evolution?.strongestBlockId ?? null,
    builtAt: evolution?.builtAt ?? pack.manifest.provenance.builtAt,
    hebbianApplied: evolution?.hebbianApplied ?? false,
    decayApplied: evolution?.decayApplied ?? false,
    structuralOps:
      evolution?.structuralOps ?? {
        split: 0,
        merge: 0,
        prune: 0,
        connect: 0
      },
    prunedBlockCount: evolution?.prunedBlockIds.length ?? 0
  };
}

export async function runRuntimeWrapperPathSoakScenario() {
  const rootDir = makeRootDir();
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  const exportRoot = path.join(rootDir, "runtime-export");
  const candidateRoot = path.join(rootDir, "candidate-pack");

  try {
    logStep("Materializing the seed active pack.");
    const seedPack = materializeSeedActivePack(activePackRoot, activationRoot);

    logStep("Compiling once through the real wrapper-facing path before learning.");
    const beforePromotionCompile = expectCompileSuccess(
      compileRuntimeContext({
        activationRoot,
        agentId: "runtime-wrapper-soak",
        message: "zebra nebula quartz",
        maxContextBlocks: 2,
        maxContextChars: 320,
        mode: "heuristic"
      })
    );
    const activeBefore = describeActivationObservability(activationRoot, "active", {
      requireActivationReady: true
    });
    const beforeNotes = beforePromotionCompile.compileResponse.diagnostics.notes;
    const beforeFallback = describeCompileFallbackUsage(beforePromotionCompile.compileResponse);

    assert.equal(beforePromotionCompile.activePackId, seedPack.manifest.packId);
    assert.equal(beforePromotionCompile.compileResponse.diagnostics.routerIdentity, activeBefore.learnedRouteFn.routerIdentity);
    assert.equal(noteValue(beforeNotes, "router_strategy="), activeBefore.learnedRouteFn.routeFnVersion);
    assert.equal(activeBefore.learnedRouteFn.routerChecksum !== null, true);
    assert.equal(noteValue(beforeNotes, "router_weights_checksum=") !== null, true);
    assert.equal(noteValue(beforeNotes, "router_freshness_checksum=") !== null, true);
    assert.equal(beforeFallback.priorityFallbackUsed, true);

    logStep("Running a real runtime turn and writing the exported bundle.");
    const feedbackContent =
      "Prefer the fresher learned route artifact after promotion when compiling wrapper-path evidence.";
    const turn = runRuntimeTurn(
      {
        agentId: "runtime-wrapper-soak",
        sessionId: "session-wrapper-soak",
        channel: "whatsapp",
        sourceStream: "openclaw/runtime/wrapper-whatsapp-live",
        userMessage: "Compile wrapper-path evidence before background promotion.",
        runtimeHints: ["wrapper-path", "promotion", "evidence"],
        sequenceStart: 901,
        compile: {
          createdAt: "2026-03-07T18:00:00.000Z"
        },
        delivery: {
          createdAt: "2026-03-07T18:03:00.000Z",
          messageId: "msg-wrapper-soak-1"
        },
        feedback: [
          {
            createdAt: "2026-03-07T18:02:00.000Z",
            content: feedbackContent
          }
        ],
        export: {
          rootDir: exportRoot,
          exportName: "runtime-wrapper-soak"
        }
      },
      {
        activationRoot
      }
    );

    assert.equal(turn.ok, true);
    assert.equal(turn.eventExport.ok, true);
    assert.equal(turn.eventExport.wroteBundle, true);
    if (!turn.ok || !turn.eventExport.ok || !turn.eventExport.wroteBundle) {
      throw new Error("runtime wrapper-path soak turn must compile and write an export bundle");
    }

    const bundle = loadRuntimeEventExportBundle(turn.eventExport.rootDir);
    const exportObservability = describeNormalizedEventExportObservability(bundle.normalizedEventExport);

    assert.equal(exportObservability.supervisionFreshnessBySource.length, 1);
    assert.equal(exportObservability.supervisionFreshnessBySource[0]?.sourceStream, "openclaw/runtime/wrapper-whatsapp-live");
    assert.equal(exportObservability.supervisionFreshnessBySource[0]?.humanLabelCount, 1);
    assert.equal(exportObservability.supervisionFreshnessBySource[0]?.selfLabelCount, 1);

    logStep("Feeding the real export through the async background teacher loop.");
    const teacherLoop = createAsyncTeacherLiveLoop({
      packLabel: "runtime-wrapper-soak",
      workspace: {
        workspaceId: "workspace-wrapper-soak",
        snapshotId: "workspace-wrapper-soak@snapshot-2",
        capturedAt: "2026-03-07T18:03:00.000Z",
        rootDir: "/workspace/openclawbrain",
        branch: "main",
        revision: "runtime-wrapper-soak-rev-2",
        labels: ["openclaw", "wrapper", "soak", "background-learning"]
      },
      learnedRouting: true,
      builtAt: "2026-03-07T18:05:00.000Z",
      liveSliceSize: 2,
      backfillSliceSize: 2,
      staleAfterMs: 300_000,
      maxQueuedExports: 2
    });
    const enqueue = teacherLoop.enqueueNormalizedEventExport(bundle.normalizedEventExport, {
      observedAt: "2026-03-07T18:04:00.000Z"
    });
    assert.equal(enqueue.accepted, true);

    const snapshot = await teacherLoop.flush();
    const materialization = snapshot.learner.lastMaterialization;
    assert.notEqual(materialization, null);
    if (materialization === null) {
      throw new Error("background teacher loop did not materialize a candidate pack");
    }

    const candidateDescriptor = materializeAlwaysOnLearningCandidatePack(candidateRoot, materialization);
    stageCandidatePack(activationRoot, candidateRoot, "2026-03-07T18:06:00.000Z");

    const stagedInspection = inspectActivationState(activationRoot, "2026-03-07T18:06:00.000Z");
    const stagedActive = describeActivationObservability(activationRoot, "active", {
      requireActivationReady: true,
      updatedAt: "2026-03-07T18:06:00.000Z"
    });
    const stagedCandidate = describeActivationObservability(activationRoot, "candidate", {
      requireActivationReady: true,
      updatedAt: "2026-03-07T18:06:00.000Z"
    });
    const stagedGraph = summarizeGraphEvolution(candidateRoot, stagedCandidate.graphDynamics.graphChecksum);

    assert.equal(stagedInspection.candidate?.packId, candidateDescriptor.manifest.packId);
    assert.equal(stagedInspection.promotion.allowed, true);
    assert.equal(stagedActive.promotionFreshness.activeBehindPromotionReadyCandidate, true);
    assert.equal(stagedCandidate.learnedRouteFn.routerIdentity, candidateDescriptor.router?.routerIdentity ?? null);
    assert.equal(stagedCandidate.learnedRouteFn.routeFnVersion, candidateDescriptor.router?.strategy ?? null);
    assert.equal(stagedGraph.graphChecksum !== null, true);
    assert.equal(stagedGraph.strongestBlockId !== null, true);

    logStep("Promoting the learned candidate and compiling again through the wrapper.");
    promoteCandidatePack(activationRoot, "2026-03-07T18:07:00.000Z");

    const afterPromotionCompile = expectCompileSuccess(
      compileRuntimeContext({
        activationRoot,
        agentId: "runtime-wrapper-soak",
        message: "Compile the fresher learned route artifact after promotion.",
        maxContextBlocks: 3,
        maxContextChars: 480,
        mode: "heuristic",
        runtimeHints: ["fresher", "learned", "route", "artifact", "promotion", "wrapper-path", "evidence"]
      })
    );
    const activeAfter = describeActivationObservability(activationRoot, "active", {
      requireActivationReady: true,
      updatedAt: "2026-03-07T18:07:00.000Z"
    });
    const afterNotes = afterPromotionCompile.compileResponse.diagnostics.notes;
    const afterFallback = describeCompileFallbackUsage(afterPromotionCompile.compileResponse);

    assert.notEqual(afterPromotionCompile.activePackId, seedPack.manifest.packId);
    assert.equal(afterPromotionCompile.activePackId, candidateDescriptor.manifest.packId);
    assert.equal(afterPromotionCompile.compileResponse.diagnostics.routerIdentity, candidateDescriptor.router?.routerIdentity ?? null);
    assert.equal(noteValue(afterNotes, "router_strategy="), activeAfter.learnedRouteFn.routeFnVersion);
    assert.equal(activeAfter.learnedRouteFn.routerChecksum, candidateDescriptor.manifest.payloadChecksums.router);
    assert.equal(noteValue(afterNotes, "router_weights_checksum=") !== null, true);
    assert.equal(noteValue(afterNotes, "router_freshness_checksum=") !== null, true);
    assert.equal(
      afterPromotionCompile.compileResponse.selectedContext.some((block) => block.text.includes(feedbackContent)),
      true
    );

    const report = {
      initialCompile: {
        packId: beforePromotionCompile.activePackId,
        routerIdentity: beforePromotionCompile.compileResponse.diagnostics.routerIdentity,
        routeFnVersion: activeBefore.learnedRouteFn.routeFnVersion,
        routerChecksum: activeBefore.learnedRouteFn.routerChecksum,
        weightsChecksum: noteValue(beforeNotes, "router_weights_checksum="),
        freshnessChecksum: noteValue(beforeNotes, "router_freshness_checksum="),
        fallbackUsage: beforeFallback
      },
      turn: {
        packId: turn.activePackId,
        exportDigest: bundle.normalizedEventExport.provenance.exportDigest,
        supervisionCountsBySource: exportObservability.supervisionFreshnessBySource,
        teacherFreshness: exportObservability.teacherFreshness
      },
      stagedFreshness: {
        active: stagedInspection.active,
        candidate: stagedInspection.candidate,
        promotion: stagedInspection.promotion,
        promotionFreshness: stagedActive.promotionFreshness
      },
      graphEvolution: stagedGraph,
      laterCompile: {
        packId: afterPromotionCompile.activePackId,
        routerIdentity: afterPromotionCompile.compileResponse.diagnostics.routerIdentity,
        routeFnVersion: activeAfter.learnedRouteFn.routeFnVersion,
        routerChecksum: activeAfter.learnedRouteFn.routerChecksum,
        weightsChecksum: noteValue(afterNotes, "router_weights_checksum="),
        freshnessChecksum: noteValue(afterNotes, "router_freshness_checksum="),
        fallbackUsage: afterFallback
      }
    };

    logStep("Runtime wrapper-path soak passed.");
    console.log(JSON.stringify(report, null, 2));
    return report;
  } finally {
    rmSync(rootDir, { recursive: true, force: true });
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await runRuntimeWrapperPathSoakScenario();
  } catch (error) {
    console.error("[runtime-wrapper-path:smoke] failed");
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
