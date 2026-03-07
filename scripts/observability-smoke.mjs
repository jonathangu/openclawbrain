#!/usr/bin/env node

import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import {
  activatePack,
  describeActivationObservability,
  describeActivationTarget,
  inspectActivationState,
  promoteCandidatePack,
  stageCandidatePack
} from "../packages/activation/dist/src/index.js";
import { compileRuntimeFromActivation, describeCompileFallbackUsage } from "../packages/compiler/dist/src/index.js";
import { CONTRACT_IDS } from "../packages/contracts/dist/src/index.js";
import {
  buildNormalizedEventExport,
  describeNormalizedEventExportObservability
} from "../packages/event-export/dist/src/index.js";
import { createFeedbackEvent, createInteractionEvent, sortNormalizedEvents } from "../packages/events/dist/src/index.js";
import { materializeCandidatePackFromNormalizedEventExport } from "../packages/learner/dist/src/index.js";
import { compileRuntimeContext } from "../packages/openclaw/dist/src/index.js";

function buildExport({
  agentId,
  sessionId,
  channel,
  sequenceStart,
  createdAt,
  streamSuffix,
  interactionKind,
  interactionMessageId,
  interactionPackId,
  feedbackKind,
  feedbackContent
}) {
  const interaction = createInteractionEvent({
    eventId: `${sessionId}:interaction:${sequenceStart}`,
    agentId,
    sessionId,
    channel,
    sequence: sequenceStart,
    kind: interactionKind,
    createdAt,
    source: {
      runtimeOwner: "openclaw",
      stream: `openclaw/runtime/${streamSuffix}`
    },
    messageId: interactionMessageId,
    ...(interactionPackId !== undefined ? { packId: interactionPackId } : {})
  });

  const feedback = createFeedbackEvent({
    eventId: `${sessionId}:feedback:${sequenceStart + 1}`,
    agentId,
    sessionId,
    channel,
    sequence: sequenceStart + 1,
    kind: feedbackKind,
    createdAt: new Date(Date.parse(createdAt) + 60_000).toISOString(),
    source: {
      runtimeOwner: "openclaw",
      stream: `openclaw/runtime/${streamSuffix}`
    },
    content: feedbackContent,
    relatedInteractionId: interaction.eventId
  });

  const sorted = sortNormalizedEvents([interaction, feedback]);
  assert.deepEqual(sorted.map((event) => event.eventId), [interaction.eventId, feedback.eventId]);

  return buildNormalizedEventExport({
    interactionEvents: [interaction],
    feedbackEvents: [feedback]
  });
}

export function runObservabilityScenario(options = {}) {
  const logPrefix = options.logPrefix ?? "observability:smoke";
  const emitSteps = options.emitSteps ?? true;
  const logStep = (message) => {
    if (emitSteps) {
      console.log(`[${logPrefix}] ${message}`);
    }
  };
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-observability-smoke-"));

  try {
    const activationRoot = path.join(rootDir, "activation");
    const activePackRoot = path.join(rootDir, "packs", "active");
    const candidatePackRoot = path.join(rootDir, "packs", "candidate");

    logStep("Building fast-boot and promoted candidate exports.");

    const activeExport = buildExport({
      agentId: "agent-observability-smoke",
      sessionId: "session-observability-active",
      channel: "cli",
      sequenceStart: 61,
      createdAt: "2026-03-06T06:01:00.000Z",
      streamSuffix: "observability-active",
      interactionKind: "memory_compiled",
      interactionMessageId: "msg-observability-active",
      interactionPackId: "pack-observability-active",
      feedbackKind: "approval",
      feedbackContent: "Keep the fast-boot active pack healthy while live events continue to stream."
    });

    const candidateExport = buildExport({
      agentId: "agent-observability-smoke",
      sessionId: "session-observability-candidate",
      channel: "cli",
      sequenceStart: 71,
      createdAt: "2026-03-06T06:11:00.000Z",
      streamSuffix: "observability-candidate",
      interactionKind: "operator_override",
      interactionMessageId: "msg-observability-candidate",
      interactionPackId: "pack-observability-candidate",
      feedbackKind: "teaching",
      feedbackContent: "Promote the fresher candidate once activation inspection proves it is healthy."
    });

    const activePack = materializeCandidatePackFromNormalizedEventExport(activePackRoot, {
      packLabel: "observability-active",
      workspace: {
        workspaceId: "workspace-observability",
        snapshotId: "workspace-observability@snapshot-active",
        capturedAt: "2026-03-06T06:05:00.000Z",
        rootDir: "/workspace/observability",
        branch: "main",
        revision: "observability-active-rev",
        labels: ["observability", "active", "fast-boot"]
      },
      normalizedEventExport: activeExport,
      learnedRouting: false,
      builtAt: "2026-03-06T06:06:00.000Z",
      offlineArtifacts: ["fast-boot"],
      structuralOps: {
        connect: 1
      }
    });

    const candidatePack = materializeCandidatePackFromNormalizedEventExport(candidatePackRoot, {
      packLabel: "observability-candidate",
      workspace: {
        workspaceId: "workspace-observability",
        snapshotId: "workspace-observability@snapshot-candidate",
        capturedAt: "2026-03-06T06:15:00.000Z",
        rootDir: "/workspace/observability",
        branch: "main",
        revision: "observability-candidate-rev",
        labels: ["observability", "candidate", "live-events-first"]
      },
      normalizedEventExport: candidateExport,
      learnedRouting: true,
      builtAt: "2026-03-06T06:16:00.000Z",
      offlineArtifacts: ["fast-boot", "live-events-first"],
      structuralOps: {
        connect: 2,
        split: 1
      }
    });

    logStep("Inspecting activation health before promotion.");

    activatePack(activationRoot, activePackRoot, "2026-03-06T06:20:00.000Z");
    stageCandidatePack(activationRoot, candidatePackRoot, "2026-03-06T06:25:00.000Z");

    const candidateEventObservability = describeNormalizedEventExportObservability(candidateExport);
    assert.equal(candidateEventObservability.supervisionFreshnessBySource.length, 1);
    assert.deepEqual(candidateEventObservability.supervisionFreshnessBySource[0], {
      sourceStream: "openclaw/runtime/observability-candidate",
      eventCount: 2,
      interactionCount: 1,
      feedbackCount: 1,
      humanLabelCount: 2,
      selfLabelCount: 0,
      freshestEventId: "session-observability-candidate:feedback:72",
      freshestSequence: 72,
      freshestCreatedAt: "2026-03-06T06:12:00.000Z",
      freshestKind: "teaching"
    });
    assert.deepEqual(candidateEventObservability.teacherFreshness, {
      freshestEventId: "session-observability-candidate:feedback:72",
      freshestSequence: 72,
      freshestCreatedAt: "2026-03-06T06:12:00.000Z",
      freshestKind: "teaching",
      sourceStream: "openclaw/runtime/observability-candidate",
      humanLabelCount: 2,
      sources: ["openclaw/runtime/observability-candidate"]
    });

    const stagedInspection = inspectActivationState(activationRoot, "2026-03-06T06:26:00.000Z");
    assert.equal(stagedInspection.active?.activationReady, true);
    assert.equal(stagedInspection.candidate?.activationReady, true);
    assert.deepEqual(stagedInspection.active?.findings ?? [], []);
    assert.deepEqual(stagedInspection.candidate?.findings ?? [], []);
    assert.equal(stagedInspection.promotion.allowed, true);
    assert.deepEqual(stagedInspection.promotion.findings, []);

    const stagedObservability = describeActivationObservability(activationRoot, "active", {
      updatedAt: "2026-03-06T06:26:00.000Z"
    });
    assert.equal(stagedObservability.learnedRouteFn.required, false);
    assert.equal(stagedObservability.learnedRouteFn.available, false);
    assert.equal(stagedObservability.promotionFreshness.activeBehindPromotionReadyCandidate, true);
    assert.deepEqual(stagedObservability.promotionFreshness.candidateAheadBy, {
      builtAt: true,
      eventRangeEnd: true,
      eventRangeCount: false,
      workspaceSnapshot: true,
      workspaceRevision: true,
      eventExportDigest: true
    });

    logStep("Promoting the fresher candidate and proving freshness.");

    promoteCandidatePack(activationRoot, "2026-03-06T06:30:00.000Z");

    const promotedInspection = inspectActivationState(activationRoot, "2026-03-06T06:31:00.000Z");
    assert.equal(promotedInspection.active?.packId, candidatePack.manifest.packId);
    assert.equal(promotedInspection.previous?.packId, activePack.manifest.packId);
    assert.equal(promotedInspection.rollback.allowed, true);
    assert.deepEqual(promotedInspection.rollback.findings, []);

    const promotedObservability = describeActivationObservability(activationRoot, "active", {
      requireActivationReady: true,
      updatedAt: "2026-03-06T06:31:00.000Z"
    });
    assert.equal(promotedObservability.learnedRouteFn.packId, candidatePack.manifest.packId);
    assert.equal(promotedObservability.learnedRouteFn.required, true);
    assert.equal(promotedObservability.learnedRouteFn.available, true);
    assert.equal(promotedObservability.learnedRouteFn.routerIdentity, candidatePack.router?.routerIdentity ?? null);
    assert.equal(promotedObservability.learnedRouteFn.routeFnVersion, "learned_route_fn_v1");
    assert.equal(promotedObservability.learnedRouteFn.routerChecksum, candidatePack.manifest.payloadChecksums.router);
    assert.equal(promotedObservability.learnedRouteFn.routerTrainedAt, candidatePack.router?.trainedAt ?? null);
    assert.equal(promotedObservability.graphDynamics.packId, candidatePack.manifest.packId);
    assert.equal(promotedObservability.graphDynamics.graphChecksum, candidatePack.manifest.payloadChecksums.graph);
    assert.equal(promotedObservability.graphDynamics.builtAt, candidatePack.manifest.provenance.builtAt);
    assert.deepEqual(promotedObservability.graphDynamics.structuralOps, {
      split: 1,
      merge: 0,
      prune: 0,
      connect: 2
    });

    const activeTarget = describeActivationTarget(activationRoot, "active", { requireActivationReady: true });
    assert.notEqual(activeTarget, null);
    assert.equal(activeTarget?.packId, candidatePack.manifest.packId);
    assert.equal(activeTarget?.workspaceSnapshot, "workspace-observability@snapshot-candidate");
    assert.equal(activeTarget?.workspaceRevision, "observability-candidate-rev");
    assert.equal(activeTarget?.eventRange.end, candidateExport.range.end);
    assert.equal(activeTarget?.eventExportDigest, candidateExport.provenance.exportDigest);
    assert.equal(activeTarget?.builtAt, candidatePack.manifest.provenance.builtAt);

    logStep("Compiling through the promoted slot and proving fallback diagnostics.");

    const compile = compileRuntimeFromActivation(
      activationRoot,
      {
        contract: CONTRACT_IDS.runtimeCompile,
        agentId: "agent-observability-smoke",
        userMessage: "zebra nebula quartz",
        maxContextBlocks: 2,
        maxContextChars: 320,
        modeRequested: "heuristic",
        compactionMode: "native"
      },
      {
        expectedTarget: {
          packId: candidatePack.manifest.packId,
          routePolicy: candidatePack.manifest.routePolicy,
          routerIdentity: candidatePack.router?.routerIdentity ?? null,
          workspaceSnapshot: "workspace-observability@snapshot-candidate",
          workspaceRevision: "observability-candidate-rev",
          eventRange: {
            start: candidateExport.range.start,
            end: candidateExport.range.end,
            count: candidateExport.range.count
          },
          eventExportDigest: candidateExport.provenance.exportDigest,
          builtAt: candidatePack.manifest.provenance.builtAt
        }
      }
    );

    assert.equal(compile.target.packId, candidatePack.manifest.packId);
    assert.equal(compile.response.diagnostics.modeRequested, "heuristic");
    assert.equal(compile.response.diagnostics.modeEffective, "learned");
    assert.equal(compile.response.diagnostics.usedLearnedRouteFn, true);
    assert.equal(compile.response.diagnostics.selectionStrategy, "pack_route_fn_selection_v1");
    assert.equal(typeof compile.response.diagnostics.selectionDigest, "string");
    assert.equal(compile.response.diagnostics.selectionDigest.length > 0, true);
    assert.match(compile.response.diagnostics.notes.join(";"), /activation_slot=active/);
    assert.match(compile.response.diagnostics.notes.join(";"), new RegExp(`target_pack_id=${candidatePack.manifest.packId}`));
    assert.match(compile.response.diagnostics.notes.join(";"), /target_route_policy=requires_learned_routing/);
    assert.match(
      compile.response.diagnostics.notes.join(";"),
      /target_workspace_snapshot=workspace-observability@snapshot-candidate/
    );
    assert.match(
      compile.response.diagnostics.notes.join(";"),
      /target_workspace_revision=observability-candidate-rev/
    );
    assert.match(
      compile.response.diagnostics.notes.join(";"),
      new RegExp(`target_event_range=${candidateExport.range.start}-${candidateExport.range.end}#${candidateExport.range.count}`)
    );
    assert.match(
      compile.response.diagnostics.notes.join(";"),
      new RegExp(`target_event_export_digest=${candidateExport.provenance.exportDigest}`)
    );
    assert.match(
      compile.response.diagnostics.notes.join(";"),
      new RegExp(`target_built_at=${candidatePack.manifest.provenance.builtAt}`)
    );
    assert.match(
      compile.response.diagnostics.notes.join(";"),
      new RegExp(`target_router_identity=${candidatePack.router?.routerIdentity}`)
    );
    assert.match(compile.response.diagnostics.notes.join(";"), /selection_mode=priority_fallback/);

    const fallbackUsage = describeCompileFallbackUsage(compile.response);
    assert.deepEqual(fallbackUsage, {
      packId: candidatePack.manifest.packId,
      modeRequested: "heuristic",
      modeEffective: "learned",
      usedLearnedRouteFn: true,
      routerIdentity: candidatePack.router?.routerIdentity ?? null,
      selectionDigest: compile.response.diagnostics.selectionDigest,
      selectionMode: "priority_fallback",
      selectionTiers: "priority_fallback_only",
      priorityFallbackUsed: true,
      notes: ["selection_mode=priority_fallback", "selection_tiers=priority_fallback_only"]
    });

    logStep("Proving the same diagnostics and hard failure through the OpenClaw serve path.");

    const served = compileRuntimeContext({
      activationRoot,
      agentId: "agent-observability-smoke",
      message: "zebra nebula quartz",
      maxContextBlocks: 2,
      mode: "heuristic"
    });

    assert.equal(served.ok, true);
    assert.equal(served.compileResponse.diagnostics.selectionDigest, compile.response.diagnostics.selectionDigest);
    assert.match(served.compileResponse.diagnostics.notes.join(";"), /activation_slot=active/);
    assert.match(served.compileResponse.diagnostics.notes.join(";"), new RegExp(`target_pack_id=${candidatePack.manifest.packId}`));
    assert.match(served.compileResponse.diagnostics.notes.join(";"), /target_route_policy=requires_learned_routing/);
    assert.match(
      served.compileResponse.diagnostics.notes.join(";"),
      new RegExp(`target_router_identity=${candidatePack.router?.routerIdentity}`)
    );

    rmSync(path.join(candidatePackRoot, "router", "model.json"), { force: true });

    const hardFailure = compileRuntimeContext({
      activationRoot,
      agentId: "agent-observability-smoke",
      message: "zebra nebula quartz",
      maxContextBlocks: 2,
      mode: "heuristic"
    });

    assert.equal(hardFailure.ok, false);
    assert.equal(hardFailure.fallbackToStaticContext, false);
    assert.equal(hardFailure.hardRequirementViolated, true);
    assert.match(hardFailure.error, /Learned-routing hotpath hard requirement violated/);
    assert.match(hardFailure.error, /router payload not found/);

    const report = {
      supervisionFreshnessBySource: candidateEventObservability.supervisionFreshnessBySource,
      teacherFreshness: candidateEventObservability.teacherFreshness,
      learnedRouteFnFreshness: promotedObservability.learnedRouteFn,
      graphDynamicsFreshness: promotedObservability.graphDynamics,
      promotionFreshness: stagedObservability.promotionFreshness,
      fallbackUsage,
      servePath: {
        selectionDigest: served.compileResponse.diagnostics.selectionDigest,
        notes: served.compileResponse.diagnostics.notes,
        hardFailure: {
          fallbackToStaticContext: hardFailure.fallbackToStaticContext,
          hardRequirementViolated: hardFailure.hardRequirementViolated,
          error: hardFailure.error
        }
      }
    };

    logStep("Observability smoke passed.");
    console.log(JSON.stringify(report, null, 2));
    return report;
  } finally {
    rmSync(rootDir, { recursive: true, force: true });
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    runObservabilityScenario();
  } catch (error) {
    console.error("[observability:smoke] failed");
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
}
