#!/usr/bin/env node

import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { activatePack, inspectActivationState, promoteCandidatePack, stageCandidatePack } from "../packages/activation/dist/src/index.js";
import { compileRuntimeFromActivation } from "../packages/compiler/dist/src/index.js";
import { CONTRACT_IDS } from "../packages/contracts/dist/src/index.js";
import { buildNormalizedEventExport } from "../packages/event-export/dist/src/index.js";
import { createFeedbackEvent, createInteractionEvent, sortNormalizedEvents } from "../packages/events/dist/src/index.js";
import { materializeCandidatePackFromNormalizedEventExport } from "../packages/learner/dist/src/index.js";

function logStep(message) {
  console.log(`[lifecycle:smoke] ${message}`);
}

function buildLifecycleExport({
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
  assert.deepEqual(
    sorted.map((event) => event.eventId),
    [interaction.eventId, feedback.eventId],
    "normalized events should sort deterministically"
  );

  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents: [interaction],
    feedbackEvents: [feedback]
  });

  assert.equal(normalizedEventExport.range.start, sequenceStart);
  assert.equal(normalizedEventExport.range.end, sequenceStart + 1);
  assert.equal(normalizedEventExport.range.count, 2);
  assert.equal(normalizedEventExport.provenance.interactionCount, 1);
  assert.equal(normalizedEventExport.provenance.feedbackCount, 1);

  return normalizedEventExport;
}

function main() {
  const rootDir = mkdtempSync(path.join(tmpdir(), "openclawbrain-lifecycle-smoke-"));

  try {
    const activationRoot = path.join(rootDir, "activation");
    const activePackRoot = path.join(rootDir, "packs", "active");
    const candidatePackRoot = path.join(rootDir, "packs", "candidate");

    logStep("Building normalized events and deterministic event exports.");

    const activeExport = buildLifecycleExport({
      agentId: "agent-lifecycle-smoke",
      sessionId: "session-lifecycle-active",
      channel: "cli",
      sequenceStart: 41,
      createdAt: "2026-03-06T04:41:00.000Z",
      streamSuffix: "lifecycle-active",
      interactionKind: "memory_compiled",
      interactionMessageId: "msg-lifecycle-active",
      interactionPackId: "pack-seed-active",
      feedbackKind: "approval",
      feedbackContent: "Keep the seed active pack stable while the Phase-2 lifecycle smoke candidate is prepared."
    });

    const candidateExport = buildLifecycleExport({
      agentId: "agent-lifecycle-smoke",
      sessionId: "session-lifecycle-candidate",
      channel: "cli",
      sequenceStart: 51,
      createdAt: "2026-03-06T04:51:00.000Z",
      streamSuffix: "lifecycle-candidate",
      interactionKind: "operator_override",
      interactionMessageId: "msg-lifecycle-candidate",
      interactionPackId: "pack-seed-candidate",
      feedbackKind: "teaching",
      feedbackContent:
        "Promote the lifecycle smoke candidate after activation staging so compiler selection surfaces activation promotion evidence."
    });

    logStep("Materializing active and candidate learner packs from exported events.");

    const activePack = materializeCandidatePackFromNormalizedEventExport(activePackRoot, {
      packLabel: "phase-2-active",
      workspace: {
        workspaceId: "workspace-phase-2",
        snapshotId: "workspace-phase-2@snapshot-active",
        capturedAt: "2026-03-06T04:45:00.000Z",
        rootDir: "/workspace/phase-2",
        branch: "codex/20260306/ts-public-converge",
        revision: "phase-2-active-rev",
        labels: ["phase-2", "lifecycle", "active"]
      },
      normalizedEventExport: activeExport,
      learnedRouting: false,
      builtAt: "2026-03-06T04:46:00.000Z",
      offlineArtifacts: ["activation-pointer-preview"],
      structuralOps: {
        connect: 1
      }
    });

    const candidatePack = materializeCandidatePackFromNormalizedEventExport(candidatePackRoot, {
      packLabel: "phase-2-candidate",
      workspace: {
        workspaceId: "workspace-phase-2",
        snapshotId: "workspace-phase-2@snapshot-candidate",
        capturedAt: "2026-03-06T04:55:00.000Z",
        rootDir: "/workspace/phase-2",
        branch: "codex/20260306/ts-public-converge",
        revision: "phase-2-candidate-rev",
        labels: ["phase-2", "lifecycle", "candidate"]
      },
      normalizedEventExport: candidateExport,
      learnedRouting: true,
      builtAt: "2026-03-06T04:56:00.000Z",
      offlineArtifacts: ["activation-pointer-preview", "compiler-smoke"],
      structuralOps: {
        connect: 2,
        split: 1
      }
    });

    assert.notEqual(activePack.manifest.packId, candidatePack.manifest.packId, "smoke lane should materialize distinct packs");
    assert.equal(candidatePack.manifest.provenance.eventExports?.exportDigest, candidateExport.provenance.exportDigest);

    logStep("Staging and promoting the candidate into the active activation slot.");

    activatePack(activationRoot, activePackRoot, "2026-03-06T05:00:00.000Z");
    stageCandidatePack(activationRoot, candidatePackRoot, "2026-03-06T05:05:00.000Z");

    const staged = inspectActivationState(activationRoot, "2026-03-06T05:06:00.000Z");
    assert.equal(staged.active?.packId, activePack.manifest.packId);
    assert.equal(staged.candidate?.packId, candidatePack.manifest.packId);
    assert.equal(staged.candidate?.activationReady, true);
    assert.equal(staged.promotion.allowed, true);

    promoteCandidatePack(activationRoot, "2026-03-06T05:10:00.000Z");

    logStep("Compiling runtime context from the promoted active activation slot.");

    const { target: activeTarget, response: compileResponse } = compileRuntimeFromActivation(
      activationRoot,
      {
        contract: CONTRACT_IDS.runtimeCompile,
        agentId: "agent-lifecycle-smoke",
        userMessage: "Compile the activation promotion evidence for the Phase-2 lifecycle smoke lane.",
        maxContextBlocks: 2,
        maxContextChars: 480,
        modeRequested: "heuristic",
        runtimeHints: ["activation", "promotion", "compiler", "evidence"],
        compactionMode: "native"
      },
      {
        expectedTarget: {
          packId: candidatePack.manifest.packId,
          routePolicy: candidatePack.manifest.routePolicy,
          routerIdentity: candidatePack.router?.routerIdentity ?? null,
          workspaceSnapshot: candidatePack.manifest.provenance.workspaceSnapshot,
          workspaceRevision: candidatePack.manifest.provenance.workspace.revision,
          eventRange: {
            start: candidatePack.manifest.provenance.eventRange.start,
            end: candidatePack.manifest.provenance.eventRange.end,
            count: candidatePack.manifest.provenance.eventRange.count
          },
          eventExportDigest: candidatePack.manifest.provenance.eventExports?.exportDigest ?? null,
          builtAt: candidatePack.manifest.provenance.builtAt
        }
      }
    );

    assert.equal(activeTarget.packId, candidatePack.manifest.packId);
    assert.equal(activeTarget.workspaceSnapshot, "workspace-phase-2@snapshot-candidate");
    assert.equal(activeTarget.eventExportDigest, candidateExport.provenance.exportDigest);
    assert.equal(compileResponse.packId, candidatePack.manifest.packId);
    assert.equal(compileResponse.diagnostics.modeEffective, "learned");
    assert.equal(compileResponse.diagnostics.usedLearnedRouteFn, true);
    assert.equal(compileResponse.diagnostics.routerIdentity, candidatePack.router?.routerIdentity ?? null);
    assert.equal(compileResponse.selectedContext.length > 0, true);
    assert.equal(
      compileResponse.selectedContext.some((block) => /activation promotion evidence/i.test(block.text)),
      true,
      "compile response should include the promoted candidate feedback context"
    );

    logStep("Lifecycle smoke passed.");
    console.log(
      JSON.stringify(
        {
          activePackId: activePack.manifest.packId,
          candidatePackId: candidatePack.manifest.packId,
          promotedPackId: activeTarget.packId,
          selectedContextIds: compileResponse.selectedContext.map((block) => block.id),
          selectionDigest: compileResponse.diagnostics.selectionDigest
        },
        null,
        2
      )
    );
  } finally {
    rmSync(rootDir, { recursive: true, force: true });
  }
}

try {
  main();
} catch (error) {
  console.error("[lifecycle:smoke] failed");
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}
