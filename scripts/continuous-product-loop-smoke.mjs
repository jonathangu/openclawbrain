import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { materializeCandidatePack } from "../packages/learner/dist/src/index.js";
import { activatePack } from "../packages/activation/dist/src/index.js";
import { runContinuousProductLoopTurn } from "../packages/openclaw/dist/src/index.js";

function logStep(message) {
  console.log(`[continuous-product-loop:smoke] ${message}`);
}

function makeRootDir() {
  return path.join(tmpdir(), `openclawbrain-continuous-product-loop-${Date.now()}-${Math.random().toString(16).slice(2)}`);
}

function materializeSeedActivePack(activePackRoot, activationRoot) {
  const pack = materializeCandidatePack(activePackRoot, {
    packLabel: "continuous-loop-seed",
    workspace: {
      workspaceId: "workspace-continuous-loop",
      snapshotId: "workspace-continuous-loop@snapshot-seed",
      capturedAt: "2026-03-07T18:00:00.000Z",
      rootDir: "/workspace/openclawbrain",
      branch: "main",
      revision: "continuous-loop-seed-rev",
      labels: ["openclaw", "runtime", "seed"]
    },
    eventRange: {
      start: 101,
      end: 102
    },
    learnedRouting: true,
    eventExports: {
      interactionEvents: [
        {
          contract: "interaction_events.v1",
          eventId: "evt-seed-int-101",
          agentId: "agent-seed",
          sessionId: "session-seed",
          channel: "whatsapp",
          sequence: 101,
          kind: "memory_compiled",
          createdAt: "2026-03-07T17:50:00.000Z",
          source: {
            runtimeOwner: "openclaw",
            stream: "openclaw/runtime/whatsapp"
          },
          packId: "pack-seed-route"
        }
      ],
      feedbackEvents: [
        {
          contract: "feedback_events.v1",
          eventId: "evt-seed-feed-102",
          agentId: "agent-seed",
          sessionId: "session-seed",
          channel: "whatsapp",
          sequence: 102,
          kind: "approval",
          createdAt: "2026-03-07T17:51:00.000Z",
          source: {
            runtimeOwner: "openclaw",
            stream: "openclaw/runtime/whatsapp"
          },
          content: "Seed pack is active for the post-attach loop."
        }
      ]
    },
    builtAt: "2026-03-07T17:52:00.000Z"
  });

  activatePack(activationRoot, activePackRoot, "2026-03-07T17:53:00.000Z");
  return pack;
}

function main() {
  const rootDir = makeRootDir();
  const activePackRoot = path.join(rootDir, "active-pack");
  const activationRoot = path.join(rootDir, "activation");
  const loopRoot = path.join(rootDir, "product-loop");

  try {
    logStep("Materializing the seed active pack.");
    const seedPack = materializeSeedActivePack(activePackRoot, activationRoot);

    logStep("Running turn 1 through event export, supervision, learning, staging, and promotion.");
    const first = runContinuousProductLoopTurn({
      activationRoot,
      loopRoot,
      packLabel: "continuous-product-loop",
      workspace: {
        workspaceId: "workspace-continuous-loop",
        snapshotId: "workspace-continuous-loop@snapshot-1",
        capturedAt: "2026-03-07T18:00:30.000Z",
        rootDir: "/workspace/openclawbrain",
        branch: "main",
        revision: "continuous-loop-rev-1",
        labels: ["openclaw", "runtime", "turn-1"]
      },
      turn: {
        agentId: "continuous-loop-smoke",
        sessionId: "session-continuous-loop",
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
    assert.equal(first.compileActivePackId, seedPack.manifest.packId);
    assert.equal(first.turn.eventExport.ok, true);
    assert.equal(first.learning.promoted, true);
    assert.equal(first.state.activePackVersion, 2);

    const promotedAfterFirst = first.state.currentActivePack;
    assert.notEqual(promotedAfterFirst, null);

    logStep("Running turn 2 and proving compile now uses the fresher learned routing artifact.");
    const second = runContinuousProductLoopTurn({
      activationRoot,
      loopRoot,
      packLabel: "continuous-product-loop",
      workspace: {
        workspaceId: "workspace-continuous-loop",
        snapshotId: "workspace-continuous-loop@snapshot-2",
        capturedAt: "2026-03-07T18:10:30.000Z",
        rootDir: "/workspace/openclawbrain",
        branch: "main",
        revision: "continuous-loop-rev-2",
        labels: ["openclaw", "runtime", "turn-2"]
      },
      state: first.state,
      turn: {
        agentId: "continuous-loop-smoke",
        sessionId: "session-continuous-loop",
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
      throw new Error("turn 2 compile should succeed");
    }

    assert.equal(second.turn.compileResponse.diagnostics.routerIdentity, promotedAfterFirst.routerIdentity);
    assert.equal(
      second.turn.compileResponse.selectedContext.some((block) =>
        block.text.includes("Prefer the fresher learned route artifact after promotion when compiling freshness evidence.")
      ),
      true
    );
    assert.equal(second.state.activePackVersion, 3);

    logStep("Continuous post-attach product loop smoke passed.");
    console.log(
      JSON.stringify(
        {
          seedPackId: seedPack.manifest.packId,
          firstTurn: {
            compileActiveVersion: first.compileActiveVersion,
            promotedPackId: first.state.currentActivePack?.packId ?? null,
            promotedRouterIdentity: first.state.currentActivePack?.routerIdentity ?? null,
            supervisionDigest: first.supervision?.supervisionDigest ?? null
          },
          secondTurn: {
            compileActiveVersion: second.compileActiveVersion,
            compiledPackId: second.turn.ok ? second.turn.compileResponse.packId : null,
            compiledRouterIdentity: second.turn.ok ? second.turn.compileResponse.diagnostics.routerIdentity : null,
            selectedContextIds: second.turn.ok ? second.turn.compileResponse.selectedContext.map((block) => block.id) : [],
            nextActiveVersion: second.state.activePackVersion
          }
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
  console.error("[continuous-product-loop:smoke] failed");
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}
