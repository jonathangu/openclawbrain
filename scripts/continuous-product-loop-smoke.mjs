import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { materializeCandidatePack } from "../packages/learner/dist/src/index.js";
import { activatePack } from "../packages/activation/dist/src/index.js";
import { compileRuntimeContext, runContinuousProductLoopTurn } from "../packages/openclaw/dist/src/index.js";

function logStep(message) {
  console.log(`[continuous-product-loop:smoke] ${message}`);
}

function makeRootDir() {
  return path.join(tmpdir(), `openclawbrain-continuous-product-loop-${Date.now()}-${Math.random().toString(16).slice(2)}`);
}

function noteValue(notes, prefix) {
  const note = notes.find((entry) => entry.startsWith(prefix));
  return note === undefined ? null : note.slice(prefix.length);
}

function snapshotActiveCompile(activationRoot, message, runtimeHints) {
  const result = compileRuntimeContext({
    activationRoot,
    agentId: "continuous-loop-smoke",
    message,
    runtimeHints,
    maxContextBlocks: 3
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    throw new Error("active compile snapshot should succeed");
  }

  return {
    packId: result.compileResponse.packId,
    routerIdentity: result.compileResponse.diagnostics.routerIdentity,
    refreshStatus: noteValue(result.compileResponse.diagnostics.notes, "router_refresh_status="),
    weightsChecksum: noteValue(result.compileResponse.diagnostics.notes, "router_weights_checksum="),
    freshnessChecksum: noteValue(result.compileResponse.diagnostics.notes, "router_freshness_checksum="),
    noOpWarning: noteValue(result.compileResponse.diagnostics.notes, "router_noop_warning="),
    notes: [...result.compileResponse.diagnostics.notes]
  };
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

    const cycleInputs = [
      {
        label: "turn 1 through event export, supervision, learning, staging, and promotion",
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
          runtimeHints: ["freshness", "promotion", "evidence", "cycle-1"],
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
        promoteUpdatedAt: "2026-03-07T18:05:00.000Z",
        expectedReason: "attach_bootstrap"
      },
      {
        label: "turn 2 and proving compile now uses the fresher learned routing artifact",
        workspace: {
          workspaceId: "workspace-continuous-loop",
          snapshotId: "workspace-continuous-loop@snapshot-2",
          capturedAt: "2026-03-07T18:10:30.000Z",
          rootDir: "/workspace/openclawbrain",
          branch: "main",
          revision: "continuous-loop-rev-2",
          labels: ["openclaw", "runtime", "turn-2"]
        },
        turn: {
          agentId: "continuous-loop-smoke",
          sessionId: "session-continuous-loop",
          channel: "whatsapp",
          userMessage: "Compile the fresher learned route artifact after promotion.",
          runtimeHints: ["freshness", "promotion", "learned", "route", "artifact", "cycle-2"],
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
        promoteUpdatedAt: "2026-03-07T18:15:00.000Z",
        expectedReason: "fresh_live_events"
      },
      {
        label: "turn 3 with fresh runtime traffic and another learned refresh",
        workspace: {
          workspaceId: "workspace-continuous-loop",
          snapshotId: "workspace-continuous-loop@snapshot-3",
          capturedAt: "2026-03-07T18:20:30.000Z",
          rootDir: "/workspace/openclawbrain",
          branch: "main",
          revision: "continuous-loop-rev-3",
          labels: ["openclaw", "runtime", "turn-3"]
        },
        turn: {
          agentId: "continuous-loop-smoke",
          sessionId: "session-continuous-loop",
          channel: "whatsapp",
          userMessage: "Compile the promoted pack after another learned refresh.",
          runtimeHints: ["freshness", "promotion", "cycle-3", "another-refresh"],
          sequenceStart: 807,
          compile: {
            createdAt: "2026-03-07T18:20:00.000Z"
          },
          delivery: {
            createdAt: "2026-03-07T18:21:00.000Z",
            messageId: "msg-loop-3"
          },
          feedback: [
            {
              createdAt: "2026-03-07T18:22:00.000Z",
              content: "Keep the newest cycle-three route evidence ahead of older promotion guidance."
            }
          ]
        },
        candidateBuiltAt: "2026-03-07T18:23:00.000Z",
        stageUpdatedAt: "2026-03-07T18:24:00.000Z",
        promoteUpdatedAt: "2026-03-07T18:25:00.000Z",
        expectedReason: "fresh_live_events"
      }
    ];
    const turnResults = [];
    const compileSnapshots = [];
    let state;

    for (const [index, cycle] of cycleInputs.entries()) {
      logStep(`Running ${cycle.label}.`);
      const result = runContinuousProductLoopTurn({
        activationRoot,
        loopRoot,
        packLabel: "continuous-product-loop",
        workspace: cycle.workspace,
        turn: cycle.turn,
        ...(state === undefined ? {} : { state }),
        candidateBuiltAt: cycle.candidateBuiltAt,
        stageUpdatedAt: cycle.stageUpdatedAt,
        promoteUpdatedAt: cycle.promoteUpdatedAt
      });

      assert.equal(result.turn.eventExport.ok, true);
      assert.equal(result.turn.ok, true);
      if (!result.turn.ok) {
        throw new Error(`turn ${index + 1} compile should succeed`);
      }

      const expectedActiveVersion = index + 1;
      const expectedCompilePackId = index === 0 ? seedPack.manifest.packId : compileSnapshots[index - 1].packId;

      assert.equal(result.compileActiveVersion, expectedActiveVersion);
      assert.equal(result.compileActivePackId, expectedCompilePackId);
      assert.equal(result.turn.compileResponse.packId, expectedCompilePackId);
      assert.equal(result.learning.materializationReason, cycle.expectedReason);
      assert.equal(result.learning.promoted, true);
      assert.equal(result.learning.promotionAllowed, true);
      assert.notEqual(result.learning.candidatePack?.packId, result.turn.compileResponse.packId);

      state = result.state;
      turnResults.push(result);

      const snapshot = snapshotActiveCompile(
        activationRoot,
        `Compile the active pack after ${cycle.workspace.snapshotId}.`,
        cycle.turn.runtimeHints
      );
      assert.match(snapshot.notes.join(";"), /brain_boundary=promoted_pack_compile_only/);
      compileSnapshots.push(snapshot);
    }

    assert.equal(compileSnapshots[0].refreshStatus, "updated");
    assert.equal(compileSnapshots[0].noOpWarning, null);
    assert.equal(compileSnapshots[1].refreshStatus, "updated");
    assert.equal(compileSnapshots[1].noOpWarning, null);
    assert.notEqual(compileSnapshots[1].routerIdentity, compileSnapshots[0].routerIdentity);
    assert.notEqual(compileSnapshots[1].weightsChecksum, compileSnapshots[0].weightsChecksum);
    assert.notEqual(compileSnapshots[1].freshnessChecksum, compileSnapshots[0].freshnessChecksum);
    assert.equal(compileSnapshots[2].refreshStatus, "updated");
    assert.equal(compileSnapshots[2].noOpWarning, null);
    assert.notEqual(compileSnapshots[2].routerIdentity, compileSnapshots[1].routerIdentity);
    assert.notEqual(compileSnapshots[2].weightsChecksum, compileSnapshots[1].weightsChecksum);
    assert.notEqual(compileSnapshots[2].freshnessChecksum, compileSnapshots[1].freshnessChecksum);
    assert.equal(turnResults[2].state.activePackVersion, 4);

    logStep("Continuous post-attach multi-cycle product loop smoke passed.");
    console.log(
      JSON.stringify(
        {
          seedPackId: seedPack.manifest.packId,
          cycles: turnResults.map((result, index) => ({
            cycle: index + 1,
            compileActiveVersion: result.compileActiveVersion,
            compileActivePackId: result.compileActivePackId,
            hotPathPackId: result.turn.ok ? result.turn.compileResponse.packId : null,
            promotedPackId: result.state.currentActivePack?.packId ?? null,
            promotedRouterIdentity: result.state.currentActivePack?.routerIdentity ?? null,
            materializationReason: result.learning.materializationReason,
            promoted: result.learning.promoted,
            promotionAllowed: result.learning.promotionAllowed,
            supervisionDigest: result.supervision?.supervisionDigest ?? null,
            routerRefreshStatus: compileSnapshots[index]?.refreshStatus ?? null,
            routerWeightsChecksum: compileSnapshots[index]?.weightsChecksum ?? null,
            routerFreshnessChecksum: compileSnapshots[index]?.freshnessChecksum ?? null,
            routerNoOpWarning: compileSnapshots[index]?.noOpWarning ?? null,
            activeVersionAfterPromotion: result.state.activePackVersion
          }))
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
