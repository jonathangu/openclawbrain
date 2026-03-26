import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { buildNormalizedEventExport } from "@openclawbrain/event-export";
import { CONTRACT_IDS } from "@openclawbrain/contracts";
import { compileRuntime } from "@openclawbrain/compiler";
import { buildCandidatePackFromNormalizedEventExport, buildStopActionUpdateBlockId } from "../src/local-learner.js";

function createWorkspace(t) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "ocb-graph-walk-stop-"));
  writeFileSync(path.join(rootDir, "README.md"), "# Graph walk stop workspace\nRelevant routing context lives here.\n", "utf8");
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return {
    workspaceId: "ws-graph-walk-stop",
    snapshotId: "snapshot-graph-walk-stop",
    capturedAt: "2026-03-24T22:40:00.000Z",
    rootDir,
    branch: "main",
    revision: "graphwalk123",
    dirty: false,
    labels: ["test"],
    files: ["README.md"]
  };
}

test("graph-walk selection stops expanding when learned STOP_LOCAL outranks traversal", (t) => {
  const workspace = createWorkspace(t);
  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents: [
      {
        contract: CONTRACT_IDS.interactionEvents,
        eventId: "evt-i1",
        agentId: "agent",
        sessionId: "sess-graph-walk-stop",
        channel: "cli",
        sequence: 1,
        kind: "memory_compiled",
        createdAt: "2026-03-24T22:40:00.000Z",
        source: { runtimeOwner: "openclaw", stream: "session.tail" },
        messageId: "msg-1"
      }
    ],
    feedbackEvents: []
  });
  const pack = buildCandidatePackFromNormalizedEventExport({
    packLabel: "graph-walk-stop-pack",
    workspace,
    normalizedEventExport,
    learnedRouting: true,
    builtAt: "2026-03-24T22:41:00.000Z",
    structuralOps: { connect: 3, split: 1, merge: 1, prune: 1 },
    pgVersion: "v2",
    serveTimeDecisions: [],
    baselineState: {
      movingAverage: 0,
      count: 0,
      alpha: 0.1,
      lastUpdatedAt: "2026-03-24T22:40:00.000Z"
    }
  });
  const sourceBlock = pack.payloads.graph.blocks.find((block) => (block.edges?.length ?? 0) > 0);
  assert.ok(sourceBlock, "expected a source block with outgoing edges");
  pack.payloads.router = {
    ...(pack.payloads.router ?? {
      contract: CONTRACT_IDS.routerArtifact,
      routerIdentity: `${pack.summary.packId}:route_fn`,
      strategy: "learned_route_fn_v1",
      trainedAt: "2026-03-24T22:42:00.000Z",
      requiresLearnedRouting: true,
      training: {
        method: "policy_gradient_v2",
        status: "updated",
        supervisionCount: 1,
        updateCount: 1,
        weightsChecksum: "test",
        objectiveChecksum: "test",
        freshnessChecksum: "test",
        noOpReason: null,
        routeTraceCount: 0,
        collectedLabels: { total: 0, humanFeedback: 0, operatorOverride: 0, selfMemory: 0 },
        objective: {
          objective: "supervised_route_pg_v2",
          updateMechanism: "policy_gradient",
          updateVersion: "route_pg_update_v2",
          profile: "route_pg_profile_v2"
        }
      },
      traces: [],
      policyUpdates: []
    }),
    policyUpdates: [
      {
        blockId: buildStopActionUpdateBlockId(sourceBlock.id),
        delta: 12,
        evidenceCount: 3,
        rewardSum: 3,
        tokenWeights: {},
        traceIds: ["stop-local"]
      }
    ]
  };
  const response = compileRuntime({
    rootDir: "/tmp/graph-walk-stop-pack",
    manifestPath: "/tmp/graph-walk-stop-pack/manifest.json",
    graphPath: "/tmp/graph-walk-stop-pack/graph.json",
    vectorPath: "/tmp/graph-walk-stop-pack/vectors.json",
    routerPath: "/tmp/graph-walk-stop-pack/router.json",
    manifest: pack.manifest,
    graph: pack.payloads.graph,
    vectors: pack.payloads.vectors,
    router: pack.payloads.router
  }, {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent",
    userMessage: "show me the relevant policy context",
    maxContextBlocks: 4,
    modeRequested: "learned",
    activePackId: pack.summary.packId,
    runtimeHints: ["policy", "context"]
  }, {
    selectionMode: "graph_walk_v1"
  });
  assert.equal(response.structuralSignals?.graphWalkHopCount ?? 0, 0, "expected learned STOP_LOCAL to halt graph-walk expansion");
});
