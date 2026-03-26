import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { CONTRACT_IDS } from "@openclawbrain/contracts";
import { buildNormalizedEventExport } from "@openclawbrain/event-export";
import { materializeCandidatePackFromNormalizedEventExport } from "@openclawbrain/learner";
import { activatePack } from "@openclawbrain/pack-format";

import { compileRuntimeContext } from "../src/index.js";

function createWorkspace(t, label) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), `${label}-workspace-`));
  writeFileSync(
    path.join(rootDir, "README.md"),
    "# Comparative mode workspace\nGraph and vector routing context live here.\n",
    "utf8",
  );
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-03-25T00:00:00.000Z",
    rootDir,
    branch: "main",
    revision: `rev-${label}`,
    dirty: false,
    labels: ["test"],
    files: ["README.md"],
  };
}

function createNormalizedEventExport(label) {
  return buildNormalizedEventExport({
    interactionEvents: [
      {
        contract: CONTRACT_IDS.interactionEvents,
        eventId: `evt-${label}-1`,
        agentId: "agent",
        sessionId: `session-${label}`,
        channel: "cli",
        sequence: 1,
        kind: "memory_compiled",
        createdAt: "2026-03-25T00:01:00.000Z",
        source: { runtimeOwner: "openclaw", stream: "session.tail" },
        messageId: `msg-${label}-1`,
      },
    ],
    feedbackEvents: [],
  });
}

function createActivationRoot(t, label, learnedRouting) {
  const workspace = createWorkspace(t, label);
  const rootDir = mkdtempSync(path.join(os.tmpdir(), `${label}-activation-`));
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  const activationRoot = path.join(rootDir, "activation");
  const packRoot = path.join(rootDir, "pack");
  materializeCandidatePackFromNormalizedEventExport(packRoot, {
    packLabel: label,
    workspace,
    normalizedEventExport: createNormalizedEventExport(label),
    learnedRouting,
    builtAt: "2026-03-25T00:02:00.000Z",
    structuralOps: { connect: 2, split: 1, merge: 1, prune: 1 },
  });
  activatePack(activationRoot, packRoot, {
    updatedAt: "2026-03-25T00:03:00.000Z",
    reason: `${label}_activate`,
  });
  return activationRoot;
}

function noteValue(notes, prefix) {
  const match = notes.find((note) => note.startsWith(prefix));
  return match === undefined ? null : match.slice(prefix.length);
}

test("compileRuntimeContext maps comparative runtime modes to native compile paths", (t) => {
  const heuristicActivationRoot = createActivationRoot(t, "heuristic-compare", false);

  const vectorOnly = compileRuntimeContext({
    activationRoot: heuristicActivationRoot,
    message: "show the routing context",
    mode: "vector_only",
    maxContextBlocks: 2,
  });
  assert.equal(vectorOnly.ok, true);
  assert.equal(vectorOnly.compileResponse.diagnostics.modeRequested, "heuristic");
  assert.equal(vectorOnly.compileResponse.diagnostics.modeEffective, "heuristic");
  assert.equal(vectorOnly.compileResponse.diagnostics.usedLearnedRouteFn, false);
  assert.equal(noteValue(vectorOnly.compileResponse.diagnostics.notes, "comparative_mode="), "vector_only");
  assert.equal(noteValue(vectorOnly.compileResponse.diagnostics.notes, "selection_engine="), "flat_rank_v1");
  assert.equal(noteValue(vectorOnly.compileResponse.diagnostics.notes, "selection_graph_walk="), null);

  const graphPriorOnly = compileRuntimeContext({
    activationRoot: heuristicActivationRoot,
    message: "show the routing context",
    mode: "graph_prior_only",
    maxContextBlocks: 2,
  });
  assert.equal(graphPriorOnly.ok, true);
  assert.equal(graphPriorOnly.compileResponse.diagnostics.modeRequested, "heuristic");
  assert.equal(graphPriorOnly.compileResponse.diagnostics.modeEffective, "heuristic");
  assert.equal(graphPriorOnly.compileResponse.diagnostics.usedLearnedRouteFn, false);
  assert.equal(noteValue(graphPriorOnly.compileResponse.diagnostics.notes, "comparative_mode="), "graph_prior_only");
  assert.equal(noteValue(graphPriorOnly.compileResponse.diagnostics.notes, "selection_engine="), "graph_walk_v1");
  assert.equal(noteValue(graphPriorOnly.compileResponse.diagnostics.notes, "selection_graph_walk="), "graph_walk_v1");

  const learnedActivationRoot = createActivationRoot(t, "learned-compare", true);
  const learnedRoute = compileRuntimeContext({
    activationRoot: learnedActivationRoot,
    message: "show the routing context",
    mode: "learned_route",
    maxContextBlocks: 2,
  });
  assert.equal(learnedRoute.ok, true);
  assert.equal(learnedRoute.compileResponse.diagnostics.modeRequested, "learned");
  assert.equal(learnedRoute.compileResponse.diagnostics.modeEffective, "learned");
  assert.equal(learnedRoute.compileResponse.diagnostics.usedLearnedRouteFn, true);
  assert.equal(noteValue(learnedRoute.compileResponse.diagnostics.notes, "comparative_mode="), "learned_route");
  assert.equal(noteValue(learnedRoute.compileResponse.diagnostics.notes, "selection_engine="), "graph_walk_v1");
  assert.equal(noteValue(learnedRoute.compileResponse.diagnostics.notes, "selection_graph_walk="), "graph_walk_v1");
});
