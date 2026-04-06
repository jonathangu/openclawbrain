import { describe, it, expect } from "vitest";
import {
  applyWeightUpdates,
  collectReinforceUpdateContributions,
  collectTeacherActionDistillContributions,
  computeReinforceUpdates,
  computeTeacherActionUpdates,
  mergePolicyWeightUpdates,
  updateBaseline,
} from "../../src/brain-core/update.js";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { START_NODE_ID } from "../../src/brain-core/types.js";
import type { Episode, TrajectoryExpansion, TrajectoryStep, BrainNode, BrainEdge, PolicyGradientSupervisionArtifact } from "../../src/brain-core/types.js";

function makeNode(id: string): BrainNode {
  return {
    id, kind: "chunk", content: `content of ${id}`,
    embedding: new Float32Array([1, 0, 0]), sourceUri: null,
    trust: "scanner", tags: [], tokenCount: 100, metadata: {},
    createdAt: Date.now(), updatedAt: Date.now(),
  };
}

function makeToolNode(id: string): BrainNode {
  return {
    id, kind: "toolcard", content: `tool ${id}`,
    embedding: new Float32Array([1, 0, 0]), sourceUri: null,
    trust: "scanner", tags: [], tokenCount: 100, metadata: {},
    createdAt: Date.now(), updatedAt: Date.now(),
  };
}

function makeEdge(source: string, target: string, weight = 0.5): BrainEdge {
  return {
    source, target, kind: "learned", weight, prior: 0.5,
    metadata: {}, decayedAt: Date.now(), createdAt: Date.now(),
  };
}

function makeStep(sourceId: string | null, targetId: string, prob: number, expansionIndex = 0): TrajectoryStep {
  return {
    stateSnapshot: {
      sourceNodeId: sourceId,
      expansionIndex,
      selectionIndex: 0,
      budgetRemaining: 1000,
      initialBudget: 1000,
      reservedTokenCost: 0,
      maxHops: 8,
      frontierSize: 0,
      frontierNodeIds: [],
      visitedCount: 0,
      firedCount: 0,
    },
    candidates: [
      { action: { type: "traverse", targetNodeId: targetId }, score: 1, probability: prob },
      { action: { type: "stop_local" }, score: -1, probability: 1 - prob },
    ],
    chosenAction: { type: "traverse", targetNodeId: targetId },
    chosenActionProbability: prob,
    stopProbability: 1 - prob,
  };
}

function makeExpansion(sourceId: string | null, targetId: string, prob: number, expansionIndex = 0): TrajectoryExpansion {
  const substep = makeStep(sourceId, targetId, prob, expansionIndex);
  return {
    sourceNodeId: sourceId,
    expansionIndex,
    frontierBefore: sourceId === null ? [] : [sourceId],
    frontierAfter: [],
    budgetBefore: 1000,
    budgetAfter: 900,
    substeps: [substep],
    selectedTargets: [targetId],
    acceptedTargets: [targetId],
    vetoedTargets: [],
  };
}

function makeEpisode(trajectory: TrajectoryExpansion[], reward: number | null): Episode {
  return {
    id: "test-ep", conversationId: null, queryText: "test",
    queryEmbedding: null, trajectory, firedNodes: [], vetoedNodes: [],
    contextChars: 0, reward, rewardSource: reward !== null ? "self" : null,
    packVersion: null, createdAt: Date.now(),
  };
}

describe("update (REINFORCE, Lemma 6.1)", () => {
  it("positive reward strengthens chosen edges", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.6)], 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates.length).toBe(1);
    expect(updates[0]).toMatchObject({ kind: "edge", source: "a", target: "b" });
    expect(updates[0].delta).toBeGreaterThan(0); // Positive reward → strengthen
  });

  it("negative reward weakens chosen edges", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.6)], -1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates.length).toBe(1);
    expect(updates[0].delta).toBeLessThan(0); // Negative reward → weaken
  });

  it("baseline reduces update magnitude", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.6)], 0.5);

    const updatesNoBaseline = computeReinforceUpdates(episode, 0.1, 0.0);
    const updatesWithBaseline = computeReinforceUpdates(episode, 0.1, 0.4);

    // With baseline closer to reward, advantage is smaller → smaller update
    expect(Math.abs(updatesWithBaseline[0].delta)).toBeLessThan(
      Math.abs(updatesNoBaseline[0].delta),
    );
  });

  it("full-trajectory credit: ALL steps get credit, not just last", () => {
    // Episode with 3 steps: a→b, b→c, c→d
    const trajectory = [
      makeExpansion("a", "b", 0.5, 0),
      makeExpansion("b", "c", 0.5, 1),
      makeExpansion("c", "d", 0.5, 2),
    ];
    const episode = makeEpisode(trajectory, 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    // All 3 edges should receive updates (full-trajectory sum)
    expect(updates.length).toBe(3);

    const edges = updates
      .filter((u): u is Extract<(typeof updates)[number], { kind: "edge" }> => u.kind === "edge")
      .map((u) => `${u.source}→${u.target}`);
    expect(edges).toContain("a→b");
    expect(edges).toContain("b→c");
    expect(edges).toContain("c→d");

    // All updates should be positive (positive reward, zero baseline)
    for (const u of updates) {
      expect(u.delta).toBeGreaterThan(0);
    }
  });

  it("preserves exact substep contributions when repeated choices collapse into one net update", () => {
    const first = makeStep("a", "b", 0.6, 0);
    const second = makeStep("a", "b", 0.3, 0);
    second.stateSnapshot.selectionIndex = 1;
    second.stateSnapshot.firedCount = 1;
    const repeatedExpansion: TrajectoryExpansion = {
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: ["b"],
      budgetBefore: 1000,
      budgetAfter: 800,
      substeps: [first, second],
      selectedTargets: ["b"],
      acceptedTargets: ["b"],
      vetoedTargets: [],
    };
    const episode = makeEpisode([repeatedExpansion], 1.0);

    const contributions = collectReinforceUpdateContributions(episode, 0.1, 0.0);
    expect(contributions).toHaveLength(2);
    expect(contributions[0]).toMatchObject({
      updateKey: "a→b",
      kind: "edge",
      sourceNodeId: "a",
      targetNodeId: "b",
      expansionIndex: 0,
      selectionIndex: 0,
      chosenActionProbability: 0.6,
    });
    expect(contributions[0]?.delta).toBeCloseTo(0.04, 10);
    expect(contributions[1]).toMatchObject({
      updateKey: "a→b",
      kind: "edge",
      sourceNodeId: "a",
      targetNodeId: "b",
      expansionIndex: 0,
      selectionIndex: 1,
      chosenActionProbability: 0.3,
    });
    expect(contributions[1]?.delta).toBeCloseTo(0.07, 10);

    const updates = computeReinforceUpdates(episode, 0.1, 0.0);
    expect(updates).toHaveLength(1);
    expect(updates[0]).toMatchObject({
      kind: "edge",
      source: "a",
      target: "b",
    });
    expect(updates[0]?.delta).toBeCloseTo(0.11, 10);
  });

  it("updates seed-phase transitions through explicit seed weights", () => {
    const episode = makeEpisode([makeExpansion(null, "b", 0.6)], 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);

    expect(updates).toEqual([
      expect.objectContaining({
        kind: "seed",
        nodeId: "b",
      }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
  });

  it("updates seed-phase tool choices through explicit tool-action priors", () => {
    const graph = new BrainGraph();
    graph.addNode(makeToolNode("tool:proof"));

    const episode = makeEpisode([makeExpansion(null, "tool:proof", 0.6)], 1.0);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0, graph);

    expect(updates).toEqual([
      expect.objectContaining({
        kind: "tool_action",
        sourceNodeId: START_NODE_ID,
        toolNodeId: "tool:proof",
      }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
  });

  it("emits tool-action updates for chosen toolcard traversals when the graph identifies the tool node", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("a"));
    graph.addNode(makeToolNode("tool:proof"));

    const toolExpansion: TrajectoryExpansion = {
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: ["tool:proof"],
      budgetBefore: 1000,
      budgetAfter: 900,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "traverse", targetNodeId: "tool:proof" }, score: 0.9, probability: 0.7 },
            { action: { type: "stop_local" }, score: 0.1, probability: 0.3 },
          ],
          chosenAction: { type: "traverse", targetNodeId: "tool:proof" },
          chosenActionProbability: 0.7,
          stopProbability: 0.3,
        },
      ],
      selectedTargets: ["tool:proof"],
      acceptedTargets: ["tool:proof"],
      vetoedTargets: [],
    };

    const updates = computeReinforceUpdates(makeEpisode([toolExpansion], 1.0), 0.1, 0.0, graph);
    expect(updates).toEqual([
      expect.objectContaining({
        kind: "tool_action",
        sourceNodeId: "a",
        toolNodeId: "tool:proof",
      }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
  });

  it("direct teacher-action distillation can move tool priors even when reward advantage is flat", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("source"));
    graph.addNode(makeToolNode("tool:proof"));

    const episode = makeEpisode([
      makeExpansion("source", "tool:proof", 0.7),
    ], 0.5);

    const supervision: PolicyGradientSupervisionArtifact[] = [{
      supervisionId: "sup-1",
      traceId: "trace-1",
      source: "teacher",
      kind: "teacher_review",
      value: 0.5,
      confidence: 1.0,
      reason: "prefer the tool",
      labelId: "label-1",
      evidenceId: "evidence-1",
      observationId: "obs-1",
      teacherTraceId: "teacher-trace-1",
      serveDecisionRecordId: null,
      selectionDigest: null,
      turnCompileEventId: null,
      activePackGraphChecksum: null,
      bindingMode: "exact_decision_id",
      attributionQuality: "exact",
      feedbackRichness: "tool_only",
      traceRequestDigest: null,
      traceSelectedNodeIds: ["source", "tool:proof"],
      traceSelectedPathNodeIds: ["source", "tool:proof"],
    }];

    const reinforceUpdates = computeReinforceUpdates(episode, 0.1, 0.5, graph);
    expect(reinforceUpdates).toHaveLength(0);

    const teacherContributions = collectTeacherActionDistillContributions(episode, 0.1, supervision, graph);
    expect(teacherContributions).toHaveLength(1);
    expect(teacherContributions[0]).toMatchObject({
      kind: "tool_action",
      sourceNodeId: "source",
      targetNodeId: "tool:proof",
    });

    const teacherUpdates = computeTeacherActionUpdates(episode, 0.1, supervision, graph);
    expect(teacherUpdates).toEqual([
      expect.objectContaining({
        kind: "tool_action",
        sourceNodeId: "source",
        toolNodeId: "tool:proof",
      }),
    ]);
    expect(teacherUpdates[0]?.delta).toBeGreaterThan(0);

    const merged = mergePolicyWeightUpdates([...reinforceUpdates, ...teacherUpdates]);
    expect(merged).toHaveLength(1);
    expect(merged[0]).toMatchObject({
      kind: "tool_action",
      sourceNodeId: "source",
      toolNodeId: "tool:proof",
    });
  });

  it("distills seed-phase tool choices into tool-action priors", () => {
    const graph = new BrainGraph();
    graph.addNode(makeToolNode("tool:proof"));

    const episode = makeEpisode([makeExpansion(null, "tool:proof", 0.7)], 0.5);
    const supervision: PolicyGradientSupervisionArtifact[] = [{
      supervisionId: "sup-seed-tool",
      traceId: "trace-seed-tool",
      source: "teacher",
      kind: "teacher_review",
      value: 0.6,
      confidence: 1.0,
      reason: "use the tool directly from the seed phase",
      labelId: "label-seed-tool",
      evidenceId: "evidence-seed-tool",
      observationId: "obs-seed-tool",
      teacherTraceId: "teacher-trace-seed-tool",
      serveDecisionRecordId: null,
      selectionDigest: null,
      turnCompileEventId: null,
      activePackGraphChecksum: null,
      bindingMode: "exact_decision_id",
      attributionQuality: "exact",
      feedbackRichness: "tool_only",
      traceRequestDigest: null,
      traceSelectedNodeIds: ["tool:proof"],
      traceSelectedPathNodeIds: ["tool:proof"],
    }];

    const updates = computeTeacherActionUpdates(episode, 0.1, supervision, graph);

    expect(updates).toEqual([
      expect.objectContaining({
        kind: "tool_action",
        sourceNodeId: START_NODE_ID,
        toolNodeId: "tool:proof",
      }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
  });

  it("prefers teacher-selected action targets over the rest of the sampled trajectory", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("source"));
    graph.addNode(makeNode("mid"));
    graph.addNode(makeToolNode("tool:proof"));

    const episode = makeEpisode([
      {
        sourceNodeId: "source",
        expansionIndex: 0,
        frontierBefore: ["source"],
        frontierAfter: ["mid"],
        budgetBefore: 1000,
        budgetAfter: 900,
        substeps: [
          {
            stateSnapshot: {
              sourceNodeId: "source",
              expansionIndex: 0,
              selectionIndex: 0,
              budgetRemaining: 1000,
              initialBudget: 1000,
              reservedTokenCost: 0,
              maxHops: 8,
              frontierSize: 0,
              frontierNodeIds: [],
              visitedCount: 0,
              firedCount: 0,
            },
            candidates: [
              { action: { type: "traverse", targetNodeId: "mid" }, score: 0.7, probability: 0.6 },
              { action: { type: "stop_local" }, score: -0.2, probability: 0.4 },
            ],
            chosenAction: { type: "traverse", targetNodeId: "mid" },
            chosenActionProbability: 0.6,
            stopProbability: 0.4,
          },
        ],
        selectedTargets: ["mid"],
        acceptedTargets: ["mid"],
        vetoedTargets: [],
      },
      {
        sourceNodeId: "mid",
        expansionIndex: 1,
        frontierBefore: ["mid"],
        frontierAfter: ["tool:proof"],
        budgetBefore: 900,
        budgetAfter: 820,
        substeps: [
          {
            stateSnapshot: {
              sourceNodeId: "mid",
              expansionIndex: 1,
              selectionIndex: 0,
              budgetRemaining: 900,
              initialBudget: 1000,
              reservedTokenCost: 0,
              maxHops: 8,
              frontierSize: 0,
              frontierNodeIds: [],
              visitedCount: 1,
              firedCount: 1,
            },
            candidates: [
              { action: { type: "traverse", targetNodeId: "tool:proof" }, score: 0.9, probability: 0.7 },
              { action: { type: "stop_local" }, score: 0.1, probability: 0.3 },
            ],
            chosenAction: { type: "traverse", targetNodeId: "tool:proof" },
            chosenActionProbability: 0.7,
            stopProbability: 0.3,
          },
        ],
        selectedTargets: ["tool:proof"],
        acceptedTargets: ["tool:proof"],
        vetoedTargets: [],
      },
      {
        sourceNodeId: "tool:proof",
        expansionIndex: 2,
        frontierBefore: ["tool:proof"],
        frontierAfter: [],
        budgetBefore: 820,
        budgetAfter: 820,
        substeps: [
          {
            stateSnapshot: {
              sourceNodeId: "tool:proof",
              expansionIndex: 2,
              selectionIndex: 0,
              budgetRemaining: 820,
              initialBudget: 1000,
              reservedTokenCost: 0,
              maxHops: 8,
              frontierSize: 0,
              frontierNodeIds: [],
              visitedCount: 2,
              firedCount: 2,
            },
            candidates: [
              { action: { type: "traverse", targetNodeId: "source" }, score: 0.1, probability: 0.2 },
              { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
            ],
            chosenAction: { type: "stop_local" },
            chosenActionProbability: 0.8,
            stopProbability: 0.8,
          },
        ],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [],
      },
    ], 0.5);

    const supervision: PolicyGradientSupervisionArtifact[] = [{
      supervisionId: "sup-2",
      traceId: "trace-2",
      source: "teacher",
      kind: "teacher_review",
      value: 0.5,
      confidence: 1.0,
      reason: "prefer the tool and stop once the answer is grounded",
      labelId: "label-2",
      evidenceId: "evidence-2",
      observationId: "obs-2",
      teacherTraceId: "teacher-trace-2",
      serveDecisionRecordId: null,
      selectionDigest: null,
      turnCompileEventId: null,
      activePackGraphChecksum: null,
      bindingMode: "exact_decision_id",
      attributionQuality: "exact",
      feedbackRichness: "followup_and_tool",
      traceRequestDigest: null,
      traceSelectedNodeIds: ["tool:proof"],
      traceSelectedPathNodeIds: ["tool:proof"],
    }];

    const contributions = collectTeacherActionDistillContributions(episode, 0.1, supervision, graph);
    expect(contributions).toHaveLength(2);
    expect(contributions[0]).toMatchObject({
      kind: "tool_action",
      sourceNodeId: "mid",
      targetNodeId: "tool:proof",
    });
    expect(contributions[1]).toMatchObject({
      kind: "stop_local",
      sourceNodeId: "tool:proof",
    });

    const updates = computeTeacherActionUpdates(episode, 0.1, supervision, graph);
    expect(updates).toHaveLength(2);
    expect(updates).toEqual([
      expect.objectContaining({ kind: "tool_action", sourceNodeId: "mid", toolNodeId: "tool:proof" }),
      expect.objectContaining({ kind: "stop_local", sourceNodeId: "tool:proof" }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
    expect(updates[1]?.delta).toBeGreaterThan(0);
  });

  it("distills teacher-selected paths directly into traverse/tool/STOP_LOCAL updates even when the sampled route disagrees", () => {
    const graph = new BrainGraph();
    graph.addNode(makeNode("source"));
    graph.addNode(makeNode("mid"));
    graph.addNode(makeToolNode("tool:proof"));

    const episode = makeEpisode([
      {
        sourceNodeId: "source",
        expansionIndex: 0,
        frontierBefore: ["source"],
        frontierAfter: ["mid"],
        budgetBefore: 1000,
        budgetAfter: 900,
        substeps: [
          {
            stateSnapshot: {
              sourceNodeId: "source",
              expansionIndex: 0,
              selectionIndex: 0,
              budgetRemaining: 1000,
              initialBudget: 1000,
              reservedTokenCost: 0,
              maxHops: 8,
              frontierSize: 0,
              frontierNodeIds: [],
              visitedCount: 0,
              firedCount: 0,
            },
            candidates: [
              { action: { type: "traverse", targetNodeId: "mid" }, score: 0.7, probability: 0.6 },
              { action: { type: "stop_local" }, score: -0.2, probability: 0.4 },
            ],
            chosenAction: { type: "traverse", targetNodeId: "mid" },
            chosenActionProbability: 0.6,
            stopProbability: 0.4,
          },
        ],
        selectedTargets: ["mid"],
        acceptedTargets: ["mid"],
        vetoedTargets: [],
      },
      {
        sourceNodeId: "mid",
        expansionIndex: 1,
        frontierBefore: ["mid"],
        frontierAfter: ["tool:proof"],
        budgetBefore: 900,
        budgetAfter: 820,
        substeps: [
          {
            stateSnapshot: {
              sourceNodeId: "mid",
              expansionIndex: 1,
              selectionIndex: 0,
              budgetRemaining: 900,
              initialBudget: 1000,
              reservedTokenCost: 0,
              maxHops: 8,
              frontierSize: 0,
              frontierNodeIds: [],
              visitedCount: 1,
              firedCount: 1,
            },
            candidates: [
              { action: { type: "traverse", targetNodeId: "tool:proof" }, score: 0.9, probability: 0.7 },
              { action: { type: "stop_local" }, score: 0.1, probability: 0.3 },
            ],
            chosenAction: { type: "traverse", targetNodeId: "tool:proof" },
            chosenActionProbability: 0.7,
            stopProbability: 0.3,
          },
        ],
        selectedTargets: ["tool:proof"],
        acceptedTargets: ["tool:proof"],
        vetoedTargets: [],
      },
      {
        sourceNodeId: "tool:proof",
        expansionIndex: 2,
        frontierBefore: ["tool:proof"],
        frontierAfter: [],
        budgetBefore: 820,
        budgetAfter: 820,
        substeps: [
          {
            stateSnapshot: {
              sourceNodeId: "tool:proof",
              expansionIndex: 2,
              selectionIndex: 0,
              budgetRemaining: 820,
              initialBudget: 1000,
              reservedTokenCost: 0,
              maxHops: 8,
              frontierSize: 0,
              frontierNodeIds: [],
              visitedCount: 2,
              firedCount: 2,
            },
            candidates: [
              { action: { type: "traverse", targetNodeId: "source" }, score: 0.1, probability: 0.2 },
              { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
            ],
            chosenAction: { type: "stop_local" },
            chosenActionProbability: 0.8,
            stopProbability: 0.8,
          },
        ],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [],
      },
    ], 0.5);

    const lowConfidenceSupervision: PolicyGradientSupervisionArtifact[] = [{
      supervisionId: "sup-low",
      traceId: "trace-low",
      source: "teacher",
      kind: "teacher_review",
      value: 0.8,
      confidence: 0.3,
      reason: "teacher prefers the direct route and stopping once grounded",
      labelId: "label-low",
      evidenceId: "evidence-low",
      observationId: "obs-low",
      teacherTraceId: "teacher-trace-low",
      serveDecisionRecordId: null,
      selectionDigest: null,
      turnCompileEventId: null,
      activePackGraphChecksum: null,
      bindingMode: "exact_decision_id",
      attributionQuality: "exact",
      feedbackRichness: "followup_and_tool",
      traceRequestDigest: null,
      traceSelectedNodeIds: ["tool:proof"],
      traceSelectedPathNodeIds: ["source", "tool:proof"],
    }];

    const highConfidenceSupervision: PolicyGradientSupervisionArtifact[] = [{
      ...lowConfidenceSupervision[0],
      supervisionId: "sup-high",
      traceId: "trace-high",
      confidence: 0.9,
    }];

    const lowContributions = collectTeacherActionDistillContributions(episode, 0.1, lowConfidenceSupervision, graph);
    expect(lowContributions).toHaveLength(2);
    expect(lowContributions).toEqual([
      expect.objectContaining({
        kind: "tool_action",
        sourceNodeId: "source",
        targetNodeId: "tool:proof",
      }),
      expect.objectContaining({
        kind: "stop_local",
        sourceNodeId: "tool:proof",
      }),
    ]);

    const highContributions = collectTeacherActionDistillContributions(episode, 0.1, highConfidenceSupervision, graph);
    expect(highContributions).toHaveLength(2);
    expect(Math.abs(lowContributions[0]?.delta ?? 0)).toBeLessThan(Math.abs(highContributions[0]?.delta ?? 0));

    const updates = computeTeacherActionUpdates(episode, 0.1, lowConfidenceSupervision, graph);
    expect(updates).toEqual([
      expect.objectContaining({
        kind: "tool_action",
        sourceNodeId: "source",
        toolNodeId: "tool:proof",
      }),
      expect.objectContaining({
        kind: "stop_local",
        sourceNodeId: "tool:proof",
      }),
    ]);
  });

  it("zero advantage produces no updates", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.5)], 0.5);
    const updates = computeReinforceUpdates(episode, 0.1, 0.5); // baseline = reward
    expect(updates.length).toBe(0);
  });

  it("null reward produces no updates", () => {
    const episode = makeEpisode([makeExpansion("a", "b", 0.5)], null);
    const updates = computeReinforceUpdates(episode, 0.1, 0.0);
    expect(updates.length).toBe(0);
  });

  it("emits learned updates for chosen stop_local substeps", () => {
    const stopExpansion: TrajectoryExpansion = {
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "traverse", targetNodeId: "b" }, score: 0.2, probability: 0.2 },
            { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 0.8,
          stopProbability: 0.8,
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
    };

    const updates = computeReinforceUpdates(makeEpisode([stopExpansion], 1.0), 0.1, 0.0);
    expect(updates).toEqual([
      expect.objectContaining({
        kind: "stop_local",
        sourceNodeId: "a",
      }),
    ]);
    expect(updates[0]?.delta).toBeGreaterThan(0);
  });

  it("negative reward weakens chosen stop_local weights", () => {
    const stopExpansion: TrajectoryExpansion = {
      sourceNodeId: null,
      expansionIndex: 0,
      frontierBefore: [],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: null,
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "traverse", targetNodeId: "b", seedScore: 0.2 }, score: 0.2, probability: 0.2 },
            { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 0.8,
          stopProbability: 0.8,
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
    };

    const updates = computeReinforceUpdates(makeEpisode([stopExpansion], -1.0), 0.1, 0.0);
    expect(updates).toEqual([
      expect.objectContaining({
        kind: "stop_local",
        sourceNodeId: START_NODE_ID,
      }),
    ]);
    expect(updates[0]?.delta).toBeLessThan(0);
  });

  it("does not emit learned stop_local updates for explicitly forced stops", () => {
    const stopExpansion: TrajectoryExpansion = {
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "traverse", targetNodeId: "b" }, score: 0.2, probability: 0.2 },
            { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 0.8,
          stopProbability: 0.8,
          stopTruth: "forced",
          stopReason: "frontier_cap",
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
      terminationReason: "frontier_cap",
    };

    const updates = computeReinforceUpdates(makeEpisode([stopExpansion], 1.0), 0.1, 0.0);
    expect(updates).toEqual([]);
  });

  it("does not emit learned stop_local updates for legacy forced-stop reasons when stopTruth is missing", () => {
    const stopExpansion: TrajectoryExpansion = {
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 0,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "traverse", targetNodeId: "b" }, score: 0.2, probability: 0.2 },
            { action: { type: "stop_local" }, score: 0.8, probability: 0.8 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 0.8,
          stopProbability: 0.8,
          stopReason: "frontier_cap",
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
      terminationReason: "frontier_cap",
    };

    const updates = computeReinforceUpdates(makeEpisode([stopExpansion], 1.0), 0.1, 0.0);
    expect(updates).toEqual([]);
  });

  it("does not emit fake stop_local updates when STOP_LOCAL is forced", () => {
    const updates = computeReinforceUpdates(makeEpisode([{
      sourceNodeId: "a",
      expansionIndex: 0,
      frontierBefore: ["a"],
      frontierAfter: [],
      budgetBefore: 1000,
      budgetAfter: 1000,
      substeps: [
        {
          stateSnapshot: {
            sourceNodeId: "a",
            expansionIndex: 0,
            selectionIndex: 0,
            budgetRemaining: 1000,
            initialBudget: 1000,
            reservedTokenCost: 1000,
            maxHops: 8,
            frontierSize: 0,
            frontierNodeIds: [],
            visitedCount: 0,
            firedCount: 0,
          },
          candidates: [
            { action: { type: "stop_local" }, score: 0.8, probability: 1 },
          ],
          chosenAction: { type: "stop_local" },
          chosenActionProbability: 1,
          stopProbability: 1,
        },
      ],
      selectedTargets: [],
      acceptedTargets: [],
      vetoedTargets: [],
    }], 1.0), 0.1, 0.0);
    expect(updates).toEqual([]);
  });

  describe("updateBaseline", () => {
    it("moves baseline toward new reward", () => {
      const newBaseline = updateBaseline(0.0, 1.0, 0.1);
      expect(newBaseline).toBeCloseTo(0.1);
    });

    it("converges over many updates", () => {
      let baseline = 0.0;
      for (let i = 0; i < 100; i++) {
        baseline = updateBaseline(baseline, 0.5, 0.1);
      }
      expect(baseline).toBeCloseTo(0.5, 1);
    });
  });

  describe("applyWeightUpdates", () => {
    it("modifies edge weight in graph", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b"));
      graph.addEdge(makeEdge("a", "b", 0.5));

      applyWeightUpdates(graph, [{ kind: "edge", source: "a", target: "b", delta: 0.2 }]);

      const edge = graph.getEdge("a", "b");
      expect(edge?.weight).toBeCloseTo(0.7);
    });

    it("creates valid updates for seed weights too", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("b"));
      graph.setSeedWeight("b", 0.5);

      applyWeightUpdates(graph, [{ kind: "seed", nodeId: "b", delta: 0.2 }]);

      expect(graph.getSeedWeight("b")).toBeCloseTo(0.7);
    });

    it("creates valid updates for stop_local weights too", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.setStopLocalWeight("a", 0.5);

      applyWeightUpdates(graph, [{ kind: "stop_local", sourceNodeId: "a", delta: 0.2 }]);

      expect(graph.getStopLocalWeight("a")).toBeCloseTo(0.7);
    });

    it("creates valid updates for tool action priors too", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeToolNode("tool:proof"));
      graph.setToolActionPrior("a", "tool:proof", 0.5);

      applyWeightUpdates(graph, [{ kind: "tool_action", sourceNodeId: "a", toolNodeId: "tool:proof", delta: 0.2 }]);

      expect(graph.getToolActionPrior("a", "tool:proof")).toBeCloseTo(0.7);
    });

    it("clamps weights to [-10, 10]", () => {
      const graph = new BrainGraph();
      graph.addNode(makeNode("a"));
      graph.addNode(makeNode("b"));
      graph.addEdge(makeEdge("a", "b", 9.5));

      applyWeightUpdates(graph, [{ kind: "edge", source: "a", target: "b", delta: 5.0 }]);

      const edge = graph.getEdge("a", "b");
      expect(edge?.weight).toBe(10);
    });
  });
});
