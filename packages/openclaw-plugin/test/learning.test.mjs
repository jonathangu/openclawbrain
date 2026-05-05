import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { BackgroundLearner, RouteFn, RouteTeacher, buildRouteGraphSnapshot, normalizePluginConfig } from '../dist/index.js';
import { MemoryStore } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-learning-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

test('background learner resolves tool outcomes into route examples and policy snapshots', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      learning: { enabled: true, minExamplesForPolicyUpdate: 1, maxPositiveExamples: 5, maxNegativeExamples: 5 },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = store.insertMemory({
      agentId: 'main',
      type: 'workflow',
      content: 'For dependency setup, use pnpm install and verify package scripts.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:workflow:deps',
      tags: ['workflow', 'package-manager'],
      importance: 0.7,
      freshness: 1,
      confidence: 0.8,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    const route = store.insertRouteDecision({
      agentId: 'main',
      sessionId: 's1',
      turnId: 't1',
      runId: 'r1',
      route: 'retrieve_memory',
      confidence: 0.82,
      latencyTier: 'cached_route',
      syncLlmUsed: false,
      fallbackUsed: false,
      turnFrame: {
        summary: 'Install dependencies',
        userGoal: 'Install dependencies',
        taskType: 'coding',
        activeObjects: [{ kind: 'repo', value: 'openclawbrain' }],
        impliedNeeds: ['Need package-manager corrections'],
        memoryQuestions: [],
        constraints: [],
        routeHints: {
          likelyNeedsCorrections: true,
          likelyNeedsPreferences: false,
          likelyNeedsWorkflow: true,
          likelyNeedsProjectContext: true,
        },
      },
      retrievalPlan: {
        queries: ['install dependencies'],
        memoryTypes: ['workflow', 'correction'],
        requiredTags: [],
        excludedTags: [],
        graphDepth: 0,
        maxCandidates: 10,
      },
      injectionPlan: { maxItems: 3, maxChars: 300, preferredFormat: 'bullets' },
      selectedMemoryIds: [memory.id],
      omittedMemoryIds: [],
      reward: 0,
    });
    store.insertInjection({
      agentId: 'main',
      memoryId: memory.id,
      routeDecisionId: route.id,
      runId: 'r1',
      turnId: 't1',
      sessionId: 's1',
      query: 'install dependencies',
      rank: 1,
      score: 0.92,
    });

    const learner = new BackgroundLearner({ store, config });
    const report = learner.processOutcomeClassification('main', {
      agentId: 'main',
      sessionId: 's1',
      turnId: 't1',
      runId: 'r1',
      sourceHook: 'after_tool_call',
      latestUserMessageRedacted: 'Install dependencies',
      recentAssistantMessage: '',
      toolObservations: [{ toolName: 'exec', ok: true, resultSummary: 'pnpm install ok' }],
      recentInjections: [],
      metadata: {},
    });

    assert.equal(report.outcomeResolutions, 1);
    assert.equal(report.routeDecisionsResolved, 1);
    assert.equal(report.routeExamplesCreated, 1);
    assert.ok(report.snapshotId);

    const resolvedRoute = store.getRouteDecision(route.id);
    assert.equal(resolvedRoute.outcome, 'tool_success');
    assert.equal(store.countRouteExamples('main', 'positive'), 1);
    assert.ok(store.getActivePolicySnapshot('main'));
    const updatedMemory = store.getMemory(memory.id);
    assert.ok(updatedMemory.usefulCount >= 1);
    assert.ok(updatedMemory.importance > 0.7);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('background learner marks injected memories corrected at agent_end', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      learning: { enabled: true, minExamplesForPolicyUpdate: 1, maxPositiveExamples: 5, maxNegativeExamples: 5 },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = store.insertMemory({
      agentId: 'main',
      type: 'correction',
      content: 'Use pnpm instead of npm for this repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:package-manager',
      tags: ['correction', 'package-manager'],
      importance: 0.8,
      freshness: 1,
      confidence: 0.9,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    const route = store.insertRouteDecision({
      agentId: 'main',
      sessionId: 's2',
      turnId: 't2',
      runId: 'r2',
      route: 'high_confidence_correction_only',
      confidence: 0.91,
      latencyTier: 'sync_memory_planner',
      syncLlmUsed: true,
      fallbackUsed: false,
      turnFrame: {
        summary: 'Update install docs',
        userGoal: 'Update install docs',
        taskType: 'coding',
        activeObjects: [{ kind: 'repo', value: 'openclawbrain' }],
        impliedNeeds: ['Need package-manager corrections'],
        memoryQuestions: [],
        constraints: [],
        routeHints: {
          likelyNeedsCorrections: true,
          likelyNeedsPreferences: false,
          likelyNeedsWorkflow: false,
          likelyNeedsProjectContext: true,
        },
      },
      retrievalPlan: {
        queries: ['update install docs'],
        memoryTypes: ['correction'],
        requiredTags: [],
        excludedTags: [],
        graphDepth: 0,
        maxCandidates: 10,
      },
      injectionPlan: { maxItems: 2, maxChars: 200, preferredFormat: 'rules' },
      selectedMemoryIds: [memory.id],
      omittedMemoryIds: [],
      reward: 0,
    });
    store.insertInjection({
      agentId: 'main',
      memoryId: memory.id,
      routeDecisionId: route.id,
      runId: 'r2',
      turnId: 't2',
      sessionId: 's2',
      query: 'update install docs',
      rank: 1,
      score: 0.95,
    });

    const learner = new BackgroundLearner({ store, config });
    const report = learner.processAgentEnd('main', {
      agentId: 'main',
      sessionId: 's2',
      turnId: 't2',
      runId: 'r2',
      sourceHook: 'agent_end',
      latestUserMessageRedacted: 'No, use pnpm instead of npm here.',
      recentAssistantMessage: '',
      toolObservations: [],
      recentInjections: [],
      metadata: {},
    });

    assert.equal(report.outcomeResolutions, 1);
    const resolvedRoute = store.getRouteDecision(route.id);
    assert.equal(resolvedRoute.outcome, 'corrected_after_injection');
    const updatedMemory = store.getMemory(memory.id);
    assert.ok(updatedMemory.importance < 0.8);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});


test('route teacher stores graph-grounded counterfactuals and activates structured policy v2', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      learning: { enabled: true },
      routeLearning: {
        enabled: true,
        teacher: { enabled: true, maxRunsPerCycle: 5, minResolvedRewardMagnitude: 0 },
        policyV2: { enabled: true, minExamples: 1, shadowBeforeActivate: false },
      },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = store.insertMemory({
      agentId: 'main',
      type: 'workflow',
      content: 'For this repo, use pnpm test for test runs.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:workflow:pnpm-test',
      tags: ['workflow', 'test'],
      importance: 0.8,
      freshness: 1,
      confidence: 0.9,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    const route = store.insertRouteDecision({
      agentId: 'main',
      sessionId: 's3',
      turnId: 't3',
      runId: 'r3',
      route: 'no_memory',
      confidence: 0.4,
      latencyTier: 'no_extra_llm',
      syncLlmUsed: false,
      fallbackUsed: false,
      turnFrame: {
        summary: 'Run the tests',
        userGoal: 'Run the tests',
        taskType: 'coding',
        activeObjects: [{ kind: 'repo', value: 'openclawbrain' }],
        impliedNeeds: ['Need package-manager corrections'],
        memoryQuestions: [],
        constraints: [],
        routeHints: {
          likelyNeedsCorrections: true,
          likelyNeedsPreferences: false,
          likelyNeedsWorkflow: true,
          likelyNeedsProjectContext: true,
        },
      },
      retrievalPlan: {
        queries: ['test workflow'],
        memoryTypes: ['workflow', 'correction'],
        requiredTags: [],
        excludedTags: [],
        graphDepth: 0,
        maxCandidates: 10,
      },
      injectionPlan: { maxItems: 2, maxChars: 200, preferredFormat: 'bullets' },
      selectedMemoryIds: [],
      omittedMemoryIds: [memory.id],
      reward: 0,
    });
    store.resolveRouteDecision(route.id, 'tool_failure', -0.8);
    buildRouteGraphSnapshot(store, 'main', route.id, ['test workflow'], [memory], 0);

    const teacher = new RouteTeacher({ store, config });
    const report = await teacher.run('main');

    assert.equal(report.teacherRuns, 1);
    assert.ok(report.counterfactuals >= 2);
    assert.ok(report.examples >= 1);
    assert.ok(report.policySnapshotId);
    const runs = store.listRouteTeacherRuns('main', 10);
    assert.equal(runs[0].verdict, 'missed_recall');
    assert.deepEqual(runs[0].teacherMemoryIds, [memory.id]);
    const counterfactuals = store.listRouteCounterfactuals('main', route.id, 20);
    assert.ok(counterfactuals.some((cf) => cf.kind === 'no_memory'));
    assert.ok(counterfactuals.some((cf) => cf.kind === 'top_k_alternate'));
    const policy = store.getActivePolicySnapshotV2('main');
    assert.ok(policy);
    assert.equal(policy.version, 'route-policy-v2');
    assert.ok(policy.rules.some((rule) => rule.route === 'retrieve_memory' && rule.memoryTypes.includes('workflow')));

    const routeFn = new RouteFn({ config, store });
    const packet = {
      agentId: 'main',
      sessionId: 's3',
      sessionKey: 's3',
      turnId: 't4',
      runId: 'r4',
      sourceHook: 'before_prompt_build',
      latestUserMessageRedacted: 'Run the tests in this repo',
      recentAssistantMessage: '',
      toolObservations: [],
      recentInjections: [],
      metadata: {},
    };
    const plan = routeFn.plan(packet);
    assert.equal(plan.route, 'retrieve_memory');
    assert.equal(plan.shouldRetrieve, true);
    assert.ok(plan.retrievalPlan.memoryTypes.includes('workflow'));
    assert.equal(plan.policySnapshotId, policy.id);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
