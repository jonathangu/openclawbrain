import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { BackgroundLearner, RouteFn, RouteTeacher, buildRouteGraphSnapshot, maybeDistillAndStorePolicyV2, maybeDistillAndStorePolicyV3, normalizePluginConfig, validatePolicySnapshotV2, validatePolicySnapshotV3 } from '../dist/index.js';
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
        policyV3: { enabled: true, minFrames: 1, shadowBeforeActivate: false, maxHarmRate: 0.5 },
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
    assert.ok(report.policySnapshotV3Id);
    assert.equal(report.routeFramesV3, 1);
    assert.ok(report.pairExamplesV3 >= 1);
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

    const policyV3 = store.getActivePolicySnapshotV3('main');
    if (policyV3) {
      assert.equal(policyV3.version, 'route-policy-v3');
      assert.ok(policyV3.rules.some((rule) => rule.route === 'retrieve_memory' && rule.memoryTypes.includes('workflow')));
    }

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
    assert.equal(plan.policySnapshotId, (policyV3 || policy).id);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('policy v2 distiller stores gated active snapshots and route fn records matched rule id', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      routeLearning: {
        enabled: true,
        policyV2: { enabled: true, minExamples: 1, shadowBeforeActivate: false, maxNoisyInjectionRate: 0.5 },
      },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const decision = store.insertRouteDecision({
      agentId: 'main',
      route: 'no_memory',
      confidence: 0.5,
      latencyTier: 'no_extra_llm',
      syncLlmUsed: false,
      fallbackUsed: false,
      turnFrame: {
        summary: 'Thanks',
        userGoal: 'Thanks',
        taskType: 'other',
        activeObjects: [],
        impliedNeeds: [],
        memoryQuestions: [],
        constraints: [],
        routeHints: {
          likelyNeedsCorrections: false,
          likelyNeedsPreferences: false,
          likelyNeedsWorkflow: false,
          likelyNeedsProjectContext: false,
        },
      },
      retrievalPlan: { queries: [], memoryTypes: [], requiredTags: [], excludedTags: [], graphDepth: 0, maxCandidates: 5 },
      injectionPlan: { maxItems: 0, maxChars: 0, preferredFormat: 'none' },
      selectedMemoryIds: [],
      omittedMemoryIds: [],
      reward: 0,
    });
    store.insertRouteTrainingExampleV2({
      agentId: 'main',
      routeDecisionId: decision.id,
      exampleKind: 'correct_silence',
      taskType: 'other',
      turnSignals: ['thanks', 'ok'],
      route: 'no_memory',
      memoryTypes: [],
      queryTemplates: [],
      graphDepth: 0,
      confidence: 0.92,
      supportCount: 4,
      harmCount: 0,
      source: 'manual_eval',
      evidenceIds: [decision.id],
    });

    const report = maybeDistillAndStorePolicyV2(store, 'main', config);
    assert.equal(report.validation.ok, true);
    assert.equal(report.snapshot.status, 'active');
    assert.equal(report.snapshot.version, 'route-policy-v2');
    assert.ok(report.snapshot.rules[0].stats.support >= 4);

    const routeFn = new RouteFn({ config, store });
    const plan = routeFn.plan({
      agentId: 'main',
      sourceHook: 'before_prompt_build',
      latestUserMessageRedacted: 'thanks ok',
      toolObservations: [],
      recentInjections: [],
      metadata: {},
    });
    assert.equal(plan.route, 'no_memory');
    assert.equal(plan.shouldRetrieve, false);
    assert.equal(plan.matchedPolicyRuleId, report.snapshot.rules[0].id);
    assert.equal(plan.policySnapshotId, report.snapshot.id);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('policy v3 distiller stores action prototypes, bandit state, and active snapshots', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      routeLearning: {
        enabled: true,
        policyV3: { enabled: true, minFrames: 1, shadowBeforeActivate: false, maxHarmRate: 0.6 },
      },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });

    const frame1 = store.insertRouteFrameV3({
      agentId: 'main',
      routeDecisionId: 'd1',
      redactedTurnSummary: 'Run the tests in this repo',
      taskType: 'coding',
      turnSignals: ['tests', 'repo', 'workflow'],
      projectHint: 'openclawbrain',
      repoHint: 'openclawbrain',
      toolHints: ['exec'],
      routeHintFlags: ['needs_workflow', 'needs_project_context'],
      chosenActionId: 'a-retrieve-workflow',
      chosenRoute: 'retrieve_memory',
      chosenMemoryTypes: ['workflow'],
      chosenGraphDepth: 1,
      chosenSyncPlanner: 'no',
      outcome: 'tool_success',
      reward: 0.8,
      rewardComponents: { retrievalHelpGain: 0.8, noisyInjectionPenalty: 0 },
      payloadHash: 'p1',
    });
    store.insertRouteFrameV3({
      agentId: 'main',
      routeDecisionId: 'd2',
      redactedTurnSummary: 'Thanks ok',
      taskType: 'other',
      turnSignals: ['thanks', 'ok'],
      toolHints: [],
      routeHintFlags: [],
      chosenActionId: 'a-silence',
      chosenRoute: 'no_memory',
      chosenMemoryTypes: [],
      chosenGraphDepth: 0,
      chosenSyncPlanner: 'no',
      outcome: 'no_signal',
      reward: 0.4,
      rewardComponents: { retrievalHelpGain: 0, noisyInjectionPenalty: 0 },
      payloadHash: 'p2',
    });
    store.upsertRouteActionPrototypeV3({
      id: 'a-retrieve-workflow',
      agentId: 'main',
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      queryTemplateFamily: ['test workflow'],
      sparseSignature: ['coding', 'tests', 'workflow', 'openclawbrain'],
      denseEmbedding: [1, 0, 0.5],
      supportPrior: 2,
      harmPrior: 0,
      status: 'active',
      provenance: 'learned',
      sourceExampleIds: ['d1'],
    });
    store.upsertRouteActionPrototypeV3({
      id: 'a-silence',
      agentId: 'main',
      route: 'no_memory',
      memoryTypes: [],
      graphDepth: 0,
      syncPlanner: 'no',
      queryTemplateFamily: [],
      sparseSignature: ['other', 'thanks', 'ok'],
      denseEmbedding: [0.2, 0.1, 0],
      supportPrior: 1,
      harmPrior: 0,
      status: 'active',
      provenance: 'distilled',
      sourceExampleIds: ['d2'],
    });
    store.insertRoutePairExampleV3({
      agentId: 'main',
      frameId: frame1.id,
      positiveActionId: 'a-retrieve-workflow',
      negativeActionId: 'a-silence',
      labelSource: 'teacher',
      marginWeight: 0.8,
      evidenceIds: ['d1'],
    });
    store.upsertRouteBanditStateV3({
      agentId: 'main',
      learnerVersion: 'linucb-lite-v1',
      featureSchemaVersion: 'route-v3-hybrid-24d-v1',
      explorationAlpha: 0.35,
      sharedWeights: [1, 0.75],
      actionStats: {
        'a-retrieve-workflow': { count: 3, rewardSum: 2.1, rewardMean: 0.7, rewardVariance: 0.02, lastReward: 0.8, positiveCount: 3, negativeCount: 0, updatedAt: new Date().toISOString() },
        'a-silence': { count: 2, rewardSum: 0.5, rewardMean: 0.25, rewardVariance: 0.01, lastReward: 0.4, positiveCount: 2, negativeCount: 0, updatedAt: new Date().toISOString() },
      },
      updatedAt: new Date().toISOString(),
    });

    const report = maybeDistillAndStorePolicyV3(store, 'main', config);
    assert.equal(report.validation.ok, true);
    assert.equal(report.snapshot.status, 'active');
    assert.equal(report.snapshot.version, 'route-policy-v3');
    assert.ok(report.snapshot.rules.some((rule) => rule.route === 'retrieve_memory'));
    assert.ok(store.getActivePolicySnapshotV3('main'));

    const routeFn = new RouteFn({ config, store });
    const plan = routeFn.plan({
      agentId: 'main',
      sourceHook: 'before_prompt_build',
      latestUserMessageRedacted: 'Run the tests in this repo',
      toolObservations: [],
      recentInjections: [],
      metadata: {},
    });
    assert.equal(plan.route, 'retrieve_memory');
    assert.equal(plan.policySnapshotId, report.snapshot.id);
    assert.ok(plan.matchedPolicyRuleId);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('policy v3 validation rejects broad retrieval and harm-heavy candidate', () => {
  const config = normalizePluginConfig({ routeLearning: { policyV3: { maxSyncPlannerRate: 0.05, maxHarmRate: 0.2 } } });
  const report = validatePolicySnapshotV3({
    agentId: 'main',
    version: 'route-policy-v3',
    status: 'candidate',
    rules: [{
      id: 'bad-v3-rule',
      actionId: 'a1',
      match: {},
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      queries: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'allowed',
      confidence: 0.9,
      evidenceIds: ['e1'],
    }],
    actionPriors: {},
    globalBudgets: { maxSyncPlannerRate: 0.05, maxInjectedMemories: 8, maxInjectedChars: 2500, defaultGraphDepth: 1 },
    evalSummary: { frames: 3, pairExamples: 1, prototypes: 1, projectedSyncPlannerRate: 1, noisyActionRate: 0.05, harmRate: 0.4 },
    sourceFrameIds: ['f1'],
    sourcePrototypeIds: ['a1'],
  }, config);
  assert.equal(report.ok, false);
  assert.ok(report.errors.some((error) => error.includes('broad_retrieval_rule')));
  assert.ok(report.errors.some((error) => error.includes('sync_planner_rate_exceeds_budget')));
  assert.ok(report.errors.some((error) => error.includes('harm_rate_exceeds_gate')));
});

test('policy v2 validation rejects broad unsafe retrieval and sync budget overflow', () => {
  const config = normalizePluginConfig({ routeLearning: { policyV2: { maxSyncPlannerRate: 0.05 } } });
  const report = validatePolicySnapshotV2({
    agentId: 'main',
    version: 'route-policy-v2',
    status: 'candidate',
    rules: [{
      id: 'bad-rule',
      match: {},
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      queries: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'allowed',
      confidence: 0.9,
      evidenceIds: ['e1'],
    }],
    globalBudgets: { maxSyncPlannerRate: 0.05, maxInjectedMemories: 8, maxInjectedChars: 2500, defaultGraphDepth: 1 },
    evalSummary: { cases: 1, wins: 1, ties: 0, misses: 0, noisyInjections: 0, harms: 0, p95LatencyMs: 0 },
    exampleIds: ['e1'],
  }, config);
  assert.equal(report.ok, false);
  assert.ok(report.errors.some((error) => error.includes('broad_retrieval_rule')));
  assert.ok(report.errors.some((error) => error.includes('sync_planner_rate_exceeds_budget')));
});
