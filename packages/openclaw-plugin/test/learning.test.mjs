import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { BackgroundLearner, RouteFn, RouteTeacher, buildRouteGraphSnapshot, maybeDistillAndStorePolicyV2, maybeDistillAndStorePolicyV3, normalizePluginConfig, rankActionPrototypesV3, scorePolicySnapshotV3, validatePolicySnapshotV2, validatePolicySnapshotV3 } from '../dist/index.js';
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

test('policy v3 distiller merges duplicate prototypes and records replay/calibration summaries', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      routeLearning: {
        enabled: true,
        policyV3: {
          enabled: true,
          minFrames: 2,
          shadowBeforeActivate: false,
          maxHarmRate: 0.6,
          maxRules: 12,
          maxRulesPerRoute: 4,
        },
      },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });

    const frames = [
      store.insertRouteFrameV3({
        agentId: 'main',
        routeDecisionId: 'dup-1',
        redactedTurnSummary: 'Run repo tests before commit',
        taskType: 'coding',
        turnSignals: ['test', 'workflow', 'repo'],
        projectHint: 'openclawbrain',
        repoHint: 'openclawbrain',
        toolHints: ['exec'],
        routeHintFlags: ['needs_workflow', 'needs_project_context'],
        chosenActionId: 'dup-a',
        chosenRoute: 'retrieve_memory',
        chosenMemoryTypes: ['workflow'],
        chosenGraphDepth: 1,
        chosenSyncPlanner: 'no',
        outcome: 'tool_success',
        reward: 0.8,
        rewardComponents: { retrievalHelpGain: 0.8, noisyInjectionPenalty: 0 },
        payloadHash: 'dup-p1',
      }),
      store.insertRouteFrameV3({
        agentId: 'main',
        routeDecisionId: 'dup-2',
        redactedTurnSummary: 'Run tests in this repo',
        taskType: 'coding',
        turnSignals: ['tests', 'workflow', 'repo'],
        projectHint: 'openclawbrain',
        repoHint: 'openclawbrain',
        toolHints: ['exec'],
        routeHintFlags: ['needs_workflow', 'needs_project_context'],
        chosenActionId: 'dup-b',
        chosenRoute: 'retrieve_memory',
        chosenMemoryTypes: ['workflow'],
        chosenGraphDepth: 1,
        chosenSyncPlanner: 'no',
        outcome: 'tool_success',
        reward: 0.75,
        rewardComponents: { retrievalHelpGain: 0.75, noisyInjectionPenalty: 0 },
        payloadHash: 'dup-p2',
      }),
      store.insertRouteFrameV3({
        agentId: 'main',
        routeDecisionId: 'dup-3',
        redactedTurnSummary: 'Thanks ok',
        taskType: 'other',
        turnSignals: ['thanks', 'ok'],
        toolHints: [],
        routeHintFlags: [],
        chosenActionId: 'dup-silence',
        chosenRoute: 'no_memory',
        chosenMemoryTypes: [],
        chosenGraphDepth: 0,
        chosenSyncPlanner: 'no',
        outcome: 'no_signal',
        reward: 0.35,
        rewardComponents: { abstainGain: 0.2, noisyInjectionPenalty: 0 },
        payloadHash: 'dup-p3',
      }),
    ];

    store.upsertRouteActionPrototypeV3({
      id: 'dup-a',
      agentId: 'main',
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      queryTemplateFamily: ['test workflow'],
      sparseSignature: ['coding', 'test', 'workflow', 'openclawbrain'],
      denseEmbedding: [1, 0.4, 0.2],
      supportPrior: 2,
      harmPrior: 0,
      status: 'active',
      provenance: 'learned',
      sourceExampleIds: [frames[0].id],
    });
    store.upsertRouteActionPrototypeV3({
      id: 'dup-b',
      agentId: 'main',
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      queryTemplateFamily: ['repo test workflow'],
      sparseSignature: ['coding', 'tests', 'workflow', 'openclawbrain'],
      denseEmbedding: [1, 0.38, 0.2],
      supportPrior: 2,
      harmPrior: 0,
      status: 'active',
      provenance: 'distilled',
      sourceExampleIds: [frames[1].id],
    });
    store.upsertRouteActionPrototypeV3({
      id: 'dup-silence',
      agentId: 'main',
      route: 'no_memory',
      memoryTypes: [],
      graphDepth: 0,
      syncPlanner: 'no',
      queryTemplateFamily: [],
      sparseSignature: ['other', 'thanks', 'ok'],
      denseEmbedding: [0.1, 0.1, 0],
      supportPrior: 1,
      harmPrior: 0,
      status: 'active',
      provenance: 'learned',
      sourceExampleIds: [frames[2].id],
    });

    store.insertRoutePairExampleV3({
      agentId: 'main',
      frameId: frames[0].id,
      positiveActionId: 'dup-a',
      negativeActionId: 'dup-silence',
      labelSource: 'teacher',
      marginWeight: 0.8,
      evidenceIds: [frames[0].id],
    });
    store.insertRoutePairExampleV3({
      agentId: 'main',
      frameId: frames[1].id,
      positiveActionId: 'dup-b',
      negativeActionId: 'dup-silence',
      labelSource: 'teacher',
      marginWeight: 0.8,
      evidenceIds: [frames[1].id],
    });

    const report = maybeDistillAndStorePolicyV3(store, 'main', config);
    assert.equal(report.validation.ok, true);
    assert.ok(report.snapshot.calibration);
    assert.ok(report.snapshot.evalSummary?.replay);
    assert.ok(report.snapshot.evalSummary?.compactness);
    assert.ok(report.snapshot.rules.length < 3);
    assert.ok(report.snapshot.evalSummary.replay.frames >= 1);
    assert.ok(report.snapshot.rules.every((rule) => rule.family));
    assert.ok(report.snapshot.rules.every((rule) => rule.canonicalActionKey));
    assert.ok(report.snapshot.rules.every((rule) => typeof rule.matchSpecificityScore === 'number'));
    assert.ok(report.snapshot.evalSummary?.thresholds?.byFamily);
    assert.ok(report.snapshot.evalSummary?.activationSummary);
    assert.ok(report.snapshot.evalSummary?.rollbackRecommendation);
    assert.ok(store.listRouteCalibrationExamplesV3('main', 20, report.snapshot.id).length >= 1);
    assert.ok(store.listRouteEvalCasesV3('main', 20, report.snapshot.id).length >= 1);
    assert.ok(store.listRouteEvalCaseLabelsV3('main', 20).length >= 1);
    assert.ok(store.listRoutePolicyCandidateReportsV3('main', 10).some((candidate) => candidate.snapshotId === report.snapshot.id));
    assert.ok(store.listRouteActionFamilyStatsV3('main', 10).length >= 2);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('policy v3 manual-review update mode keeps candidate snapshots shadow-only', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      routeLearning: {
        enabled: true,
        policyV3: {
          enabled: true,
          updateMode: 'manual_review_required',
          minFrames: 2,
          maxHarmRate: 0.6,
        },
      },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });

    const frameA = store.insertRouteFrameV3({
      agentId: 'main',
      routeDecisionId: 'mr-1',
      redactedTurnSummary: 'Run project tests',
      taskType: 'coding',
      turnSignals: ['test', 'repo', 'workflow'],
      projectHint: 'openclawbrain',
      repoHint: 'openclawbrain',
      toolHints: ['exec'],
      routeHintFlags: ['needs_workflow'],
      chosenActionId: 'mr-action',
      chosenRoute: 'retrieve_memory',
      chosenMemoryTypes: ['workflow'],
      chosenGraphDepth: 1,
      chosenSyncPlanner: 'no',
      outcome: 'tool_success',
      reward: 0.7,
      rewardComponents: { retrievalHelpGain: 0.7 },
      payloadHash: 'mr-p1',
    });
    const frameB = store.insertRouteFrameV3({
      agentId: 'main',
      routeDecisionId: 'mr-2',
      redactedTurnSummary: 'Run tests in repo',
      taskType: 'coding',
      turnSignals: ['tests', 'repo', 'workflow'],
      projectHint: 'openclawbrain',
      repoHint: 'openclawbrain',
      toolHints: ['exec'],
      routeHintFlags: ['needs_workflow'],
      chosenActionId: 'mr-action',
      chosenRoute: 'retrieve_memory',
      chosenMemoryTypes: ['workflow'],
      chosenGraphDepth: 1,
      chosenSyncPlanner: 'no',
      outcome: 'tool_success',
      reward: 0.72,
      rewardComponents: { retrievalHelpGain: 0.72 },
      payloadHash: 'mr-p2',
    });
    store.upsertRouteActionPrototypeV3({
      id: 'mr-action',
      agentId: 'main',
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      queryTemplateFamily: ['repo test workflow'],
      sparseSignature: ['coding', 'test', 'repo', 'workflow'],
      denseEmbedding: [1, 0.3, 0.1],
      supportPrior: 2,
      harmPrior: 0,
      status: 'active',
      provenance: 'learned',
      sourceExampleIds: [frameA.id, frameB.id],
    });
    store.upsertRouteActionPrototypeV3({
      id: 'mr-silence',
      agentId: 'main',
      route: 'no_memory',
      memoryTypes: [],
      graphDepth: 0,
      syncPlanner: 'no',
      queryTemplateFamily: [],
      sparseSignature: ['other', 'silence'],
      denseEmbedding: [0.1, 0, 0],
      supportPrior: 1,
      harmPrior: 0,
      status: 'active',
      provenance: 'learned',
      sourceExampleIds: [frameA.id],
    });
    store.insertRoutePairExampleV3({
      agentId: 'main',
      frameId: frameA.id,
      positiveActionId: 'mr-action',
      negativeActionId: 'mr-silence',
      labelSource: 'teacher',
      marginWeight: 0.6,
      evidenceIds: [frameA.id],
    });

    const report = maybeDistillAndStorePolicyV3(store, 'main', config);
    assert.equal(report.validation.ok, true);
    assert.equal(report.snapshot.status, 'shadow');
    assert.equal(report.snapshot.evalSummary.activationStatusReason, 'manual_review_required');
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('policy v3 keeps cold-start prototypes out of active distillation until enough evidence accumulates', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      routeLearning: {
        enabled: true,
        policyV3: {
          enabled: true,
          minFrames: 2,
          coldStartMinSamples: 3,
          maxHarmRate: 0.6,
        },
      },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });

    const frame1 = store.insertRouteFrameV3({
      agentId: 'main',
      routeDecisionId: 'cs-1',
      redactedTurnSummary: 'Run repo tests',
      taskType: 'coding',
      turnSignals: ['test', 'repo', 'workflow'],
      projectHint: 'openclawbrain',
      repoHint: 'openclawbrain',
      toolHints: ['exec'],
      routeHintFlags: ['needs_workflow'],
      chosenActionId: 'cs-action',
      chosenRoute: 'retrieve_memory',
      chosenMemoryTypes: ['workflow'],
      chosenGraphDepth: 1,
      chosenSyncPlanner: 'no',
      outcome: 'tool_success',
      reward: 0.7,
      rewardComponents: { retrievalHelpGain: 0.7 },
      payloadHash: 'cs-p1',
    });
    const frame2 = store.insertRouteFrameV3({
      agentId: 'main',
      routeDecisionId: 'cs-2',
      redactedTurnSummary: 'Run tests before commit',
      taskType: 'coding',
      turnSignals: ['tests', 'repo', 'workflow'],
      projectHint: 'openclawbrain',
      repoHint: 'openclawbrain',
      toolHints: ['exec'],
      routeHintFlags: ['needs_workflow'],
      chosenActionId: 'cs-action',
      chosenRoute: 'retrieve_memory',
      chosenMemoryTypes: ['workflow'],
      chosenGraphDepth: 1,
      chosenSyncPlanner: 'no',
      outcome: 'tool_success',
      reward: 0.68,
      rewardComponents: { retrievalHelpGain: 0.68 },
      payloadHash: 'cs-p2',
    });

    store.upsertRouteActionPrototypeV3({
      id: 'cs-action',
      agentId: 'main',
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      queryTemplateFamily: ['repo test workflow'],
      sparseSignature: ['coding', 'test', 'repo', 'workflow'],
      denseEmbedding: [1, 0.2, 0.1],
      supportPrior: 2,
      harmPrior: 0,
      status: 'cold_start',
      provenance: 'learned',
      sourceExampleIds: [frame1.id, frame2.id],
    });

    const before = maybeDistillAndStorePolicyV3(store, 'main', config);
    assert.equal(before.snapshot, undefined);
    assert.equal(store.getActivePolicySnapshotV3('main'), null);

    store.upsertRouteActionPrototypeV3({
      id: 'cs-action',
      agentId: 'main',
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      queryTemplateFamily: ['repo test workflow'],
      sparseSignature: ['coding', 'test', 'repo', 'workflow'],
      denseEmbedding: [1, 0.2, 0.1],
      supportPrior: 1,
      harmPrior: 0,
      status: 'cold_start',
      provenance: 'learned',
      sourceExampleIds: [frame1.id],
    });

    const after = maybeDistillAndStorePolicyV3(store, 'main', config);
    assert.equal(after.validation.ok, true);
    assert.ok(after.snapshot);
    assert.equal(store.getRouteActionPrototypeV3('cs-action').status, 'active');
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('route shadow decisions are stored and finalized after outcome resolution', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, mode: 'balanced', activationRoot: root });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    store.insertPolicySnapshotV3({
      agentId: 'main',
      version: 'route-policy-v3',
      status: 'shadow',
      rules: [{
        id: 'shadow-rule',
        actionId: 'shadow-action',
        match: { taskType: 'coding', turnSignals: ['test', 'repo'] },
        route: 'retrieve_memory',
        memoryTypes: ['workflow'],
        queries: ['test workflow'],
        graphDepth: 1,
        syncPlanner: 'no',
        confidence: 0.74,
        evidenceIds: ['e1'],
      }],
      actionPriors: {
        'shadow-action': { support: 1, harm: 0, pairWinRate: 0.6, banditMeanReward: 0.2, banditCount: 1 },
      },
      globalBudgets: { maxSyncPlannerRate: 0.1, maxInjectedMemories: 8, maxInjectedChars: 2500, defaultGraphDepth: 1, minCalibratedConfidence: 0.4, abstainMargin: 0.05 },
      evalSummary: { frames: 1, pairExamples: 0, prototypes: 1, projectedSyncPlannerRate: 0, noisyActionRate: 0, harmRate: 0 },
      calibration: {
        method: 'histogram_binning_v1',
        holdoutFrames: 1,
        comparableFrames: 1,
        globalThreshold: 0.4,
        abstainMargin: 0.05,
        globalBuckets: [{ minScore: 0.5, maxScore: 1, successRate: 0.8, count: 1 }],
        routeThresholds: { retrieve_memory: 0.4 },
        routeBuckets: { retrieve_memory: [{ minScore: 0.5, maxScore: 1, successRate: 0.8, count: 1 }] },
      },
      sourceFrameIds: [],
      sourcePrototypeIds: ['shadow-action'],
    });

    const decision = store.insertRouteDecision({
      agentId: 'main',
      route: 'retrieve_memory',
      confidence: 0.8,
      latencyTier: 'no_sync',
      syncLlmUsed: false,
      fallbackUsed: false,
      turnFrame: {
        summary: 'Run repo tests',
        userGoal: 'Run repo tests',
        taskType: 'coding',
        activeObjects: [{ kind: 'repo', value: 'openclawbrain' }],
        impliedNeeds: ['test workflow'],
        memoryQuestions: [],
        constraints: [],
        routeHints: {
          likelyNeedsCorrections: false,
          likelyNeedsPreferences: false,
          likelyNeedsWorkflow: true,
          likelyNeedsProjectContext: true,
        },
      },
      retrievalPlan: { queries: ['test workflow'], memoryTypes: ['workflow'], requiredTags: [], excludedTags: [], graphDepth: 1, maxCandidates: 8 },
      injectionPlan: { maxItems: 4, maxChars: 800, preferredFormat: 'bullets' },
      selectedMemoryIds: [],
      omittedMemoryIds: [],
      reward: 0,
    });

    const shadowSnapshot = store.listPolicySnapshotsV3('main', 5)[0];
    const match = scorePolicySnapshotV3(shadowSnapshot, decision.turnFrame, 'Run repo tests', { requireActive: false });
    store.insertRouteShadowDecisionV3({
      agentId: 'main',
      routeDecisionId: decision.id,
      snapshotId: shadowSnapshot.id,
      snapshotStatus: shadowSnapshot.status,
      proposedRoute: match.rule.route,
      proposedActionId: match.rule.actionId,
      proposedRuleId: match.rule.id,
      rawScore: match.rawScore,
      calibratedScore: match.calibratedScore,
      threshold: match.threshold,
      abstained: match.abstained,
      routingMode: 'workflow_exact',
      reasonCode: match.reasonCode,
    });

    store.finalizeRouteShadowDecisionsV3(decision.id, 'retrieve_memory', 0.8);
    const shadows = store.listRouteShadowDecisionsV3('main', 10, decision.id);
    assert.equal(shadows.length, 1);
    assert.equal(shadows[0].matchedObservedRoute, true);
    assert.equal(shadows[0].reward, 0.8);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('policy v3 calibration can abstain on weak matches', () => {
  const snapshot = {
    id: 'snap-weak',
    agentId: 'main',
    version: 'route-policy-v3',
    status: 'active',
    rules: [{
      id: 'weak-rule',
      actionId: 'weak-action',
      match: { taskType: 'coding', turnSignals: ['test', 'repo'] },
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      queries: ['test workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      confidence: 0.72,
      evidenceIds: ['e1'],
      priors: { support: 1, harm: 0, pairWinRate: 0.5, banditMeanReward: 0.1, banditCount: 1 },
    }],
    actionPriors: {
      'weak-action': { support: 1, harm: 0, pairWinRate: 0.5, banditMeanReward: 0.1, banditCount: 1 },
    },
    globalBudgets: { maxSyncPlannerRate: 0.1, maxInjectedMemories: 8, maxInjectedChars: 2500, defaultGraphDepth: 1, minCalibratedConfidence: 0.85, abstainMargin: 0.05 },
    evalSummary: { frames: 3, pairExamples: 1, prototypes: 1, projectedSyncPlannerRate: 0, noisyActionRate: 0, harmRate: 0 },
    calibration: {
      method: 'histogram_binning_v1',
      holdoutFrames: 3,
      comparableFrames: 3,
      globalThreshold: 0.85,
      abstainMargin: 0.05,
      globalBuckets: [{ minScore: 0.6, maxScore: 0.8, successRate: 0.45, count: 3 }],
      routeThresholds: { retrieve_memory: 0.85 },
      routeBuckets: { retrieve_memory: [{ minScore: 0.6, maxScore: 0.8, successRate: 0.45, count: 3 }] },
    },
    sourceFrameIds: ['f1'],
    sourcePrototypeIds: ['weak-action'],
    createdAt: new Date().toISOString(),
  };

  const match = scorePolicySnapshotV3(snapshot, {
    summary: 'Run repo tests',
    userGoal: 'Run repo tests',
    taskType: 'coding',
    activeObjects: [{ kind: 'repo', value: 'openclawbrain' }],
    impliedNeeds: ['test workflow'],
    memoryQuestions: [],
    constraints: [],
    routeHints: {
      likelyNeedsCorrections: false,
      likelyNeedsPreferences: false,
      likelyNeedsWorkflow: true,
      likelyNeedsProjectContext: true,
    },
  }, 'Run repo tests');

  assert.equal(match.matched, false);
  assert.equal(match.abstained, true);
  assert.ok(String(match.reasonCode).startsWith('policy_v3_abstain:'));
});

test('policy v3 family thresholding is stricter for correction rules', () => {
  const snapshot = {
    id: 'snap-correction-floor',
    agentId: 'main',
    version: 'route-policy-v3',
    status: 'active',
    rules: [{
      id: 'corr-rule',
      actionId: 'corr-action',
      family: 'correction',
      match: { taskType: 'coding', turnSignals: ['correction', 'repo'] },
      route: 'high_confidence_correction_only',
      memoryTypes: ['correction'],
      queries: ['package correction'],
      graphDepth: 0,
      syncPlanner: 'no',
      confidence: 0.42,
      rawConfidence: 0.42,
      evidenceIds: ['e1'],
      priors: { support: 1, harm: 0, pairWinRate: 0.5, banditMeanReward: 0, banditCount: 1 },
    }],
    actionPriors: {
      'corr-action': { support: 1, harm: 0, pairWinRate: 0.5, banditMeanReward: 0, banditCount: 1 },
    },
    globalBudgets: { maxSyncPlannerRate: 0.1, maxInjectedMemories: 8, maxInjectedChars: 2500, defaultGraphDepth: 1, minCalibratedConfidence: 0.62, abstainMargin: 0.05 },
    evalSummary: { frames: 3, pairExamples: 1, prototypes: 1, projectedSyncPlannerRate: 0, noisyActionRate: 0, harmRate: 0 },
    calibration: {
      method: 'histogram_binning_v1',
      holdoutFrames: 3,
      comparableFrames: 3,
      globalThreshold: 0.55,
      abstainMargin: 0.05,
      globalBuckets: [{ minScore: 0.5, maxScore: 0.8, successRate: 0.7, count: 3 }],
      routeThresholds: { high_confidence_correction_only: 0.55 },
      routeBuckets: { high_confidence_correction_only: [{ minScore: 0.5, maxScore: 0.8, successRate: 0.7, count: 3 }] },
    },
    sourceFrameIds: ['f1'],
    sourcePrototypeIds: ['corr-action'],
    createdAt: new Date().toISOString(),
  };

  const match = scorePolicySnapshotV3(snapshot, {
    summary: 'Correction: fix the package manager issue in this repo',
    userGoal: 'Fix the package manager issue',
    taskType: 'coding',
    activeObjects: [{ kind: 'repo', value: 'openclawbrain' }],
    impliedNeeds: ['correction'],
    memoryQuestions: [],
    constraints: [],
    routeHints: {
      likelyNeedsCorrections: true,
      likelyNeedsPreferences: false,
      likelyNeedsWorkflow: false,
      likelyNeedsProjectContext: true,
    },
  }, 'Correction: fix the package manager issue in this repo');

  assert.equal(match.matched, false);
  assert.equal(match.abstained, true);
  assert.ok(match.threshold >= 0.7);
});

test('policy v3 hybrid ranking favors correction prototypes for correction turns', () => {
  const ranked = rankActionPrototypesV3({
    taskType: 'coding',
    turnSignals: ['correction', 'pnpm', 'repo'],
    projectHint: 'openclawbrain',
    repoHint: 'openclawbrain',
    toolHints: ['exec'],
    routeHintFlags: ['needs_correction', 'needs_project_context'],
    redactedTurnSummary: 'Fix package manager correction in this repo',
  }, [
    {
      id: 'corr',
      agentId: 'main',
      route: 'high_confidence_correction_only',
      memoryTypes: ['correction'],
      graphDepth: 0,
      syncPlanner: 'no',
      queryTemplateFamily: ['package manager correction'],
      sparseSignature: ['coding', 'correction', 'pnpm', 'repo'],
      denseEmbedding: [1, 0.2, 0.1],
      supportPrior: 2,
      harmPrior: 0,
      status: 'active',
      provenance: 'learned',
      sourceExampleIds: ['e1'],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    {
      id: 'workflow',
      agentId: 'main',
      route: 'retrieve_memory',
      memoryTypes: ['workflow'],
      graphDepth: 1,
      syncPlanner: 'no',
      queryTemplateFamily: ['test workflow'],
      sparseSignature: ['coding', 'workflow', 'test', 'repo'],
      denseEmbedding: [0.7, 0.3, 0.1],
      supportPrior: 2,
      harmPrior: 0,
      status: 'active',
      provenance: 'learned',
      sourceExampleIds: ['e2'],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ], null);

  assert.equal(ranked[0].prototype.id, 'corr');
});

test('policy v3 teacher ingestion does not invert likely_missed counterfactuals into no_memory winners', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      routing: { minRouteConfidence: 0.55 },
      routeLearning: {
        enabled: true,
        teacher: { enabled: true, maxRunsPerCycle: 10, minResolvedRewardMagnitude: 0 },
        policyV2: { enabled: true, minExamples: 1, shadowBeforeActivate: false, maxNoisyInjectionRate: 0.6 },
        policyV3: { enabled: true, minFrames: 1, shadowBeforeActivate: false, maxHarmRate: 0.4, maxSyncPlannerRate: 0.7 },
      },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const correction = store.insertMemory({
      agentId: 'main',
      type: 'correction',
      content: 'Use pnpm instead of npm.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:correction:pnpm',
      tags: ['correction'],
      importance: 0.9,
      freshness: 1,
      confidence: 0.95,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });

    const helpfulCorrection = store.insertRouteDecision({
      agentId: 'main',
      sessionId: 's-corr',
      turnId: 't-corr',
      runId: 'r-corr',
      route: 'high_confidence_correction_only',
      confidence: 0.92,
      latencyTier: 'sync_memory_planner',
      syncLlmUsed: true,
      fallbackUsed: false,
      turnFrame: {
        summary: 'Fix install docs package manager',
        userGoal: 'Fix install docs package manager',
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
        queries: ['package manager correction'],
        memoryTypes: ['correction'],
        requiredTags: [],
        excludedTags: [],
        graphDepth: 0,
        maxCandidates: 10,
      },
      injectionPlan: { maxItems: 1, maxChars: 120, preferredFormat: 'rules' },
      selectedMemoryIds: [correction.id],
      omittedMemoryIds: [],
      reward: 0,
    });
    store.resolveRouteDecision(helpfulCorrection.id, 'tool_success', 0.95);
    buildRouteGraphSnapshot(store, 'main', helpfulCorrection.id, ['package manager correction'], [correction], 0);

    const teacher = new RouteTeacher({ store, config });
    await teacher.run('main');

    const activeV3 = store.getActivePolicySnapshotV3('main');
    assert.ok(activeV3);

    const routeFn = new RouteFn({ config, store });
    const plan = routeFn.plan({
      agentId: 'main',
      sessionId: 'probe',
      sessionKey: 'probe',
      turnId: 'probe-turn',
      runId: 'probe-run',
      sourceHook: 'before_prompt_build',
      latestUserMessageRedacted: 'Fix install docs package manager',
      recentAssistantMessage: '',
      toolObservations: [],
      recentInjections: [],
      metadata: {},
    });

    assert.equal(plan.route, 'high_confidence_correction_only');
    assert.equal(plan.shouldRetrieve, true);
    assert.ok(plan.retrievalPlan.memoryTypes.includes('correction'));
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
