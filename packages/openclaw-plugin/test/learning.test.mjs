import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { BackgroundLearner, normalizePluginConfig } from '../dist/index.js';
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
      latestUserMessage: 'Install dependencies',
      redactedLatestUserMessage: 'Install dependencies',
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
