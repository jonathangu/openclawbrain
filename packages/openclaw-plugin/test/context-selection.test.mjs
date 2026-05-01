import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizePluginConfig } from '../dist/config.js';
import { ContextSelector } from '../dist/context-selector.js';

const config = normalizePluginConfig({ enabled: true, mode: 'balanced', routing: { maxInjectedMemories: 3, maxInjectedChars: 500, minRouteConfidence: 0.7 } });

const packet = {
  agentId: 'main',
  sourceHook: 'before_prompt_build',
  latestUserMessage: 'Install dependencies for OpenClawBrain and use the right package manager',
  redactedLatestUserMessage: 'Install dependencies for OpenClawBrain and use the right package manager',
  toolObservations: [],
  recentInjections: [],
  metadata: {},
};

const plan = {
  route: 'retrieve_memory',
  confidence: 0.8,
  turnFrame: {
    summary: 'Install dependencies',
    userGoal: 'Set up repo',
    taskType: 'coding',
    activeObjects: [],
    impliedNeeds: [],
    memoryQuestions: [],
    constraints: [],
    routeHints: { likelyNeedsCorrections: true, likelyNeedsPreferences: false, likelyNeedsWorkflow: true, likelyNeedsProjectContext: true },
  },
  retrievalPlan: { queries: ['install dependencies'], memoryTypes: ['correction', 'workflow'], requiredTags: [], excludedTags: [], graphDepth: 0, maxCandidates: 20 },
  injectionPlan: { maxItems: 3, maxChars: 500, preferredFormat: 'bullets' },
  shouldRetrieve: true,
  enqueueCapture: false,
  latencyReason: 'test',
};

test('context selector prioritizes corrections and formats prompt block', () => {
  const selector = new ContextSelector(config);
  const selection = selector.select({
    packet,
    plan,
    candidates: [
      {
        id: 'm1', agentId: 'main', type: 'correction', content: 'Use pnpm instead of npm for this repo',
        scopeKind: 'repo', scopeKey: 'openclawbrain', normalizedKey: 'repo:pkg', tags: [],
        importance: 0.9, freshness: 1, confidence: 0.95, useCount: 0, usefulCount: 0, captureCount: 1,
        createdAt: '', updatedAt: '', lastSeenAt: '',
      },
      {
        id: 'm2', agentId: 'main', type: 'workflow', content: 'Run tests after install',
        scopeKind: 'repo', scopeKey: 'openclawbrain', normalizedKey: 'repo:wf', tags: [],
        importance: 0.5, freshness: 1, confidence: 0.8, useCount: 0, usefulCount: 0, captureCount: 1,
        createdAt: '', updatedAt: '', lastSeenAt: '',
      },
    ],
  });

  assert.equal(selection.shouldInject, true);
  assert.equal(selection.selectedMemoryIds[0], 'm1');
  assert.match(selection.distilledContext, /Must follow: Use pnpm instead of npm/);
  assert.match(selection.distilledContext, /Workflow: Run tests after install/);
});
