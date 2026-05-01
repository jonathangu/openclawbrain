import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { FakeLlmClient, MemoryPlanner, normalizePluginConfig, RouteFn } from '../dist/index.js';
import { MemoryStore } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-memory-planner-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

test('memory planner uses LLM-selected memory ids for ambiguous retrieval', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      llm: { enabled: true, plannerModel: 'fake-planner' },
      routing: { enabled: true, maxCandidateMemories: 10, maxInjectedMemories: 2, maxInjectedChars: 400 },
    });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const chosen = store.insertMemory({
      agentId: 'main',
      type: 'preference',
      content: 'For implementation plans, give concrete file-by-file details.',
      scopeKind: 'agent',
      normalizedKey: 'user:style:file-by-file',
      tags: ['style', 'planning'],
      importance: 0.95,
      freshness: 1,
      confidence: 0.94,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    store.insertMemory({
      agentId: 'main',
      type: 'workflow',
      content: 'Inspect package.json before editing build steps.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:workflow:package-json',
      tags: ['workflow'],
      importance: 0.7,
      freshness: 1,
      confidence: 0.8,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });

    const planner = new MemoryPlanner({
      config,
      routeFn: new RouteFn({ config, store }),
      store,
      client: new FakeLlmClient({
        responses: [{
          route: 'retrieve_memory',
          confidence: 0.93,
          shouldRetrieve: true,
          selectedMemoryIds: [chosen.id],
          likelyFeedbackType: 'none',
        }],
      }),
    });

    const result = await planner.run({
      agentId: 'main',
      sessionId: 's1',
      sessionKey: 'k1',
      turnId: 't1',
      runId: 'r1',
      sourceHook: 'before_prompt_build',
      latestUserMessage: 'Send me the final implementation plan.',
      redactedLatestUserMessage: 'Send me the final implementation plan.',
      recentAssistantMessage: '',
      toolObservations: [],
      recentInjections: [],
      metadata: {},
    });

    assert.equal(result.routePlan.latencyReason, 'llm memory planner');
    assert.ok(result.contextSelection.shouldInject);
    assert.equal(result.contextSelection.selectedMemoryIds[0], chosen.id);
    assert.match(result.contextSelection.distilledContext, /file-by-file details/);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
