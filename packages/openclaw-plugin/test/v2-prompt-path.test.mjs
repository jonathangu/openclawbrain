import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { handleTurnHook, normalizePluginConfig, readProofEvents, readStatus } from '../dist/index.js';
import { MemoryStore } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-v2-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

test('balanced mode before_prompt_build injects retrieved memory and records route/injection', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = store.insertMemory({
      agentId: 'main',
      type: 'correction',
      content: 'Use pnpm instead of npm for this repo',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:package-manager',
      tags: ['package-manager'],
      importance: 0.95,
      freshness: 1,
      confidence: 0.95,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    store.close();

    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      hooks: { allowPromptInjection: true },
      routing: { enabled: true, maxCandidateMemories: 20, maxInjectedMemories: 3, maxInjectedChars: 500 },
    });

    const result = await handleTurnHook({
      agentId: 'main',
      userMessage: 'Install dependencies for OpenClawBrain',
      ctx: { sessionId: 's1', sessionKey: 'k1', runId: 'r1' },
    }, config, {}, 'before_prompt_build');

    assert.ok(result.prependContext);
    assert.match(result.prependContext, /Use pnpm instead of npm/);

    const proofs = await readProofEvents({ activationRoot: root, agentId: 'main', limit: 20 });
    assert.ok(proofs.some((event) => event.kind === 'route_decision' || event.kind === 'llm_route_decision'));

    const status = await readStatus({ activationRoot: root, agentId: 'main' });
    assert.ok(status);
    assert.equal(status.lastDecisionKind, 'memory_injected');

    const store2 = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const routes = store2.getRecentRouteDecisions('main');
    assert.equal(routes.length, 1);
    assert.equal(routes[0].selectedMemoryIds[0], memory.id);
    const pendingInjections = store2.getPendingInjections('main');
    assert.equal(pendingInjections.length, 1);
    assert.equal(pendingInjections[0].memoryId, memory.id);
    store2.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
