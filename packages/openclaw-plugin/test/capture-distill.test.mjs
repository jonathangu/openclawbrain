import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { CaptureOrchestrator, sanitizeToolEvent } from '../dist/capture.js';
import { normalizePluginConfig } from '../dist/config.js';
import { FeedbackDistiller } from '../dist/feedback-distiller.js';
import { FakeLlmClient } from '../dist/llm-client.js';
import { MemoryOperationApplier } from '../dist/memory-operations.js';
import { MemoryStore } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-capture-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

test('capture orchestrator builds redacted packets', () => {
  const config = normalizePluginConfig({ enabled: true, hooks: { allowConversationAccess: true }, maxContextChars: 500 });
  const packet = new CaptureOrchestrator().fromAgentEnd({
    agentId: 'main',
    turnId: 't1',
    userMessage: 'Email me at private@example.com and use pnpm instead of npm',
    assistantMessage: 'ok',
    ctx: { sessionId: 's1', sessionKey: 'k1', runId: 'r1', profile: 'main' },
  }, config);

  assert.equal(packet.agentId, 'main');
  assert.equal(packet.sourceHook, 'agent_end');
  assert.match(packet.latestUserMessageRedacted, /\[redacted-email\]/);
  assert.doesNotMatch(packet.latestUserMessageRedacted, /private@example.com/);
});

test('sanitizeToolEvent redacts args and results', () => {
  const config = normalizePluginConfig({ maxContextChars: 500 });
  const item = sanitizeToolEvent({
    toolName: 'exec',
    args: { token: 'sk-secret-secret-secret-secret' },
    result: { url: 'https://example.com' },
    durationMs: 123,
  }, config);
  assert.equal(item.toolName, 'exec');
  assert.equal(item.durationMs, 123);
  assert.match(item.argsSummary, /\[redacted-secret\]/);
  assert.match(item.resultSummary, /\[redacted-url\]/);
});

test('feedback distiller fallback returns no-op distillation', async () => {
  const config = normalizePluginConfig({ enabled: true, llm: { enabled: true, feedbackModel: 'fake' } });
  const distiller = new FeedbackDistiller({
    client: new FakeLlmClient({ handler: () => { throw new Error('boom'); } }),
    config,
  });
  const result = await distiller.distill({
    agentId: 'main',
    sourceHook: 'agent_end',
    latestUserMessageRedacted: 'use pnpm',
    toolObservations: [],
    recentInjections: [],
    metadata: {},
  });
  assert.equal(result.output.shouldStore, false);
  assert.equal(result.audit.fallbackUsed, true);
});

test('feedback distiller fallback does not store codewords', async () => {
  const config = normalizePluginConfig({ enabled: true, llm: { enabled: true, feedbackModel: 'fake' } });
  const distiller = new FeedbackDistiller({
    client: new FakeLlmClient({ handler: () => { throw new Error('boom'); } }),
    config,
  });
  const result = await distiller.distill({
    agentId: 'main',
    sourceHook: 'agent_end',
    latestUserMessageRedacted: 'Remember that the app has a special codeword: [redacted-secret]',
    toolObservations: [],
    recentInjections: [],
    metadata: {},
  });
  assert.equal(result.output.shouldStore, false);
  assert.equal(result.output.memoryCandidates.length, 0);
  assert.equal(result.output.audit.modelReasonCode, 'sensitive_codeword_not_stored');
});

test('memory operation applier creates and updates memories and resolves injections', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, capture: { minConfidence: 0.7 }, llm: { feedbackModel: 'fake' } });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const injectionMemory = store.insertMemory({
      agentId: 'main',
      type: 'correction',
      content: 'Use npm',
      scopeKind: 'repo',
      normalizedKey: 'repo:pkg-manager:old',
      tags: [],
      importance: 0.2,
      freshness: 1,
      confidence: 0.5,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    const injection = store.insertInjection({
      agentId: 'main',
      memoryId: injectionMemory.id,
      query: 'install deps',
      rank: 1,
      score: 0.5,
    });

    const applier = new MemoryOperationApplier({ store, config });
    const result = applier.applyDistillation({
      version: 1,
      shouldStore: true,
      confidence: 0.95,
      feedbackType: 'correction',
      memoryCandidates: [{
        type: 'correction',
        distilledText: 'Use pnpm instead of npm for this repo',
        subject: 'package manager',
        scope: { kind: 'repo', key: 'openclawbrain' },
        normalizedKey: 'repo:openclawbrain:package-manager',
        tags: ['package-manager'],
        confidence: 0.95,
        importanceHint: 0.9,
        retention: 'durable',
        contradictions: [],
      }],
      injectionFeedback: [{
        injectionId: injection.id,
        memoryId: injection.memoryId,
        outcome: 'user_corrected',
        confidence: 0.9,
        evidence: 'user corrected package manager choice',
      }],
      workflowCandidates: [],
      audit: { modelReasonCode: 'explicit_user_correction', storeRawTranscript: false, redactionNeeded: true },
    }, {
      agentId: 'main',
      sessionId: 's1',
      turnId: 't1',
      runId: 'r1',
      sourceHook: 'agent_end',
      latestUserMessageRedacted: 'use pnpm',
      toolObservations: [],
      recentInjections: [],
      metadata: { promptHash: 'h1' },
    });

    assert.equal(result.memoryIds.length, 1);
    assert.equal(result.resolvedInjections, 1);
    const stored = store.findMemoryByNormalizedKey('main', 'repo:openclawbrain:package-manager', 'repo', 'openclawbrain');
    assert.ok(stored);
    assert.match(stored.content, /pnpm/);
    assert.equal(store.getPendingInjections('main').length, 0);
    assert.equal(store.getProofEvents('main').length, 1);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
