import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { MemoryStore, uuid, now, dbPathForAgent } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-test-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

// ── Memory nodes ──────────────────────────────────────────────────────────────

test('insert and get memory node', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const node = store.insertMemory({
      agentId: 'main', type: 'correction', content: 'Use pnpm instead of npm',
      scopeKind: 'repo', scopeKey: 'openclawbrain', normalizedKey: 'repo:openclawbrain:package-manager',
      tags: ['correction', 'package-manager'], importance: 0.8, freshness: 1.0, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
      sourceHook: 'before_prompt_build',
    });
    assert.ok(node.id);
    assert.equal(node.content, 'Use pnpm instead of npm');
    assert.equal(node.type, 'correction');
    assert.equal(node.scopeKind, 'repo');

    const fetched = store.getMemory(node.id);
    assert.ok(fetched);
    assert.equal(fetched.id, node.id);
    assert.equal(fetched.content, 'Use pnpm instead of npm');
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('find memory by normalized key', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    store.insertMemory({
      agentId: 'main', type: 'preference', content: 'User prefers file-by-file plans',
      scopeKind: 'global_user', normalizedKey: 'preference:plan-style',
      tags: ['style'], importance: 0.6, freshness: 1.0, confidence: 0.8,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    const found = store.findMemoryByNormalizedKey('main', 'preference:plan-style', 'global_user');
    assert.ok(found);
    assert.equal(found.content, 'User prefers file-by-file plans');
    assert.equal(found.type, 'preference');

    const notFound = store.findMemoryByNormalizedKey('main', 'nonexistent', 'global_user');
    assert.equal(notFound, null);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('update memory node', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const node = store.insertMemory({
      agentId: 'main', type: 'correction', content: 'Use pnpm',
      scopeKind: 'repo', normalizedKey: 'repo:pkg-mgr',
      tags: ['correction'], importance: 0.5, freshness: 1.0, confidence: 0.5,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    const updated = store.updateMemory(node.id, { importance: 0.9, confidence: 0.95 });
    assert.ok(updated);
    assert.equal(updated.importance, 0.9);
    assert.equal(updated.confidence, 0.95);
    assert.ok(updated.updatedAt);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('supersede and soft delete', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const old = store.insertMemory({
      agentId: 'main', type: 'correction', content: 'Use npm',
      scopeKind: 'repo', normalizedKey: 'repo:pkg-mgr',
      tags: [], importance: 0.5, freshness: 1.0, confidence: 0.5,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    const newer = store.insertMemory({
      agentId: 'main', type: 'correction', content: 'Use pnpm',
      scopeKind: 'repo', normalizedKey: 'repo:pkg-mgr-2',
      tags: [], importance: 0.8, freshness: 1.0, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    store.supersedeMemory(old.id, newer.id);
    const superseded = store.getMemory(old.id);
    assert.ok(superseded);
    assert.equal(superseded.supersededBy, newer.id);

    store.softDeleteMemory(newer.id);
    const deleted = store.getMemory(newer.id);
    assert.ok(deleted);
    assert.ok(deleted.deletedAt);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('list and count memories', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    store.insertMemory({
      agentId: 'main', type: 'correction', content: 'Use pnpm',
      scopeKind: 'repo', normalizedKey: 'repo:pkg-mgr',
      tags: [], importance: 0.8, freshness: 1.0, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    store.insertMemory({
      agentId: 'main', type: 'preference', content: 'File-by-file plans',
      scopeKind: 'global_user', normalizedKey: 'pref:plan',
      tags: [], importance: 0.6, freshness: 1.0, confidence: 0.8,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    store.insertMemory({
      agentId: 'main', type: 'workflow', content: 'Read PLAN.md before code',
      scopeKind: 'repo', normalizedKey: 'repo:workflow:plan-first',
      tags: [], importance: 0.5, freshness: 1.0, confidence: 0.7,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });

    const all = store.listMemories('main');
    assert.equal(all.length, 3);
    // ordered by importance DESC
    assert.equal(all[0].importance, 0.8);
    assert.equal(all[1].importance, 0.6);
    assert.equal(all[2].importance, 0.5);

    const corrections = store.listMemories('main', { type: 'correction' });
    assert.equal(corrections.length, 1);
    assert.equal(corrections[0].content, 'Use pnpm');

    assert.equal(store.countMemories('main'), 3);
    assert.equal(store.countMemories('main', 'correction'), 1);
    assert.equal(store.countMemories('main', 'context'), 0);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Memory edges ──────────────────────────────────────────────────────────────

test('insert and upsert edges', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const a = store.insertMemory({
      agentId: 'main', type: 'correction', content: 'Use pnpm',
      scopeKind: 'repo', normalizedKey: 'repo:pkg-mgr',
      tags: [], importance: 0.8, freshness: 1.0, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    const b = store.insertMemory({
      agentId: 'main', type: 'workflow', content: 'Read PLAN.md',
      scopeKind: 'repo', normalizedKey: 'repo:wf:plan',
      tags: [], importance: 0.5, freshness: 1.0, confidence: 0.7,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });

    const edge = store.insertEdge({
      agentId: 'main', fromId: a.id, toId: b.id,
      relation: 'related', weight: 0.5, evidenceCount: 1,
    });
    assert.ok(edge.id);
    assert.equal(edge.relation, 'related');

    // upsert bumps evidence count
    const upserted = store.upsertEdge('main', a.id, b.id, 'related');
    assert.equal(upserted.id, edge.id);
    assert.equal(upserted.evidenceCount, 2);
    assert.ok(upserted.weight > 0.5);

    const edges = store.getEdges(a.id);
    assert.equal(edges.length, 1);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Injection events ──────────────────────────────────────────────────────────

test('insert and resolve injection outcomes', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const node = store.insertMemory({
      agentId: 'main', type: 'correction', content: 'Use pnpm',
      scopeKind: 'repo', normalizedKey: 'repo:pkg-mgr',
      tags: [], importance: 0.8, freshness: 1.0, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });

    const inj = store.insertInjection({
      agentId: 'main', memoryId: node.id, query: 'install deps',
      rank: 1, score: 0.95,
    });
    assert.ok(inj.id);
    assert.equal(inj.outcome, 'pending');

    const pending = store.getPendingInjections('main');
    assert.equal(pending.length, 1);

    store.resolveInjectionOutcome(inj.id, 'helped');
    const after = store.getPendingInjections('main');
    assert.equal(after.length, 0);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Route decisions ───────────────────────────────────────────────────────────

test('insert and resolve route decisions', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const turnFrame = {
      summary: 'Install dependencies', userGoal: 'Setup repo',
      taskType: 'coding', activeObjects: [{ kind: 'repo', value: 'openclawbrain' }],
      impliedNeeds: ['package manager'], memoryQuestions: [], constraints: [],
      routeHints: { likelyNeedsCorrections: true, likelyNeedsPreferences: false, likelyNeedsWorkflow: false, likelyNeedsProjectContext: true },
    };
    const decision = store.insertRouteDecision({
      agentId: 'main', route: 'retrieve_memory', confidence: 0.85,
      latencyTier: 'cached_route', syncLlmUsed: false, fallbackUsed: false,
      turnFrame, retrievalPlan: { queries: ['package manager'], memoryTypes: ['correction'], requiredTags: [], excludedTags: [], graphDepth: 0, maxCandidates: 20 },
      injectionPlan: { maxItems: 3, maxChars: 500, preferredFormat: 'bullets' },
      selectedMemoryIds: ['mem-1'],
      omittedMemoryIds: [],
      reward: 0,
    });
    assert.ok(decision.id);
    assert.equal(decision.route, 'retrieve_memory');
    assert.equal(decision.outcome, 'pending');

    const recent = store.getRecentRouteDecisions('main');
    assert.equal(recent.length, 1);

    const unresolved = store.getUnresolvedRouteDecisions('main');
    assert.equal(unresolved.length, 1);

    store.resolveRouteDecision(decision.id, 'helpful_context', 1.0);
    const resolved = store.getRouteDecision(decision.id);
    assert.ok(resolved);
    assert.equal(resolved.outcome, 'helpful_context');
    assert.equal(resolved.reward, 1.0);

    const afterResolve = store.getUnresolvedRouteDecisions('main');
    assert.equal(afterResolve.length, 0);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Job queue ─────────────────────────────────────────────────────────────────

test('enqueue, claim, complete jobs', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    store.enqueueJob({
      agentId: 'main', kind: 'feedback_distillation', priority: 5,
      payload: { turnId: 't1' }, maxAttempts: 3, availableAt: now(),
    });
    store.enqueueJob({
      agentId: 'main', kind: 'route_learning', priority: 10,
      payload: { examples: [] }, maxAttempts: 3, availableAt: now(),
    });

    assert.equal(store.getJobQueueDepth('main'), 2);

    // higher priority first
    const job = store.claimNextJob();
    assert.ok(job);
    assert.equal(job.kind, 'route_learning');
    assert.equal(job.status, 'running');
    assert.equal(job.attempts, 1);

    assert.equal(store.getJobQueueDepth('main'), 1);

    store.completeJob(job.id);
    assert.equal(store.getJobQueueDepth('main'), 1);

    const job2 = store.claimNextJob();
    assert.ok(job2);
    assert.equal(job2.kind, 'feedback_distillation');
    store.completeJob(job2.id);
    assert.equal(store.getJobQueueDepth('main'), 0);

    // no more jobs
    const noJob = store.claimNextJob();
    assert.equal(noJob, null);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('job retry and dead letter', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    store.enqueueJob({
      agentId: 'main', kind: 'feedback_distillation', priority: 0,
      payload: {}, maxAttempts: 2, availableAt: now(),
    });

    const job = store.claimNextJob();
    assert.ok(job);
    store.failJob(job.id, 'timeout');
    assert.equal(store.getJobQueueDepth('main'), 1); // re-queued

    const job2 = store.claimNextJob();
    assert.ok(job2);
    assert.equal(job2.attempts, 2);
    store.failJob(job2.id, 'timeout again');
    assert.equal(store.getJobQueueDepth('main'), 0); // dead
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Proof events ──────────────────────────────────────────────────────────────

test('insert and read proof events', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const ev = store.insertProofEvent({
      agentId: 'main', kind: 'memory_captured',
      rawTranscriptStored: false,
      payload: { confidence: 0.9, feedbackType: 'correction', memoryCount: 1 },
    });
    assert.ok(ev.id);
    assert.equal(ev.kind, 'memory_captured');
    assert.equal(ev.rawTranscriptStored, false);

    const events = store.getProofEvents('main');
    assert.equal(events.length, 1);
    assert.equal(events[0].kind, 'memory_captured');
    assert.deepEqual(events[0].payload, { confidence: 0.9, feedbackType: 'correction', memoryCount: 1 });
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('proof events never store raw transcript', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const ev = store.insertProofEvent({
      agentId: 'main', kind: 'stay_silent',
      rawTranscriptStored: false,
      payload: { reasonCode: 'raw_transcript_upload_requested' },
    });
    assert.equal(ev.rawTranscriptStored, false);
    const events = store.getProofEvents('main');
    assert.equal(events[0].rawTranscriptStored, false);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Distillation run audit ────────────────────────────────────────────────────

test('insert distillation run audit row', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const run = store.insertDistillationRun({
      agentId: 'main', phase: 'agent_end_feedback',
      model: 'gpt-4o-mini', promptVersion: 'feedback-distiller-v1',
      inputHash: 'abc123', outputJson: '{"shouldStore":true}',
      validationStatus: 'valid',
    });
    assert.ok(run.id);
    assert.equal(run.phase, 'agent_end_feedback');
    assert.equal(run.model, 'gpt-4o-mini');
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Route examples and policy snapshots ───────────────────────────────────────

test('insert route example and policy snapshot', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const example = store.insertRouteExample({
      agentId: 'main',
      turnFrame: { summary: 'install deps', userGoal: 'setup', taskType: 'coding', activeObjects: [], impliedNeeds: [], memoryQuestions: [], constraints: [], routeHints: { likelyNeedsCorrections: true, likelyNeedsPreferences: false, likelyNeedsWorkflow: false, likelyNeedsProjectContext: false } },
      routeDecision: { route: 'retrieve_memory', confidence: 0.85 },
      outcome: 'helpful_context', reward: 1.0,
      lesson: 'Always retrieve package-manager corrections for dependency turns',
      tags: ['package-manager', 'correction'],
    });
    assert.ok(example.id);

    const examples = store.getRouteExamples('main');
    assert.equal(examples.length, 1);
    assert.equal(examples[0].reward, 1.0);

    const snapshot = store.insertPolicySnapshot({
      agentId: 'main',
      policyText: 'For dependency turns, retrieve package-manager corrections.',
      examples: [example.id],
      model: 'gpt-4o-mini', promptVersion: 'route-learner-v1',
      active: true,
    });
    assert.ok(snapshot.id);
    assert.equal(snapshot.active, true);

    const active = store.getActivePolicySnapshot('main');
    assert.ok(active);
    assert.equal(active.id, snapshot.id);

    // new active supersedes old
    const snapshot2 = store.insertPolicySnapshot({
      agentId: 'main', policyText: 'Updated policy', examples: [],
      active: true,
    });
    const active2 = store.getActivePolicySnapshot('main');
    assert.ok(active2);
    assert.equal(active2.id, snapshot2.id);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── Transactions ──────────────────────────────────────────────────────────────

test('transaction rollback on error', async () => {
  const root = await tempRoot();
  try {
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    try {
      store.transaction(() => {
        store.insertMemory({
          agentId: 'main', type: 'correction', content: 'Use pnpm',
          scopeKind: 'repo', normalizedKey: 'repo:pkg-mgr',
          tags: [], importance: 0.8, freshness: 1.0, confidence: 0.9,
          useCount: 0, usefulCount: 0, captureCount: 1,
        });
        throw new Error('rollback test');
      });
    } catch (e) {
      assert.equal(e.message, 'rollback test');
    }
    // rolled back — no memories
    assert.equal(store.countMemories('main'), 0);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('dbPathForAgent substitutes agentId', () => {
  assert.equal(
    dbPathForAgent('~/.openclawbrain/activation/${agentId}', 'main'),
    path.join(process.env.HOME, '.openclawbrain/activation/main', 'openclawbrain.db'),
  );
  assert.equal(
    dbPathForAgent('/tmp/brain', 'pelican'),
    '/tmp/brain/openclawbrain.db',
  );
});
