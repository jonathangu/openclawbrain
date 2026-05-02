import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import plugin, { auditPayload, buildMemoryCorpusSupplement, explainLastPayload, graphPayload, learnPayload, normalizePluginConfig, searchPayload } from '../dist/index.js';
import { MemoryStore } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-search-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

test('plugin registers additive memory supplements and inspectable routes', () => {
  const routes = [];
  const promptSupplements = [];
  const corpusSupplements = [];
  plugin.register({
    pluginConfig: { enabled: true },
    on() {},
    registerService() {},
    registerHttpRoute(route) { routes.push(route); },
    registerMemoryPromptSupplement(builder) { promptSupplements.push(builder); },
    registerMemoryCorpusSupplement(supplement) { corpusSupplements.push(supplement); },
    logger: { debug() {}, warn() {} },
  });

  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/graph'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/learn'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/search'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/audit'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/explain-last'));
  assert.equal(promptSupplements.length, 1);
  assert.equal(corpusSupplements.length, 1);
  assert.ok(promptSupplements[0]({ availableTools: new Set(), citationsMode: 'auto' }).length >= 1);
});

test('search, graph, learn payloads and corpus supplement expose stored memories', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, mode: 'balanced' });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = store.insertMemory({
      agentId: 'main',
      type: 'preference',
      content: 'For implementation feedback, give concrete file-by-file details.',
      scopeKind: 'agent',
      normalizedKey: 'user:style:file-by-file',
      tags: ['style', 'planning'],
      importance: 0.9,
      freshness: 1,
      confidence: 0.92,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    store.insertPolicySnapshot({
      agentId: 'main',
      policyText: 'Prefer retrieve_memory on planning turns.',
      examples: [],
      model: 'deterministic',
      promptVersion: 'route-learning-v1',
      active: true,
    });
    store.insertCaptureAudit({
      agentId: 'main',
      turnId: 'turn-1',
      sessionId: 'session-1',
      retrievalIntent: { intent: 'no_retrieval', shouldRetrieve: false, includeRecallRules: false },
      captureIntent: { intent: 'standing_preference', shouldConsiderCapture: true, confidence: 0.75, reason: 'User stated preference', matchedSignals: ['I prefer'] },
      captureJobCreated: true,
      distillerRan: true,
      distillerModel: 'fake',
      distillerLatencyMs: 12,
      fallbackRan: false,
      candidateCount: 1,
      storedCount: 1,
      rejectedCount: 0,
      rejectionReasons: ['stored'],
      safeCandidatePreview: 'For implementation feedback, give concrete file-by-file details.',
      evidenceHash: 'h1',
    });
    store.close();

    const search = searchPayload(config, 'main', 'file-by-file', 10);
    assert.equal(search.results.length, 1);
    assert.equal(search.results[0].id, memory.id);

    const graph = graphPayload(config, 'main', 10);
    assert.equal(graph.nodes.length, 1);
    assert.equal(graph.nodes[0].id, memory.id);
    assert.equal(graph.counts.nodes, 1);

    const learn = learnPayload(config, 'main', 10);
    assert.ok(learn.activePolicySnapshot);
    assert.equal(learn.policySnapshots.length, 1);

    const audit = auditPayload(config, 'main', 10);
    assert.equal(audit.rows.length, 1);
    assert.equal(audit.rows[0].captureIntent, 'standing_preference');
    assert.equal(audit.captureOpportunityRate, 1);

    const explain = explainLastPayload(config, 'main');
    assert.equal(explain.ok, true);
    assert.match(explain.summary, /stored/);
    assert.equal(explain.capture.intent, 'standing_preference');

    const supplements = [];
    plugin.register({
      pluginConfig: { enabled: true, activationRoot: root, mode: 'balanced' },
      on() {},
      registerService() {},
      registerHttpRoute() {},
      registerMemoryPromptSupplement() {},
      registerMemoryCorpusSupplement(supplement) { supplements.push(supplement); },
      logger: { debug() {}, warn() {} },
    });
    const corpus = supplements[0];
    const results = await corpus.search({ query: 'file-by-file', maxResults: 5 });
    assert.equal(results.length, 1);
    const fetched = await corpus.get({ lookup: memory.id });
    assert.match(fetched.content, /file-by-file details/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('corpus get does not expose deleted or narrow session memories', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, mode: 'balanced' });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const deleted = store.insertMemory({
      agentId: 'main', type: 'preference', content: 'Deleted preference should not appear',
      scopeKind: 'agent', normalizedKey: 'deleted:pref', tags: [], importance: 0.9, freshness: 1, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    const sessionOnly = store.insertMemory({
      agentId: 'main', type: 'preference', content: 'Session codeword blue heron',
      scopeKind: 'session', scopeKey: 'session-a', normalizedKey: 'session:secret', tags: [], importance: 0.9, freshness: 1, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    store.softDeleteMemory(deleted.id);
    store.close();

    const corpus = buildMemoryCorpusSupplement(config);
    assert.equal(await corpus.get({ lookup: deleted.id }), null);
    assert.equal(await corpus.get({ lookup: sessionOnly.id }), null);
    const results = await corpus.search({ query: 'blue heron', maxResults: 10 });
    assert.equal(results.length, 0);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('HTTP payload helpers reject disallowed agent ids', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, scopes: { agents: ['main'] } });
    const store = new MemoryStore({ activationRoot: root, agentId: 'victim' });
    store.insertMemory({
      agentId: 'victim', type: 'preference', content: 'Victim-only memory',
      scopeKind: 'agent', normalizedKey: 'victim:pref', tags: [], importance: 0.9, freshness: 1, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    store.close();
    const payload = searchPayload(config, 'victim', 'Victim-only', 10);
    assert.equal(payload.ok, false);
    assert.equal(payload.reason, 'agent_not_allowed');
    assert.equal('results' in payload, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
