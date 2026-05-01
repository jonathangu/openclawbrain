import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import plugin, { graphPayload, learnPayload, normalizePluginConfig, searchPayload } from '../dist/index.js';
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
    store.close();

    const search = searchPayload(config, 'main', 'file-by-file', 10);
    assert.equal(search.results.length, 1);
    assert.equal(search.results[0].id, memory.id);

    const graph = graphPayload(config, 'main', 10);
    assert.equal(graph.nodes.length, 1);
    assert.equal(graph.nodes[0].id, memory.id);

    const learn = learnPayload(config, 'main', 10);
    assert.ok(learn.activePolicySnapshot);
    assert.equal(learn.policySnapshots.length, 1);

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
