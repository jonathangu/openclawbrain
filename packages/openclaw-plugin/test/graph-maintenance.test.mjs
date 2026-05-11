import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import plugin, {
  GraphMaintenanceEngine,
  graphMaintenancePayload,
  handleBrainCommand,
  normalizePluginConfig,
  processAutomaticGraphMaintenance,
} from '../dist/index.js';
import { MemoryStore } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-graph-maint-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

function config(root) {
  return normalizePluginConfig({ activationRoot: root, scopes: { agents: ['main'] } });
}

function insertMemory(store, overrides = {}) {
  return store.insertMemory({
    agentId: 'main',
    type: 'preference',
    content: 'User prefers concise operator summaries in Telegram.',
    scopeKind: 'global_user',
    normalizedKey: `pref:test:${Math.random().toString(16).slice(2)}`,
    tags: ['test'],
    importance: 0.7,
    freshness: 1,
    confidence: 0.9,
    useCount: 0,
    usefulCount: 0,
    captureCount: 1,
    ...overrides,
  });
}

test('graph maintenance dry-run reports health and creates deterministic proposals', async () => {
  const root = await tempRoot();
  try {
    const cfg = config(root);
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const first = insertMemory(store, {
      content: 'Use pnpm in the OpenClawBrain repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:tooling:a',
    });
    const duplicate = insertMemory(store, {
      content: 'Use pnpm in the OpenClawBrain repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:tooling:b',
    });
    store.insertEdge({ agentId: 'main', fromId: first.id, toId: 'missing-node', relation: 'related', weight: 0.5, evidenceCount: 1 });

    const report = new GraphMaintenanceEngine({ store, config: cfg }).dryRun('main');
    assert.equal(report.ok, true);
    assert.equal(report.health.counts.exactDuplicateClusters, 1);
    assert.equal(report.health.counts.badEdges, 1);
    assert.ok(report.proposals.some((proposal) => proposal.proposalType === 'merge_exact_duplicate_nodes'));
    assert.ok(report.proposals.some((proposal) => proposal.proposalType === 'retire_bad_edge'));
    assert.ok(report.proposals.every((proposal) => JSON.stringify(proposal.evidence).includes('Use pnpm') || proposal.proposalType !== 'block_tombstone_recapture'));
    assert.ok(store.getGraphMaintenanceRun(report.run.id));
    assert.ok(store.listGraphMaintenanceProposals('main').length >= 2);
    assert.notEqual(first.id, duplicate.id);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('exact duplicate apply preserves canonical lineage and redacted proof', async () => {
  const root = await tempRoot();
  try {
    const cfg = config(root);
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const canonical = insertMemory(store, {
      content: 'Use pnpm in this repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:pkg:a',
      importance: 0.9,
    });
    const duplicate = insertMemory(store, {
      content: 'Use pnpm in this repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:pkg:b',
      importance: 0.4,
    });

    const engine = new GraphMaintenanceEngine({ store, config: cfg });
    const proposal = engine.dryRun('main').proposals.find((item) => item.proposalType === 'merge_exact_duplicate_nodes');
    assert.ok(proposal);
    const result = engine.applyProposal('main', proposal.id);
    assert.equal(result.ok, true);
    assert.equal(result.proposal.status, 'applied');
    assert.equal(store.getMemory(duplicate.id).supersededBy, canonical.id);
    assert.ok(store.listMemoryNodeLineage('main', duplicate.id).some((lineage) => lineage.parentMemoryId === canonical.id && lineage.relation === 'merged_into'));
    assert.ok(store.getProofEvents('main', 10).some((event) => event.kind === 'graph_maintenance_applied' && event.rawTranscriptStored === false));
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('bad edge apply retires the edge and records an observation', async () => {
  const root = await tempRoot();
  try {
    const cfg = config(root);
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = insertMemory(store);
    const edge = store.insertEdge({ agentId: 'main', fromId: memory.id, toId: 'missing-node', relation: 'related', weight: 0.5, evidenceCount: 1 });

    const engine = new GraphMaintenanceEngine({ store, config: cfg });
    const proposal = engine.dryRun('main').proposals.find((item) => item.proposalType === 'retire_bad_edge');
    assert.ok(proposal);
    const result = engine.applyProposal('main', proposal.id);
    assert.equal(result.ok, true);
    assert.equal(store.listEdgesForAgent('main').some((candidate) => candidate.id === edge.id), false);
    assert.ok(store.listMemoryEdgeObservations('main').some((obs) => obs.edgeId === edge.id && obs.observationType === 'bad_edge_retired'));
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('stale high-authority proposal is review gated and not safe auto-applied', async () => {
  const root = await tempRoot();
  try {
    const cfg = config(root);
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = insertMemory(store, {
      type: 'workflow',
      content: 'Deploy this repo with the old release script.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:deploy-script',
      importance: 0.95,
      freshness: 0.2,
    });
    store.patchMemoryValidity(memory.id, {
      temporalValidity: 'stale',
      behavioralAuthorityScore: 0.9,
      validationStrategy: 'environment_check',
      stateReason: 'test_stale_high_authority',
    });

    const engine = new GraphMaintenanceEngine({ store, config: cfg });
    const proposal = engine.dryRun('main').proposals.find((item) => item.proposalType === 'mark_stale_high_authority');
    assert.ok(proposal);
    assert.equal(proposal.risk, 'medium');
    assert.equal(proposal.status, 'pending_review');
    assert.equal(engine.applyProposal('main', proposal.id).ok, false);
    assert.equal(store.getMemoryValidity(memory.id).behavioralAuthorityScore, 0.9);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('tombstone recapture proposals use hashes and do not leak tombstoned content', async () => {
  const root = await tempRoot();
  try {
    const cfg = config(root);
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const tombstone = insertMemory(store, {
      type: 'recall_rule',
      content: 'Secret codeword is Blue Heron.',
      scopeKind: 'app',
      scopeKey: 'CormorantAI',
      normalizedKey: 'recall:app:cormorantai:codeword',
    });
    store.tombstoneMemory(tombstone.id, { redactContent: true, reason: 'test_forget' });
    insertMemory(store, {
      type: 'recall_rule',
      content: 'When asked for the app codeword, provide the remembered phrase.',
      scopeKind: 'app',
      scopeKey: 'CormorantAI',
      normalizedKey: 'recall:app:cormorantai:codeword:rev:test',
    });

    const proposal = new GraphMaintenanceEngine({ store, config: cfg }).dryRun('main').proposals.find((item) => item.proposalType === 'block_tombstone_recapture');
    assert.ok(proposal);
    assert.equal(proposal.risk, 'high');
    assert.doesNotMatch(JSON.stringify(proposal), /Blue Heron/i);
    assert.match(JSON.stringify(proposal.evidence), /sha256:/);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('scoped exception and feedback learning stay proposal or observation only', async () => {
  const root = await tempRoot();
  try {
    const cfg = config(root);
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const a = insertMemory(store, { normalizedKey: 'pref:concise-default' });
    const b = insertMemory(store, { content: 'Use status, blocker, next action in Codex handoffs.', normalizedKey: 'workflow:codex:handoff' });
    store.insertMemoryAuthorityEvent({ agentId: 'main', memoryId: a.id, eventType: 'overridden_by_current_instruction', source: 'test', reason: 'current_instruction:asks_for_depth' });
    store.insertMemoryAuthorityEvent({ agentId: 'main', memoryId: a.id, eventType: 'overridden_by_current_instruction', source: 'test', reason: 'current_instruction:asks_for_depth' });
    store.insertRouteTeacherRun({
      agentId: 'main',
      routeDecisionId: 'route-1',
      model: 'test',
      promptVersion: 'test',
      inputHash: 'in',
      outputHash: 'out',
      verdict: 'correct_route',
      teacherRoute: 'retrieve_memory',
      teacherMemoryIds: [a.id, b.id],
      teacherQueries: ['codex handoff'],
      teacherGraphDepth: 1,
      syncPlannerWorthIt: false,
      confidence: 0.8,
      rationale: 'test',
      validated: true,
    });

    const engine = new GraphMaintenanceEngine({ store, config: cfg });
    const report = engine.dryRun('main');
    const scoped = report.proposals.find((proposal) => proposal.proposalType === 'propose_scoped_exception');
    assert.ok(scoped);
    assert.equal(scoped.status, 'pending_review');
    assert.equal(engine.applyProposal('main', scoped.id).ok, false);

    const feedback = report.proposals.find((proposal) => proposal.proposalType === 'record_feedback_edge_observation');
    assert.ok(feedback);
    const applied = engine.applyProposal('main', feedback.id);
    assert.equal(applied.ok, true);
    const observation = store.listMemoryEdgeObservations('main').find((obs) => obs.observationType === 'route_teacher_behavioral_observation');
    assert.ok(observation);
    assert.equal(observation.sourceIndependence, 'derived');
    assert.equal(observation.causalAttribution, 'unknown');
    assert.equal(observation.edgeFamily, 'behavioral');
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('graph routes and /brain graph commands are available through the plugin surface', async () => {
  const root = await tempRoot();
  try {
    const cfg = config(root);
    const routes = [];
    const commands = [];
    plugin.register({
      registerService() {},
      registerCommand(command) { commands.push(command); },
      registerHttpRoute(route) { routes.push(route); },
      on() {},
      runtime: { config: { current: () => ({ plugins: { entries: { openclawbrain: { config: cfg } } } }) } },
    });
    assert.ok(commands.some((command) => command.name === 'brain'));
    assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/graph/health'));
    assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/graph/dry-run'));
    assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/graph/apply'));

    const payload = graphMaintenancePayload(cfg, { query: { agentId: 'main' } }, 'health');
    assert.equal(payload.ok, true);
    const commandResult = await handleBrainCommand({ args: 'graph health', agentId: 'main' }, cfg, {});
    assert.match(commandResult.text, /graph health/i);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('automatic graph maintenance proposes passively and only auto-applies safe low-risk repairs', async () => {
  const root = await tempRoot();
  try {
    const cfg = normalizePluginConfig({
      activationRoot: root,
      scopes: { agents: ['main'] },
      graphMaintenance: {
        enabled: true,
        mode: 'passive',
        safeAutoApply: true,
        maxSafeAutoApplyPerRun: 5,
      },
    });
    let store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const first = insertMemory(store, {
      content: 'Use pnpm in this repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:pkg:first',
      importance: 0.9,
    });
    const duplicate = insertMemory(store, {
      content: 'Use pnpm in this repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:pkg:duplicate',
      importance: 0.4,
    });
    const stale = insertMemory(store, {
      type: 'workflow',
      content: 'Deploy with the old release script.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:old-deploy-script',
      importance: 0.95,
      freshness: 0.1,
    });
    store.patchMemoryValidity(stale.id, {
      temporalValidity: 'stale',
      behavioralAuthorityScore: 0.92,
      validationStrategy: 'environment_check',
      stateReason: 'test_stale_high_authority',
    });
    store.close();

    await processAutomaticGraphMaintenance(cfg, { logger: { info() {}, warn() {} } });

    store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const firstAfter = store.getMemory(first.id);
    const duplicateAfter = store.getMemory(duplicate.id);
    assert.equal([firstAfter.supersededBy, duplicateAfter.supersededBy].filter(Boolean).length, 1);
    assert.equal(store.getMemoryValidity(stale.id).behavioralAuthorityScore, 0.92);
    assert.ok(store.listGraphMaintenanceRuns('main').some((run) => run.mode === 'safe_auto' && run.proposalsApplied >= 1));
    assert.ok(store.listGraphMaintenanceProposals('main', { status: 'pending_review' }).some((proposal) => proposal.proposalType === 'mark_stale_high_authority'));
    assert.ok(store.getProofEvents('main', 20).some((event) => event.kind === 'graph_maintenance_auto_cycle'));
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
