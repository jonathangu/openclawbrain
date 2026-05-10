import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  buildCodexBridgeStatus,
  buildCodexHandoff,
  CodexBridgeStore,
  formatHandoffBrief,
  handleBrainCommand,
  normalizePluginConfig,
  processCodexBridgeWatches,
} from '../dist/index.js';

const require = createRequire(import.meta.url);
const BetterSqlite3 = require('better-sqlite3');

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-codex-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

function createCodexState(dbPath, threads = []) {
  const db = new BetterSqlite3(dbPath);
  db.exec(`
    CREATE TABLE threads (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      cwd TEXT NOT NULL,
      git_branch TEXT,
      git_sha TEXT,
      updated_at INTEGER NOT NULL,
      updated_at_ms INTEGER,
      archived INTEGER NOT NULL DEFAULT 0,
      model TEXT,
      reasoning_effort TEXT
    );
    CREATE TABLE thread_goals (
      thread_id TEXT PRIMARY KEY NOT NULL,
      goal_id TEXT NOT NULL,
      objective TEXT NOT NULL,
      status TEXT NOT NULL,
      token_budget INTEGER,
      tokens_used INTEGER NOT NULL DEFAULT 0,
      time_used_seconds INTEGER NOT NULL DEFAULT 0,
      updated_at_ms INTEGER NOT NULL
    );
  `);
  const insertThread = db.prepare(`
    INSERT INTO threads (
      id, title, cwd, git_branch, git_sha, updated_at, updated_at_ms, archived, model, reasoning_effort
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertGoal = db.prepare(`
    INSERT INTO thread_goals (
      thread_id, goal_id, objective, status, token_budget, tokens_used, time_used_seconds, updated_at_ms
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  for (const thread of threads) {
    insertThread.run(
      thread.id,
      thread.title,
      thread.cwd,
      thread.branch ?? null,
      thread.sha ?? null,
      Math.floor((thread.updatedAtMs ?? Date.now()) / 1000),
      thread.updatedAtMs ?? Date.now(),
      thread.archived ? 1 : 0,
      thread.model ?? 'gpt-5.5',
      thread.reasoningEffort ?? 'high',
    );
    if (thread.goal) {
      insertGoal.run(
        thread.id,
        thread.goal.goalId ?? `goal-${thread.id}`,
        thread.goal.objective,
        thread.goal.status,
        thread.goal.tokenBudget ?? null,
        thread.goal.tokensUsed ?? 0,
        thread.goal.timeUsedSeconds ?? 0,
        thread.goal.updatedAtMs ?? thread.updatedAtMs ?? Date.now(),
      );
    }
  }
  db.close();
}

function config(root, statePath, extra = {}) {
  return normalizePluginConfig({
    activationRoot: root,
    codexBridge: {
      preferAppServer: false,
      statePaths: [statePath],
      bridgeStatePath: path.join(root, '${agentId}', 'codex-continuity.sqlite'),
      appServerTimeoutMs: 100,
      ...extra,
    },
  });
}

test('Codex bridge reads SQLite fallback, labels it stale, and separates active goals', async () => {
  const root = await tempRoot();
  try {
    const dbPath = path.join(root, 'state_5.sqlite');
    const now = Date.now();
    createCodexState(dbPath, [
      { id: 'thread-active', title: 'Finish continuity bridge', cwd: '/repo/openclawbrain', branch: 'main', updatedAtMs: now, goal: { objective: 'Build bridge', status: 'active' } },
      { id: 'thread-done', title: 'Older completed work', cwd: '/repo/old', branch: 'done', updatedAtMs: now - 1000, goal: { objective: 'Done thing', status: 'complete' } },
    ]);
    const status = await buildCodexBridgeStatus(config(root, dbPath), 'main', { nowMs: () => now });
    assert.equal(status.ok, true);
    assert.equal(status.source, 'sqlite_fallback');
    assert.equal(status.stale, true);
    assert.equal(status.counts.threads, 2);
    assert.equal(status.counts.activeGoals, 1);
    assert.equal(status.activeGoals[0].id, 'thread-active');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('Codex bridge uses app-server reader when available', async () => {
  const root = await tempRoot();
  try {
    const cfg = normalizePluginConfig({
      activationRoot: root,
      codexBridge: {
        preferAppServer: true,
        statePaths: [path.join(root, 'missing.sqlite')],
        bridgeStatePath: path.join(root, '${agentId}', 'codex-continuity.sqlite'),
      },
    });
    const status = await buildCodexBridgeStatus(cfg, 'main', {
      nowMs: () => 1778420000000,
      appServerReader: {
        async listThreads() {
          return {
            threads: [
              {
                id: 'app-thread',
                title: 'Live app-server thread',
                cwd: '/repo/live',
                updatedAtMs: 1778420000000,
                goal: { goalId: 'g1', objective: 'Observe Codex', status: 'active', tokensUsed: 10, timeUsedSeconds: 5, updatedAtMs: 1778420000000 },
              },
            ],
          };
        },
      },
    });
    assert.equal(status.source, 'app_server');
    assert.equal(status.stale, false);
    assert.equal(status.capabilities.appServerAvailable, true);
    assert.equal(status.activeGoals[0].id, 'app-thread');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('Codex bridge redacts thread summaries before audit storage', async () => {
  const root = await tempRoot();
  try {
    const dbPath = path.join(root, 'state_5.sqlite');
    createCodexState(dbPath, [
      { id: 'secret-thread', title: 'Email me@example.com token=sk-1234567890abcdef1234', cwd: '/repo/private', updatedAtMs: Date.now() },
    ]);
    const cfg = config(root, dbPath);
    const status = await buildCodexBridgeStatus(cfg, 'main');
    assert.doesNotMatch(status.latestThreads[0].title, /me@example|sk-123/);
    const store = new CodexBridgeStore({ config: cfg, agentId: 'main' });
    store.recordEvent({
      agentId: 'main',
      eventType: 'status_snapshot',
      eventClass: 'status_snapshot',
      source: status.source,
      decision: 'recorded',
      notified: false,
      reason: 'test',
      redactedSummary: status.latestThreads[0].title,
      dedupeKey: 'redaction-test',
    });
    const event = store.listEvents('main', 1)[0];
    assert.doesNotMatch(event.redactedSummary, /me@example|sk-123/);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('Codex watch processing dedupes completion notifications', async () => {
  const root = await tempRoot();
  try {
    const dbPath = path.join(root, 'state_5.sqlite');
    const now = Date.now();
    createCodexState(dbPath, [
      { id: 'watched-thread', title: 'Watched task', cwd: '/repo/openclawbrain', updatedAtMs: now, goal: { objective: 'Finish', status: 'complete', updatedAtMs: now } },
    ]);
    const cfg = config(root, dbPath);
    const store = new CodexBridgeStore({ config: cfg, agentId: 'main' });
    store.createWatch({
      agentId: 'main',
      scope: 'thread',
      threadId: 'watched-thread',
      notifyChannel: 'telegram',
      notifyTarget: '-123',
      allowedClasses: ['completion', 'failure', 'blocker', 'approval_required', 'auth_failure'],
      sensitivity: 'normal',
      verbosity: 'blockers_and_completion',
    });
    store.close();
    const sent = [];
    const api = {
      runtime: {
        config: { current: () => ({}) },
        channel: { outbound: { loadAdapter: async () => ({ sendText: async (payload) => sent.push(payload) }) } },
      },
    };
    const first = await processCodexBridgeWatches(cfg, api, { nowMs: () => now });
    const second = await processCodexBridgeWatches(cfg, api, { nowMs: () => now });
    assert.equal(first.notified, 1);
    assert.equal(second.notified, 0);
    assert.equal(sent.length, 1);
    assert.match(sent[0].text, /Codex update: completion/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('Codex handoff keeps observed facts separate from Codex-reported claims', async () => {
  const root = await tempRoot();
  try {
    const dbPath = path.join(root, 'state_5.sqlite');
    createCodexState(dbPath, [
      { id: 'handoff-thread', title: 'Handoff task', cwd: '/repo/openclawbrain', branch: 'main', updatedAtMs: Date.now(), goal: { objective: 'Tests passed according to Codex', status: 'complete' } },
    ]);
    const status = await buildCodexBridgeStatus(config(root, dbPath), 'main');
    const brief = buildCodexHandoff(status, 'handoff-thread');
    assert.ok(brief.observedFacts.some((item) => item.includes('Workspace: /repo/openclawbrain')));
    assert.ok(brief.codexReportedClaims.some((item) => item.includes('Tests passed according to Codex')));
    assert.ok(brief.evidence.some((item) => item.includes('No command output')));
    assert.match(formatHandoffBrief(brief), /Codex-reported claims:/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('Brain command exposes status and refuses write path by default', async () => {
  const root = await tempRoot();
  try {
    const dbPath = path.join(root, 'state_5.sqlite');
    createCodexState(dbPath, [
      { id: 'cmd-thread', title: 'Command task', cwd: '/repo/openclawbrain', updatedAtMs: Date.now(), goal: { objective: 'Command visible', status: 'active' } },
    ]);
    const cfg = config(root, dbPath);
    const status = await handleBrainCommand({ args: 'codex status', channel: 'telegram', isAuthorizedSender: true }, cfg);
    assert.match(status.text, /Codex continuity:/);
    const goal = await handleBrainCommand({ args: 'codex goal do it', channel: 'telegram', isAuthorizedSender: true }, cfg);
    assert.match(goal.text, /disabled/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
