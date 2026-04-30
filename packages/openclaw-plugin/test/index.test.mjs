import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import plugin, { normalizePluginConfig, redactedTurnFromPromptEvent, resolveOpenClawBrainConfig } from '../src/index.mjs';

test('plugin registers before_prompt_build and agent_end hooks', () => { const calls = []; plugin.register({ pluginConfig: {}, on: (name, fn) => calls.push([name, fn]) }); assert.deepEqual(calls.map(([name]) => name), ['before_prompt_build', 'agent_end']); });

test('before_prompt_build injects correction and writes proof under activation root', async () => { const root = await mkdtemp(path.join(tmpdir(), 'ocb-plugin-')); try { const hooks = {}; plugin.register({ pluginConfig: { enabled: true, mode: 'conservative', activationRoot: root, scopes: { agents: ['main'] }, candidateMemories: [{ id: 'correction-redacted', kind: 'correction', text: 'avoid work email for family tasks', relevance: 1 }] }, on: (name, fn) => { hooks[name] = fn; }, logger: { warn() {} } }); const result = await hooks.before_prompt_build({ agentId: 'main', turnId: 't1', turnType: 'stale-memory-conflict', userMessageRedacted: 'ambiguous family email request' }); assert.match(result.prependSystemContext, /Relevant user correction/); const proof = await readFile(path.join(root, 'proof-events.jsonl'), 'utf8'); assert.match(proof, /correction_only/); } finally { await rm(root, { recursive: true, force: true }); } });

test('before_prompt_build stays silent when scoped agent does not match', async () => { const hooks = {}; plugin.register({ pluginConfig: { scopes: { agents: ['family'] } }, on: (name, fn) => { hooks[name] = fn; } }); assert.deepEqual(await hooks.before_prompt_build({ agentId: 'main', userMessageRedacted: 'continue' }), {}); });

test('config normalization forces rawTranscriptUpload false', () => { const config = normalizePluginConfig({ rawTranscriptUpload: true, mode: 'active' }); assert.equal(config.rawTranscriptUpload, false); assert.equal(config.mode, 'active'); });

test('redacted turn builder does not require raw transcript fields', () => { const turn = redactedTurnFromPromptEvent({ agentId: 'main', userMessageRedacted: 'continue' }, normalizePluginConfig({})); assert.equal(turn.summary, 'continue'); assert.equal(turn.agentId, 'main'); });

test('resolves live plugin config when OpenClaw runtime exposes it', () => { const config = resolveOpenClawBrainConfig({ pluginConfig: { mode: 'conservative' }, runtime: { config: { loadConfig: () => ({ plugins: { entries: { openclawbrain: { config: { mode: 'proof-only', activationRoot: '/tmp/ocb-live', scopes: { agents: [] } } } } } }) } } }); assert.equal(config.mode, 'proof-only'); assert.equal(config.activationRoot, '/tmp/ocb-live'); assert.deepEqual(config.scopes.agents, []); });
