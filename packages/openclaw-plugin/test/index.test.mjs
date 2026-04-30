import assert from 'node:assert/strict';
import { lstat, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import plugin, {
  appendProofEvent,
  classifyTurn,
  decidePolicy,
  handleTurnHook,
  normalizePluginConfig,
  readActivationContext,
  readProofEvents,
  readStatus,
  redactText,
  redactedTurnFromPromptEvent,
  resolveOpenClawBrainConfig
} from '../dist/index.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-plugin-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

test('config defaults off and ignores root openclawbrain config', () => {
  const config = resolveOpenClawBrainConfig({
    runtime: { config: { loadConfig: () => ({ openclawbrain: { enabled: true }, plugins: { entries: {} } }) } }
  });
  assert.equal(config.enabled, false);
  assert.equal(config.mode, 'conservative');
  assert.deepEqual(config.scopes.agents, ['main']);
  assert.equal(config.hooks.allowPromptInjection, false);
});

test('rawTranscriptUpload=true fails closed/off', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, rawTranscriptUpload: true, activationRoot: root });
    assert.equal(config.enabled, false);
    assert.equal(config.rawTranscriptUpload, true);
    const result = await handleTurnHook({ agentId: 'main', userMessage: 'continue with secret=abc123' }, config, {}, 'before_prompt_build');
    assert.deepEqual(result, {});
    const proof = await readFile(path.join(root, 'proof-events.jsonl'), 'utf8');
    assert.match(proof, /raw_transcript_upload_requested/);
    assert.match(proof, /"rawTranscriptStored":false/);
    assert.doesNotMatch(proof, /secret=abc123/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('redaction covers common sensitive values', () => {
  const redacted = redactText('Email me@example.com token=sk-1234567890abcdef1234 call 415-555-1212 https://example.com/a');
  assert.doesNotMatch(redacted, /me@example/);
  assert.doesNotMatch(redacted, /sk-123/);
  assert.doesNotMatch(redacted, /415-555/);
  assert.doesNotMatch(redacted, /https:\/\//);
  assert.match(redacted, /\[redacted-email\]/);
});

test('policy maps selected v0.1 turn slices', () => {
  assert.equal(decidePolicy({ mode: 'conservative', turnType: 'direct-answer' }).kind, 'stay_silent');
  assert.equal(decidePolicy({ mode: 'conservative', turnType: 'unknown' }).kind, 'stay_silent');
  assert.equal(decidePolicy({ mode: 'conservative', turnType: 'correction-follow-up' }).kind, 'correction_only');
  assert.equal(decidePolicy({ mode: 'conservative', turnType: 'stale-memory-conflict' }).kind, 'correction_only');
  assert.equal(decidePolicy({ mode: 'conservative', turnType: 'continuation' }).kind, 'full_context');
  assert.equal(decidePolicy({ mode: 'conservative', turnType: 'retrieval-heavy' }).kind, 'full_context');
  const tool = decidePolicy({ mode: 'conservative', turnType: 'tool-heavy' });
  assert.equal(tool.kind, 'full_context');
  assert.equal(tool.verificationHint, true);
  assert.equal(classifyTurn({ redactedPrompt: 'please continue' }), 'continuation');
});

test('context files are fixed, local, redacted, clipped, and owner-only root', async () => {
  const root = await tempRoot();
  try {
    await writeFile(path.join(root, 'context.md'), `Context with user@example.com\n${'x'.repeat(2000)}`);
    await writeFile(path.join(root, 'corrections.md'), 'Use family calendar, not work calendar.');
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, maxContextChars: 700 });
    const activation = await readActivationContext(config, 'main', { kind: 'full_context', slice: 'continuation' });
    assert.match(activation.text, /\[redacted-email\]/);
    assert.ok(activation.text.length <= 700);
    assert.deepEqual(activation.rejectedFiles, []);
    assert.equal(activation.usedFileIdsRedacted.length, 2);
    const stat = await lstat(root);
    assert.ok((stat.mode & 0o077) === 0 || process.platform === 'win32');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('context reader rejects symlinks and oversize files before reading', async () => {
  const root = await tempRoot();
  try {
    const outside = path.join(root, 'outside.md');
    await writeFile(outside, 'outside secret');
    await symlink(outside, path.join(root, 'corrections.md'));
    await writeFile(path.join(root, 'context.md'), 'x'.repeat(20000));
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, maxContextChars: 500 });
    const activation = await readActivationContext(config, 'main', { kind: 'full_context', slice: 'continuation' });
    assert.equal(activation.text, '');
    assert.deepEqual(activation.rejectedFiles.map((entry) => entry.reasonCode).sort(), ['oversize_rejected', 'symlink_rejected']);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('direct-answer returns no prompt mutation but writes stay_silent proof', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, hooks: { allowPromptInjection: true } });
    const result = await handleTurnHook({ agentId: 'main', turnType: 'direct-answer', userMessage: 'What is 2+2?' }, config, {}, 'before_prompt_build');
    assert.deepEqual(result, {});
    const proof = await readProofEvents({ activationRoot: root, agentId: 'main', limit: 5 });
    assert.equal(proof.at(-1).decisionKind, 'stay_silent');
    assert.equal(proof.at(-1).rawUserTextStored, false);
    assert.equal(proof.at(-1).hashesOnlyForUserText, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('correction and full-context injections are bounded and redacted', async () => {
  const root = await tempRoot();
  try {
    await writeFile(path.join(root, 'corrections.md'), 'Prefer family inbox. Email private@example.com');
    await writeFile(path.join(root, 'context.md'), 'Continue the deployment checklist.');
    await writeFile(path.join(root, 'tool-guidance.md'), 'Run tests and inspect output.');
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, maxContextChars: 900, hooks: { allowPromptInjection: true } });
    const correction = await handleTurnHook({ agentId: 'main', turnType: 'correction-follow-up', userMessage: 'actually use family inbox' }, config, {}, 'before_prompt_build');
    assert.match(correction.prependContext, /correction guidance/);
    assert.match(correction.prependContext, /\[redacted-email\]/);
    assert.ok(correction.prependContext.length <= 1400);
    const tool = await handleTurnHook({ agentId: 'main', turnType: 'tool-heavy', userMessage: 'run tests' }, config, {}, 'before_prompt_build');
    assert.match(tool.prependContext, /Verification hint/);
    assert.match(tool.prependContext, /tool-guidance.md/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('prompt injection disabled returns no mutation and writes fail-closed proof', async () => {
  const root = await tempRoot();
  try {
    await writeFile(path.join(root, 'corrections.md'), 'Use family inbox.');
    const config = normalizePluginConfig({ enabled: true, activationRoot: root, hooks: { allowPromptInjection: false } });
    const result = await handleTurnHook({ agentId: 'main', turnType: 'stale-memory-conflict', userMessage: 'stale memory conflict' }, config, {}, 'before_prompt_build');
    assert.deepEqual(result, {});
    const proof = await readFile(path.join(root, 'proof-events.jsonl'), 'utf8');
    assert.match(proof, /prompt_injection_disabled/);
    assert.match(proof, /stay_silent/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('proof store retention and status are bounded and precise', async () => {
  const root = await tempRoot();
  try {
    for (let index = 0; index < 55; index += 1) {
      await appendProofEvent({ agentId: 'main', eventId: `e${index}`, decisionKind: 'stay_silent', rawUserText: `do not store ${index}` }, { activationRoot: root, agentId: 'main', proofRetentionEvents: 50 });
    }
    const events = await readProofEvents({ activationRoot: root, agentId: 'main', limit: 100 });
    assert.equal(events.length, 50);
    assert.equal(events[0].eventId, 'e5');
    assert.equal(events[0].rawTranscriptStored, false);
    assert.equal(events[0].rawUserTextStored, false);
    assert.equal(events[0].redactionApplied, true);
    assert.equal(events[0].hashesOnlyForUserText, true);
    assert.equal('containsRealUserData' in events[0], false);
    assert.doesNotMatch(JSON.stringify(events), /do not store/);
    const status = await readStatus({ activationRoot: root, agentId: 'main' });
    assert.equal(status, null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('plugin registers native surfaces, primary hook, optional hook safely, and gated agent_end', async () => {
  const calls = [];
  const routes = [];
  const services = [];
  plugin.register({
    pluginConfig: { enabled: true },
    supportsHook: (name) => name !== 'agent_turn_prepare',
    on: (name, fn) => calls.push([name, fn]),
    registerHttpRoute: (route) => routes.push(route),
    registerService: (service) => services.push(service),
    logger: { debug() {}, warn() {} }
  });
  assert.ok(calls.map(([name]) => name).includes('before_prompt_build'));
  assert.ok(!calls.map(([name]) => name).includes('agent_turn_prepare'));
  assert.ok(!calls.map(([name]) => name).includes('agent_end'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/status'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/proof'));
  assert.equal(services[0].id, 'openclawbrain');
  assert.equal(typeof services[0].start, 'function');
  assert.equal(typeof services[0].stop, 'function');

  const withoutHookDiscovery = [];
  plugin.register({ pluginConfig: { enabled: true }, on: (name, fn) => withoutHookDiscovery.push([name, fn]) });
  assert.ok(!withoutHookDiscovery.map(([name]) => name).includes('agent_turn_prepare'));

  const withConversation = [];
  plugin.register({ pluginConfig: { enabled: true, hooks: { allowConversationAccess: true } }, supportsHook: (name) => name === 'agent_end', on: (name, fn) => withConversation.push([name, fn]) });
  assert.ok(withConversation.map(([name]) => name).includes('agent_end'));
});

test('redacted turn hashes raw user text without storing it', () => {
  const config = normalizePluginConfig({ enabled: true });
  const turn = redactedTurnFromPromptEvent({ ctx: { agentId: 'main', sessionKey: 'session-secret' }, userMessage: 'my email is user@example.com' }, config);
  assert.match(turn.promptHash, /^sha256:/);
  assert.match(turn.sessionKeyHash, /^sha256:/);
  assert.match(turn.summary, /\[redacted-email\]/);
  assert.doesNotMatch(JSON.stringify(turn), /user@example.com|session-secret/);
});
