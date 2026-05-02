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
import { MemoryStore } from '../dist/memory-store.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-plugin-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

test('config defaults on and ignores root openclawbrain config', () => {
  const config = resolveOpenClawBrainConfig({
    runtime: { config: { current: () => ({ openclawbrain: { enabled: false }, plugins: { entries: {} } }) } }
  });
  assert.equal(config.enabled, true);
  assert.equal(config.mode, 'balanced');
  assert.equal(config.llm.baseUrl, 'http://127.0.0.1:11434/v1');
  assert.equal(config.llm.routeModel, 'qwen2.5:32b-instruct');
  assert.deepEqual(config.scopes.agents, ['main']);
  assert.equal(config.hooks.allowPromptContext, true);
  assert.equal(config.hooks.allowConversationAccess, true);
  assert.equal(config.hooks.allowToolObservation, true);
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

test('policy maps selected legacy turn slices', () => {
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
    const config = normalizePluginConfig({ enabled: true, mode: 'conservative', activationRoot: root, hooks: { allowPromptContext: true } });
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
    const config = normalizePluginConfig({ enabled: true, mode: 'conservative', activationRoot: root, maxContextChars: 900, hooks: { allowPromptContext: true } });
    const correction = await handleTurnHook({ agentId: 'main', turnType: 'correction-follow-up', userMessage: 'actually use family inbox' }, config, {}, 'before_prompt_build');
    assert.match(correction.prependContext, /correction context/);
    assert.match(correction.prependContext, /\[redacted-email\]/);
    assert.ok(correction.prependContext.length <= 1400);
    const tool = await handleTurnHook({ agentId: 'main', turnType: 'tool-heavy', userMessage: 'run tests' }, config, {}, 'before_prompt_build');
    assert.match(tool.prependContext, /Verification hint/);
    assert.match(tool.prependContext, /tool-guidance.md/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('prompt context disabled returns no mutation and writes fail-closed proof', async () => {
  const root = await tempRoot();
  try {
    await writeFile(path.join(root, 'corrections.md'), 'Use family inbox.');
    const config = normalizePluginConfig({ enabled: true, mode: 'conservative', activationRoot: root, hooks: { allowPromptContext: false } });
    const result = await handleTurnHook({ agentId: 'main', turnType: 'stale-memory-conflict', userMessage: 'stale memory conflict' }, config, {}, 'before_prompt_build');
    assert.deepEqual(result, {});
    const proof = await readFile(path.join(root, 'proof-events.jsonl'), 'utf8');
    assert.match(proof, /prompt_context_disabled/);
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

test('proof store skips malformed JSONL mirror records', async () => {
  const root = await tempRoot();
  try {
    const file = path.join(root, 'proof-events.jsonl');
    await writeFile(file, `${JSON.stringify({ agentId: 'main', eventId: 'before', decisionKind: 'stay_silent' })}\nue}\n`);
    await appendProofEvent({ agentId: 'main', eventId: 'after', decisionKind: 'stay_silent' }, { activationRoot: root, agentId: 'main', proofRetentionEvents: 50 });
    const proof = await readFile(file, 'utf8');
    assert.doesNotMatch(proof, /^ue}$/m);
    assert.match(proof, /"eventId":"before"/);
    assert.match(proof, /"eventId":"after"/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('plugin registers native surfaces, primary hook, optional hook safely, and default agent_end', async () => {
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
  assert.ok(calls.map(([name]) => name).includes('agent_end'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/status'));
  assert.ok(routes.some((route) => route.path === '/plugins/openclawbrain/proof'));
  assert.equal(services[0].id, 'openclawbrain');
  assert.equal(typeof services[0].start, 'function');
  assert.equal(typeof services[0].stop, 'function');

  const withoutHookDiscovery = [];
  plugin.register({ pluginConfig: { enabled: true }, on: (name, fn) => withoutHookDiscovery.push([name, fn]) });
  assert.ok(!withoutHookDiscovery.map(([name]) => name).includes('agent_turn_prepare'));

  const withoutConversation = [];
  plugin.register({ pluginConfig: { enabled: true, hooks: { allowConversationAccess: false } }, supportsHook: (name) => name === 'agent_end', on: (name, fn) => withoutConversation.push([name, fn]) });
  assert.ok(!withoutConversation.map(([name]) => name).includes('agent_end'));
});

test('redacted turn hashes raw user text without storing it', () => {
  const config = normalizePluginConfig({ enabled: true });
  const turn = redactedTurnFromPromptEvent({ ctx: { agentId: 'main', sessionKey: 'session-secret' }, userMessage: 'my email is user@example.com' }, config);
  assert.match(turn.promptHash, /^sha256:/);
  assert.match(turn.sessionKeyHash, /^sha256:/);
  assert.match(turn.summary, /\[redacted-email\]/);
  assert.doesNotMatch(JSON.stringify(turn), /user@example.com|session-secret/);
});

test('before_prompt_build prompt text is captured for memory routing', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      llm: { enabled: false },
      routing: { enabled: true },
    });
    const result = await handleTurnHook({ agentId: 'main', prompt: 'Remember that I prefer concise Telegram replies.' }, config, {}, 'before_prompt_build');
    assert.deepEqual(result, {});
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const audit = store.listCaptureAudit('main', 10);
    assert.equal(audit.length, 1);
    assert.equal(audit[0].captureIntent.intent, 'explicit_store');
    assert.equal(audit[0].captureJobCreated, true);
    assert.equal(store.countMemories('main', 'preference'), 0);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('registered typed hook preserves OpenClaw ctx and suppresses synthetic heartbeat capture', async () => {
  const root = await tempRoot();
  try {
    const calls = [];
    plugin.register({
      pluginConfig: {
        enabled: true,
        mode: 'balanced',
        activationRoot: root,
        llm: { enabled: false },
        routing: { enabled: true },
        learning: { enabled: false },
      },
      on: (name, fn) => calls.push([name, fn]),
      registerHttpRoute() {},
      registerService() {},
      supportsHook: () => false,
      logger: { debug() {}, warn() {} },
    });
    const hook = calls.find(([name]) => name === 'before_prompt_build')[1];
    await hook(
      { prompt: 'Remember that I prefer concise Telegram replies.' },
      { agentId: 'main', sessionKey: 'agent:main:abc', sessionId: 'session-abc', runId: 'run-abc', trigger: 'user', messageProvider: 'telegram' },
    );

    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    let audit = store.listCaptureAudit('main', 10);
    assert.equal(audit[0].sessionId, 'session-abc');
    assert.equal(audit[0].runId, 'run-abc');
    assert.equal(audit[0].captureIntent.intent, 'explicit_store');
    assert.equal(audit[0].captureJobCreated, true);

    await hook(
      { prompt: 'Read HEARTBEAT.md if it exists. Do not infer or repeat old tasks.' },
      { agentId: 'main', sessionKey: 'agent:main:abc', sessionId: 'session-abc', runId: 'run-heartbeat', trigger: 'heartbeat' },
    );
    audit = store.listCaptureAudit('main', 10);
    assert.equal(audit[0].runId, 'run-heartbeat');
    assert.equal(audit[0].captureIntent.intent, 'one_off');
    assert.equal(audit[0].captureIntent.shouldConsiderCapture, false);
    assert.equal(audit[0].captureJobCreated, false);
    assert.match(audit[0].rejectionReasons.join(' '), /System-generated heartbeat prompt/);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('capture and routing off modes are authoritative', async () => {
  const root = await tempRoot();
  try {
    const captureOff = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: root,
      llm: { enabled: false },
      capture: { mode: 'off' },
      routing: { enabled: true },
    });
    await handleTurnHook({ agentId: 'main', prompt: 'Remember that I prefer pnpm.' }, captureOff, {}, 'before_prompt_build');
    let store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    let audit = store.listCaptureAudit('main', 10);
    assert.equal(audit.length, 1);
    assert.equal(audit[0].captureJobCreated, false);
    assert.equal(store.getJobQueueDepth('main'), 0);
    store.close();

    const routingOffRoot = await tempRoot();
    const routingOff = normalizePluginConfig({
      enabled: true,
      mode: 'balanced',
      activationRoot: routingOffRoot,
      llm: { enabled: false },
      routing: { enabled: true, mode: 'off' },
    });
    const result = await handleTurnHook({ agentId: 'main', prompt: 'Use my prior package-manager preference.' }, routingOff, {}, 'before_prompt_build');
    assert.deepEqual(result, {});
    store = new MemoryStore({ activationRoot: routingOffRoot, agentId: 'main' });
    audit = store.listCaptureAudit('main', 10);
    assert.equal(audit.length, 0);
    store.close();
    await rm(routingOffRoot, { recursive: true, force: true });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('agent_end synthetic capture and tool outcomes without correlation ids do not enqueue or resolve unrelated work', async () => {
  const root = await tempRoot();
  try {
    const calls = [];
    plugin.register({
      pluginConfig: {
        enabled: true,
        mode: 'balanced',
        activationRoot: root,
        llm: { enabled: false },
        learning: { enabled: true },
        hooks: { allowConversationAccess: true, allowToolObservation: true },
      },
      on: (name, fn) => calls.push([name, fn]),
      supportsHook: (name) => name === 'agent_end' || name === 'after_tool_call',
      registerHttpRoute() {},
      registerService() {},
      logger: { debug() {}, warn() {} },
    });
    const agentEnd = calls.find(([name]) => name === 'agent_end')[1];
    const afterTool = calls.find(([name]) => name === 'after_tool_call')[1];

    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const mem = store.insertMemory({
      agentId: 'main', type: 'preference', content: 'Use pnpm',
      scopeKind: 'agent', normalizedKey: 'pref:pnpm', tags: [], importance: 0.8, freshness: 1, confidence: 0.9,
      useCount: 0, usefulCount: 0, captureCount: 1,
    });
    store.insertInjection({ agentId: 'main', memoryId: mem.id, runId: 'real-run', turnId: 'real-turn', query: 'deps', rank: 1, score: 0.8 });
    store.close();

    await agentEnd({ userMessage: 'Going forward, always use pnpm.' }, { agentId: 'main', trigger: 'heartbeat', runId: 'heartbeat-run', sessionId: 's1' });
    await afterTool({ toolName: 'exec', ok: true, args: { password: 'hunter2' } }, { agentId: 'main' });

    const verify = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const audit = verify.listCaptureAudit('main', 10);
    assert.equal(audit[0].captureIntent.intent, 'one_off');
    assert.equal(audit[0].captureJobCreated, false);
    assert.equal(verify.getJobQueueDepth('main'), 0);
    assert.equal(verify.getPendingInjections('main').length, 1);
    verify.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
