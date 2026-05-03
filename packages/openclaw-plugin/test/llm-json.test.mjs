import assert from 'node:assert/strict';
import test from 'node:test';

import { FakeLlmClient, OllamaNativeLlmClient, isOllamaLoopbackBaseUrl } from '../dist/llm-client.js';
import {
  JsonParseError,
  JsonTimeoutError,
  JsonValidationError,
  normalizeJsonCandidate,
  runJsonWithValidation,
  validateWithGuard,
  withTimeout,
} from '../dist/llm-json.js';

function isSimplePlan(value) {
  return Boolean(
    value &&
    typeof value === 'object' &&
    typeof value.route === 'string' &&
    typeof value.confidence === 'number'
  );
}

test('normalizeJsonCandidate parses JSON strings', () => {
  assert.deepEqual(normalizeJsonCandidate('{"ok":true}'), { ok: true });
  assert.throws(() => normalizeJsonCandidate('{oops'), JsonParseError);
});

test('withTimeout resolves successful promise', async () => {
  const value = await withTimeout(Promise.resolve(42), 50);
  assert.equal(value, 42);
});

test('withTimeout rejects on timeout', async () => {
  await assert.rejects(
    withTimeout(new Promise((resolve) => setTimeout(() => resolve(1), 50)), 5),
    JsonTimeoutError,
  );
});

test('runJsonWithValidation accepts valid object output', async () => {
  const client = new FakeLlmClient({ responses: [{ route: 'retrieve_memory', confidence: 0.9 }] });
  const result = await runJsonWithValidation({
    client,
    call: {
      task: 'route decision',
      model: 'fake',
      systemPrompt: 'Return JSON',
      input: { userMessage: 'install deps' },
    },
    validate: (value) => validateWithGuard(value, isSimplePlan, 'not a valid route plan'),
  });

  assert.deepEqual(result.output, { route: 'retrieve_memory', confidence: 0.9 });
  assert.equal(result.audit.validationStatus, 'valid');
  assert.equal(result.audit.fallbackUsed, false);
});

test('runJsonWithValidation parses valid JSON string output', async () => {
  const client = new FakeLlmClient({ responses: ['{"route":"no_memory","confidence":0.7}'] });
  const result = await runJsonWithValidation({
    client,
    call: {
      task: 'route decision',
      model: 'fake',
      systemPrompt: 'Return JSON',
      input: { userMessage: 'thanks' },
    },
    validate: (value) => validateWithGuard(value, isSimplePlan),
  });

  assert.equal(result.output.route, 'no_memory');
  assert.equal(result.audit.validationStatus, 'valid');
});

test('runJsonWithValidation rejects invalid output without repair/fallback', async () => {
  const client = new FakeLlmClient({ responses: [{ nope: true }] });
  await assert.rejects(
    runJsonWithValidation({
      client,
      call: {
        task: 'route decision',
        model: 'fake',
        systemPrompt: 'Return JSON',
        input: {},
      },
      validate: (value) => validateWithGuard(value, isSimplePlan, 'missing route/confidence'),
    }),
    JsonValidationError,
  );
});

test('runJsonWithValidation repairs invalid output', async () => {
  const client = new FakeLlmClient({ responses: [{ nope: true }] });
  const result = await runJsonWithValidation({
    client,
    call: {
      task: 'route decision',
      model: 'fake',
      systemPrompt: 'Return JSON',
      input: {},
    },
    validate: (value) => validateWithGuard(value, isSimplePlan, 'missing route/confidence'),
    repair: () => ({ route: 'capture_only', confidence: 0.55 }),
  });

  assert.equal(result.output.route, 'capture_only');
  assert.equal(result.audit.validationStatus, 'repaired');
  assert.equal(result.audit.repaired, true);
});

test('runJsonWithValidation uses fallback on timeout', async () => {
  const client = new FakeLlmClient({
    handler: async () => new Promise((resolve) => setTimeout(() => resolve({ route: 'retrieve_memory', confidence: 1 }), 50)),
  });
  const result = await runJsonWithValidation({
    client,
    call: {
      task: 'route decision',
      model: 'fake',
      systemPrompt: 'Return JSON',
      input: {},
    },
    validate: (value) => validateWithGuard(value, isSimplePlan),
    timeoutMs: 5,
    fallback: () => ({ route: 'cached_route', confidence: 0.4 }),
  });

  assert.equal(result.output.route, 'cached_route');
  assert.equal(result.audit.validationStatus, 'fallback');
  assert.equal(result.audit.fallbackUsed, true);
});

test('runJsonWithValidation retries before fallback', async () => {
  const client = new FakeLlmClient({
    handler: (_call, attempt) => (attempt === 1 ? '{bad json' : { route: 'retrieve_memory', confidence: 0.88 }),
  });
  const result = await runJsonWithValidation({
    client,
    call: {
      task: 'route decision',
      model: 'fake',
      systemPrompt: 'Return JSON',
      input: {},
    },
    validate: (value) => validateWithGuard(value, isSimplePlan),
    maxAttempts: 2,
  });

  assert.equal(result.output.route, 'retrieve_memory');
  assert.equal(result.audit.attempts, 2);
});

test('isOllamaLoopbackBaseUrl detects local Ollama endpoints', () => {
  assert.equal(isOllamaLoopbackBaseUrl('http://127.0.0.1:11434/v1'), true);
  assert.equal(isOllamaLoopbackBaseUrl('http://localhost:11434'), true);
  assert.equal(isOllamaLoopbackBaseUrl('https://api.example.com/v1'), false);
});

test('OllamaNativeLlmClient uses native chat with think disabled', async () => {
  let request;
  const client = new OllamaNativeLlmClient({
    baseUrl: 'http://127.0.0.1:11434/v1',
    fetchImpl: async (url, init) => {
      request = { url: String(url), body: JSON.parse(init.body) };
      return new Response(JSON.stringify({ message: { content: '{"route":"no_memory","confidence":0.7}' } }), { status: 200 });
    },
  });

  const raw = await client.runJson({
    task: 'route decision',
    model: 'qwen3.5:35b-a3b-coding-nvfp4',
    systemPrompt: 'Return JSON',
    input: { userMessage: 'thanks' },
    maxTokens: 1200,
    temperature: 0,
  });

  assert.equal(raw, '{"route":"no_memory","confidence":0.7}');
  assert.equal(request.url, 'http://127.0.0.1:11434/api/chat');
  assert.equal(request.body.think, false);
  assert.equal(request.body.stream, false);
  assert.equal(request.body.format, 'json');
  assert.equal(request.body.options.num_predict, 1200);
});
