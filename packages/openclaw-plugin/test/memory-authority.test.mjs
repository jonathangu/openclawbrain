import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { detectCaptureIntent } from '../dist/capture-intent.js';
import { ContextSelector } from '../dist/context-selector.js';
import { normalizePluginConfig } from '../dist/config.js';
import { MemoryOperationApplier } from '../dist/memory-operations.js';
import { MemoryStore } from '../dist/memory-store.js';
import { RouteFn } from '../dist/route-fn.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-authority-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

function packet(text, extra = {}) {
  return {
    agentId: 'main',
    sourceHook: 'before_prompt_build',
    latestUserMessageRedacted: text,
    recentAssistantMessage: '',
    toolObservations: [],
    recentInjections: [],
    metadata: { promptHash: `hash:${text}`, repo: 'openclawbrain', ...(extra.metadata || {}) },
    ...extra,
  };
}

function candidate(overrides = {}) {
  return {
    type: 'workflow',
    distilledText: 'Use pnpm install before running tests in this repo.',
    subject: 'OpenClawBrain package manager',
    scope: { kind: 'repo', key: 'openclawbrain' },
    normalizedKey: 'repo:openclawbrain:package-manager',
    tags: ['workflow', 'package-manager'],
    confidence: 0.9,
    importanceHint: 0.8,
    retention: 'durable',
    contradictions: [],
    ...overrides,
  };
}

test('authority resolver marks stale environment-owned workflow for verification before use', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, mode: 'balanced', activationRoot: root });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = store.insertMemory({
      agentId: 'main',
      type: 'workflow',
      content: 'Use pnpm install before running tests in this repo.',
      scopeKind: 'repo',
      scopeKey: 'openclawbrain',
      normalizedKey: 'repo:openclawbrain:package-manager',
      tags: ['workflow', 'package-manager'],
      importance: 0.8,
      freshness: 0.45,
      confidence: 0.9,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });
    store.patchMemoryValidity(memory.id, {
      temporalValidity: 'stale',
      validationStrategy: 'environment_check',
      revalidateAfter: new Date(Date.now() - 86400000).toISOString(),
      stateReason: 'test_stale',
    });

    const turn = packet('Install dependencies and run tests for OpenClawBrain.');
    const plan = new RouteFn({ config, store }).plan(turn);
    const selection = new ContextSelector(config).select({ packet: turn, plan, candidates: [memory], store });

    assert.equal(selection.shouldInject, true);
    assert.equal(selection.selected[0].authorityDecision, 'verify_before_use');
    assert.match(selection.distilledContext, /Verify before using: Workflow: Use pnpm install/);
    assert.equal(selection.audit.authority[0].requiredAction, 'verify_environment');
    assert.ok(store.listMemoryAuthorityEvents('main', 20, memory.id).some((event) => event.eventType === 'verification_requested'));
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('current task instruction overrides a stale global preference without deleting it', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, mode: 'balanced', activationRoot: root });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const memory = store.insertMemory({
      agentId: 'main',
      type: 'preference',
      content: 'User prefers concise answers by default.',
      scopeKind: 'global_user',
      normalizedKey: 'preference:answer-length',
      tags: ['style'],
      importance: 0.7,
      freshness: 1,
      confidence: 0.9,
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
    });

    const turn = packet('Give me a deep, detailed critique of the memory authority design.');
    const plan = new RouteFn({ config, store }).plan(turn);
    const selection = new ContextSelector(config).select({ packet: turn, plan, candidates: [memory], store });

    assert.equal(selection.shouldInject, false);
    assert.equal(selection.omitted[0].reason, 'current_instruction_override');
    assert.equal(Boolean(store.getMemory(memory.id).deletedAt), false);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('same normalized key with changed value creates lineage instead of overwriting history', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, mode: 'balanced', activationRoot: root });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const applier = new MemoryOperationApplier({ store, config });
    const firstPacket = packet('Going forward, use npm in this repo.');
    const secondPacket = packet('Actually use pnpm in this repo instead.');

    const first = applier.applyDistillation({
      version: 1,
      shouldStore: true,
      confidence: 0.9,
      feedbackType: 'workflow',
      memoryCandidates: [candidate({ distilledText: 'Use npm in this repo.' })],
      injectionFeedback: [],
      workflowCandidates: [],
      audit: { modelReasonCode: 'test', storeRawTranscript: false, redactionNeeded: false },
    }, firstPacket, { captureIntent: detectCaptureIntent(firstPacket) });
    const oldId = first.memoryIds[0];

    const second = applier.applyDistillation({
      version: 1,
      shouldStore: true,
      confidence: 0.92,
      feedbackType: 'workflow',
      memoryCandidates: [candidate({ distilledText: 'Use pnpm in this repo.' })],
      injectionFeedback: [],
      workflowCandidates: [],
      audit: { modelReasonCode: 'test', storeRawTranscript: false, redactionNeeded: false },
    }, secondPacket, { captureIntent: detectCaptureIntent(secondPacket) });
    const newId = second.memoryIds[0];

    const oldMemory = store.getMemory(oldId);
    const newMemory = store.getMemory(newId);
    assert.equal(oldMemory.content, 'Use npm in this repo.');
    assert.equal(oldMemory.supersededBy, newId);
    assert.equal(newMemory.content, 'Use pnpm in this repo.');
    assert.match(newMemory.normalizedKey, /^repo:openclawbrain:package-manager:rev:/);
    assert.equal(store.getMemoryValidity(oldId).behavioralAvailability, 'never_use');
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('tombstoned memory blocks future recapture for the same normalized key', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, mode: 'balanced', activationRoot: root });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const applier = new MemoryOperationApplier({ store, config });
    const rememberPacket = packet('If I ask for the CormorantAI app codeword, answer Blue Heron.', { metadata: { app: 'CormorantAI' } });
    const recallCandidate = candidate({
      type: 'recall_rule',
      distilledText: 'When the user explicitly asks for the CormorantAI app codeword, provide the user-authorized answer.',
      positive: 'Blue Heron',
      subject: 'CormorantAI app codeword',
      scope: { kind: 'app', key: 'CormorantAI' },
      normalizedKey: 'recall:app:cormorantai:app-codeword',
      tags: ['recall_rule', 'recall_value', 'cormorantai', 'codeword'],
      riskClass: 'sensitive_recall',
      disclosure: 'on_explicit_user_request_only',
      proactiveInjectionAllowed: false,
    });

    const stored = applier.applyDistillation({
      version: 1,
      shouldStore: true,
      confidence: 0.9,
      feedbackType: 'context',
      memoryCandidates: [recallCandidate],
      injectionFeedback: [],
      workflowCandidates: [],
      audit: { modelReasonCode: 'test', storeRawTranscript: false, redactionNeeded: true },
    }, rememberPacket, { captureIntent: detectCaptureIntent(rememberPacket) });
    assert.equal(stored.storedCandidates, 1);

    const deletePacket = packet('Do not store the CormorantAI app codeword anymore.', { metadata: { app: 'CormorantAI' } });
    const deleted = applier.applyDistillation({
      version: 1,
      shouldStore: false,
      confidence: 0.95,
      feedbackType: 'delete_or_suppress',
      memoryCandidates: [],
      injectionFeedback: [],
      workflowCandidates: [],
      audit: { modelReasonCode: 'test_delete', storeRawTranscript: false, redactionNeeded: true },
    }, deletePacket, { captureIntent: detectCaptureIntent(deletePacket) });
    assert.equal(deleted.deletedOrSuppressed, 1);
    assert.equal(store.getMemoryValidity(stored.memoryIds[0]).retentionState, 'tombstoned');

    const recapture = applier.applyDistillation({
      version: 1,
      shouldStore: true,
      confidence: 0.9,
      feedbackType: 'context',
      memoryCandidates: [recallCandidate],
      injectionFeedback: [],
      workflowCandidates: [],
      audit: { modelReasonCode: 'test_recapture', storeRawTranscript: false, redactionNeeded: true },
    }, rememberPacket, { captureIntent: detectCaptureIntent(rememberPacket) });
    assert.equal(recapture.storedCandidates, 0);
    assert.ok(recapture.rejectionReasons.includes('tombstoned_memory_blocked'));
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
