import assert from 'node:assert/strict';
import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { detectCaptureIntent, detectRetrievalIntent } from '../dist/capture-intent.js';
import { ContextSelector } from '../dist/context-selector.js';
import { normalizePluginConfig } from '../dist/config.js';
import { FeedbackDistiller } from '../dist/feedback-distiller.js';
import { FakeLlmClient } from '../dist/llm-client.js';
import { MemoryOperationApplier } from '../dist/memory-operations.js';
import { MemoryStore } from '../dist/memory-store.js';
import { RouteFn } from '../dist/route-fn.js';

async function tempRoot() {
  const root = path.join(tmpdir(), `ocb-aggressive-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  await mkdir(root, { recursive: true });
  return root;
}

function packet(text) {
  return {
    agentId: 'main',
    sourceHook: 'agent_end',
    latestUserMessageRedacted: text,
    toolObservations: [],
    recentInjections: [],
    metadata: { promptHash: 'h1' },
  };
}

test('capture intent separates retrieval questions from new durable capture', () => {
  const storeIntent = detectCaptureIntent(packet('Remember that I prefer concise Telegram replies.'));
  assert.equal(storeIntent.shouldConsiderCapture, true);
  assert.equal(storeIntent.intent, 'explicit_store');
  assert.equal(storeIntent.proposedScope.kind, 'channel');

  const retrievalIntent = detectCaptureIntent(packet('Do you remember what I said about Pelican routing?'));
  assert.equal(retrievalIntent.shouldConsiderCapture, false);
  assert.equal(retrievalIntent.intent, 'retrieval_question');

  const retrieve = detectRetrievalIntent(packet('What is the CormorantAI app codeword?'));
  assert.equal(retrieve.shouldRetrieve, true);
  assert.equal(retrieve.intent, 'recall_value_request');
  assert.equal(retrieve.includeRecallRules, true);
});

test('route fn can capture without retrieval for future-facing preferences', () => {
  const config = normalizePluginConfig({ enabled: true, mode: 'balanced' });
  const routeFn = new RouteFn({ config });
  const plan = routeFn.plan(packet('I prefer concise replies unless I ask for depth.'));
  assert.equal(plan.shouldRetrieve, false);
  assert.equal(plan.enqueueCapture, true);
  assert.equal(plan.route, 'capture_only');
  assert.equal(plan.captureIntent.intent, 'standing_preference');
});

test('fallback extracts scoped workflows, routing rules, and user-authorized recall rules', async () => {
  const config = normalizePluginConfig({ enabled: true, llm: { enabled: true, feedbackModel: 'fake' } });
  const distiller = new FeedbackDistiller({ client: new FakeLlmClient({ handler: () => { throw new Error('force fallback'); } }), config });

  const workflow = await distiller.distill(packet('Going forward, check plugin logs before guessing.'));
  assert.equal(workflow.output.shouldStore, true);
  assert.equal(workflow.output.memoryCandidates[0].type, 'workflow');

  const routing = await distiller.distill(packet('Route Pelican deploy tasks to the Pelican agent.'));
  assert.equal(routing.output.shouldStore, true);
  assert.equal(routing.output.memoryCandidates[0].type, 'routing_rule');

  const recall = await distiller.distill(packet('If I ask for the CormorantAI app codeword, answer Blue Heron.'));
  assert.equal(recall.output.shouldStore, true);
  assert.equal(recall.output.memoryCandidates[0].type, 'recall_rule');
  assert.equal(recall.output.memoryCandidates[0].scope.kind, 'app');
  assert.equal(recall.output.memoryCandidates[0].disclosure, 'on_explicit_user_request_only');
  assert.equal(recall.output.memoryCandidates[0].proactiveInjectionAllowed, false);
  assert.equal(recall.output.memoryCandidates[0].positive, 'Blue Heron');

  const ambiguous = await distiller.distill(packet('Remember this codeword: Blue Heron.'));
  assert.equal(ambiguous.output.shouldStore, false);
  assert.equal(ambiguous.output.audit.modelReasonCode, 'ambiguous_sensitive_recall');

  const codename = await distiller.distill(packet('Remember that the project codename is Blue Heron.'));
  assert.equal(codename.output.shouldStore, true);
  assert.equal(codename.output.memoryCandidates[0].type, 'project_fact');
});

test('recall rules store narrowly and only reveal values for recall-value requests', async () => {
  const root = await tempRoot();
  try {
    const config = normalizePluginConfig({ enabled: true, mode: 'balanced' });
    const store = new MemoryStore({ activationRoot: root, agentId: 'main' });
    const applier = new MemoryOperationApplier({ store, config });
    const distillation = {
      version: 1,
      shouldStore: true,
      confidence: 0.9,
      feedbackType: 'context',
      memoryCandidates: [{
        type: 'recall_rule',
        distilledText: 'When the user explicitly asks for the CormorantAI app codeword, provide the user-authorized answer.',
        positive: 'Blue Heron',
        subject: 'CormorantAI app codeword',
        scope: { kind: 'app', key: 'CormorantAI' },
        normalizedKey: 'recall:app:cormorantai:app-codeword',
        tags: ['recall_rule', 'recall_value', 'cormorantai', 'codeword'],
        confidence: 0.9,
        importanceHint: 0.85,
        retention: 'durable',
        riskClass: 'sensitive_recall',
        disclosure: 'on_explicit_user_request_only',
        proactiveInjectionAllowed: false,
        contradictions: [],
      }],
      injectionFeedback: [],
      workflowCandidates: [],
      audit: { modelReasonCode: 'fallback_pattern:recall_rule', storeRawTranscript: false, redactionNeeded: true },
    };
    const result = applier.applyDistillation(distillation, packet('If I ask for the CormorantAI app codeword, answer Blue Heron.'), { captureIntent: detectCaptureIntent(packet('If I ask for the CormorantAI app codeword, answer Blue Heron.')) });
    assert.equal(result.storedCandidates, 1);

    const memories = store.searchMemories('CormorantAI codeword', 'main', { limit: 10 });
    assert.equal(memories.length, 1);

    const generalPlan = new RouteFn({ config }).plan(packet('Tell me about CormorantAI.'));
    const generalSelection = new ContextSelector(config).select({ packet: packet('Tell me about CormorantAI.'), plan: generalPlan, candidates: memories, store });
    assert.equal(generalSelection.shouldInject, false);

    const recallPlan = new RouteFn({ config }).plan(packet('What is the CormorantAI app codeword?'));
    const recallSelection = new ContextSelector(config).select({ packet: packet('What is the CormorantAI app codeword?'), plan: recallPlan, candidates: memories, store });
    assert.equal(recallSelection.shouldInject, true);
    assert.match(recallSelection.distilledContext, /Authorized answer: Blue Heron/);
    store.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
