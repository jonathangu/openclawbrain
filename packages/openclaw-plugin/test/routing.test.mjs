import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizePluginConfig } from '../dist/config.js';
import { LatencyController } from '../dist/latency-controller.js';
import { RouteCache, RouteFn } from '../dist/route-fn.js';

const config = normalizePluginConfig({ enabled: true, mode: 'balanced' });

test('latency controller prefers no extra llm for short low-signal turns', () => {
  const controller = new LatencyController(config);
  const decision = controller.chooseTier({
    agentId: 'main',
    latestUserMessage: 'thanks',
    taskValueEstimate: 'low',
    candidateCount: 0,
    configMode: 'balanced',
  });
  assert.equal(decision.kind, 'no_extra_llm');
  assert.equal(decision.fallback, 'no_memory');
});

test('latency controller escalates high-signal corrections to sync planner', () => {
  const controller = new LatencyController(config);
  const decision = controller.chooseTier({
    agentId: 'main',
    latestUserMessage: 'Actually use pnpm instead of npm',
    hasHighConfidenceCorrectionCandidate: true,
    taskValueEstimate: 'high',
    candidateCount: 2,
    configMode: 'balanced',
  });
  assert.equal(decision.kind, 'sync_memory_planner');
});

test('route fn plans correction retrieval for explicit correction', () => {
  const routeFn = new RouteFn({ config });
  const plan = routeFn.plan({
    agentId: 'main',
    sourceHook: 'before_prompt_build',
    latestUserMessageRedacted: 'Actually, use pnpm instead of npm for this repo',
    toolObservations: [],
    recentInjections: [],
    metadata: { turnType: 'correction' },
  });
  assert.equal(plan.route, 'high_confidence_correction_only');
  assert.equal(plan.shouldRetrieve, true);
  assert.equal(plan.enqueueCapture, true);
  assert.deepEqual(plan.retrievalPlan.memoryTypes, ['correction']);
});

test('route fn plans broader retrieval for implementation planning', () => {
  const routeFn = new RouteFn({ config });
  const plan = routeFn.plan({
    agentId: 'main',
    sourceHook: 'before_prompt_build',
    latestUserMessageRedacted: 'Build the implementation plan file-by-file for OpenClawBrain v0.2',
    toolObservations: [],
    recentInjections: [],
    metadata: { turnType: 'planning' },
  });
  assert.equal(plan.route, 'retrieve_memory');
  assert.equal(plan.shouldRetrieve, true);
  assert.ok(plan.retrievalPlan.memoryTypes.includes('preference'));
  assert.ok(plan.retrievalPlan.memoryTypes.includes('workflow'));
});

test('route fn enqueues capture for explicit remember requests', () => {
  const routeFn = new RouteFn({ config });
  const plan = routeFn.plan({
    agentId: 'main',
    sourceHook: 'before_prompt_build',
    latestUserMessageRedacted: 'Remember that this project uses pnpm for tests',
    toolObservations: [],
    recentInjections: [],
    metadata: { turnType: 'correction' },
  });
  assert.equal(plan.route, 'retrieve_memory');
  assert.equal(plan.shouldRetrieve, true);
  assert.equal(plan.enqueueCapture, true);
});

test('route cache reuses the prior plan for the same fingerprint', () => {
  const cache = new RouteCache();
  const routeFn = new RouteFn({ config, cache });
  const packet = {
    agentId: 'main',
    sourceHook: 'before_prompt_build',
    latestUserMessageRedacted: 'Install dependencies',
    toolObservations: [],
    recentInjections: [],
    metadata: { turnType: 'coding' },
  };
  const first = routeFn.plan(packet);
  const second = routeFn.plan(packet);
  assert.equal(first.route, second.route);
  assert.equal(second.latencyReason, 'cached route plan');
});
