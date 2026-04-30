import assert from 'node:assert/strict';
import test from 'node:test';
import { classifyRedactedTurn, decideOpenClawBrainIntervention, validateRuntimePolicyInput } from '../src/index.mjs';

test('selected product policy stays silent for direct-answer turns', () => {
  const decision = decideOpenClawBrainIntervention({ profileId: 'main', runtimeMode: 'conservative', redactedTurn: { turnId: 't1', turnType: 'direct-answer', summary: 'What is 12% of 80?' }, candidateMemories: [{ id: 'm1', kind: 'context', text: 'Unneeded context', relevance: 1 }] });
  assert.equal(decision.kind, 'stay_silent');
  assert.equal(decision.proof.raw_text_stored, false);
});

test('selected product policy injects correction only for stale-memory conflict', () => {
  const decision = decideOpenClawBrainIntervention({ profileId: 'family', openclawProfile: 'FamilyProfile', agentId: 'family-agent', sessionKey: 'session-redacted', runtimeMode: 'conservative', redactedTurn: { turnId: 't2', turnType: 'stale-memory-conflict', summary: 'Ambiguous family email task' }, candidateMemories: [{ id: 'correction-redacted', kind: 'correction', text: 'avoid using work email for family-related tasks', relevance: 0.99 }] });
  assert.equal(decision.kind, 'correction_only');
  assert.match(decision.message, /Relevant user correction/);
  assert.equal(decision.proof.profile_id, 'family');
  assert.equal(decision.proof.openclaw_profile, 'FamilyProfile');
  assert.equal(decision.proof.agent_id, 'family-agent');
  assert.match(decision.proof.session_key_hash, /^sha256:/);
  assert.equal(decision.proof.contains_real_user_data, false);
});

test('selected product policy injects bounded context for continuation', () => {
  const decision = decideOpenClawBrainIntervention({ profileId: 'bountiful', runtimeMode: 'conservative', redactedTurn: { turnId: 't3', summary: 'continue' }, candidateMemories: [{ id: 'state-redacted', kind: 'context', text: 'previous task was drafting onboarding; next section is proof command', relevance: 0.9 }] });
  assert.equal(decision.kind, 'full_context');
  assert.match(decision.context, /Continuation context/);
});

test('proof-only records proof without injecting', () => {
  const decision = decideOpenClawBrainIntervention({ profileId: 'main', runtimeMode: 'proof-only', redactedTurn: { turnId: 't4', turnType: 'tool-heavy', summary: 'verify latest status' }, candidateMemories: [{ id: 'tool-redacted', kind: 'context', text: 'prefer read-only verification', relevance: 0.8 }] });
  assert.equal(decision.kind, 'proof_only');
  assert.equal(decision.proof.decision, 'proof_only');
});

test('rejects raw fields and secret-like values', () => {
  const issues = validateRuntimePolicyInput({ profileId: 'main', runtimeMode: 'conservative', redactedTurn: { rawText: 'private', summary: 'hello' }, candidateMemories: [{ id: 'm', text: 'sk-abc1234567890' }] });
  assert.match(issues.join('\n'), /raw\/unredacted/);
  assert.match(issues.join('\n'), /secret-like/);
});

test('classifies common redacted turns conservatively', () => {
  assert.equal(classifyRedactedTurn({ summary: 'continue' }), 'continuation');
  assert.equal(classifyRedactedTurn({ summary: 'What is 12% of 80?' }), 'direct-answer');
  assert.equal(classifyRedactedTurn({ summary: 'please verify current package status' }), 'tool-heavy');
});
