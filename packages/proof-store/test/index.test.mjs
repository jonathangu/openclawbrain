import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { appendProofEvent, readProofEvents, renderProof, renderStatus } from '../src/index.mjs';

test('writes redacted proof events and status per profile', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'ocb-proof-'));
  try { await appendProofEvent({ profile_id: 'main', decision: 'correction_only', turn_type: 'stale-memory-conflict', reason: 'user correction wins' }, { activationRoot: root, profileId: 'main' }); const events = await readProofEvents({ activationRoot: root, profileId: 'main' }); assert.equal(events.length, 1); assert.equal(events[0].raw_text_stored, false); assert.match(await renderStatus({ activationRoot: root, profileId: 'main' }), /Runtime adapter: connected/); assert.match(await renderProof({ activationRoot: root, profileId: 'main' }), /correction_only: 1/); } finally { await rm(root, { recursive: true, force: true }); }
});

test('rejects unsafe proof events', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'ocb-proof-'));
  try { await assert.rejects(() => appendProofEvent({ profile_id: 'main', rawText: 'private' }, { activationRoot: root }), /unsafe proof event rejected/); } finally { await rm(root, { recursive: true, force: true }); }
});
