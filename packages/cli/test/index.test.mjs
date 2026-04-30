import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { appendProofEvent } from '../../proof-store/src/index.mjs';
import { main, parseArgs } from '../src/index.mjs';

test('parses profile and dry-run options', () => { assert.deepEqual(parseArgs(['enable', '--profile', 'main', '--dry-run']).options, { profile: 'main', dryRun: true }); });

test('status and proof render profile-local state', async () => { const root = await mkdtemp(path.join(tmpdir(), 'ocb-cli-')); try { await appendProofEvent({ profile_id: 'main', decision: 'stay_silent', turn_type: 'direct-answer', reason: 'direct answer' }, { activationRoot: root, profileId: 'main' }); const output = []; await main(['status', '--profile', 'main', '--activation-root', root], { log: (text) => output.push(text) }); await main(['proof', '--profile', 'main', '--activation-root', root], { log: (text) => output.push(text) }); assert.match(output.join('\n'), /OpenClawBrain status/); assert.match(output.join('\n'), /stay_silent: 1/); } finally { await rm(root, { recursive: true, force: true }); } });

test('enable dry-run reports plugin config path', async () => { const root = await mkdtemp(path.join(tmpdir(), 'ocb-cli-')); try { const output = []; await main(['enable', '--profile', 'main', '--activation-root', root, '--dry-run'], { log: (text) => output.push(text) }); assert.match(output.join('\n'), /plugins\.entries\.openclawbrain\.config/); } finally { await rm(root, { recursive: true, force: true }); } });
