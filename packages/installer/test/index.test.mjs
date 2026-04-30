import assert from 'node:assert/strict';
import test from 'node:test';
import { buildOpenClawBrainConfigPatch, enableOpenClawBrain } from '../src/index.mjs';

test('builds plugin-scoped config patch, not root openclawbrain config', () => { const patch = buildOpenClawBrainConfigPatch({ profile: 'main', activationRoot: '/tmp/ocb-main' }); assert.equal(patch.openclawbrain, undefined); assert.equal(patch.plugins.entries.openclawbrain.config.rawTranscriptUpload, false); assert.equal(patch.plugins.entries.openclawbrain.config.openclawProfile, 'main'); assert.deepEqual(patch.plugins.entries.openclawbrain.config.scopes, { agents: ['main'], sessionKeys: [] }); assert.equal(patch.plugins.entries.openclawbrain.hooks.allowPromptInjection, true); });

test('enable dry-run uses documented plugin/config paths and restarts gateway', async () => { const results = await enableOpenClawBrain({ profile: 'main', activationRoot: '/tmp/ocb-main', dryRun: true }); assert.ok(results.every((result) => result.ok)); const commands = results.map((r) => r.command).join('\n'); assert.match(commands, /plugins enable openclawbrain/); assert.match(commands, /plugins\.entries\.openclawbrain\.config\.mode/); assert.match(commands, /plugins\.entries\.openclawbrain\.config\.scopes/); assert.match(commands, /gateway restart/); });
