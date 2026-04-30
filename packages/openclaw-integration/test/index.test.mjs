import test from 'node:test';
import assert from 'node:assert/strict';
import integration, { buildOpenClawBrainConfigPatch, normalizePluginConfig } from '../src/index.mjs';

test('openclaw integration exports the native plugin entry', () => {
  assert.equal(integration.id, 'openclawbrain');
  assert.equal(typeof integration.register, 'function');
});

test('openclaw integration keeps config under native plugin entry', () => {
  const patch = buildOpenClawBrainConfigPatch({ profile: 'family', activationRoot: '/tmp/ocb-family' });
  assert.equal(patch.openclawbrain, undefined);
  assert.equal(patch.plugins.entries.openclawbrain.config.activationRoot, '/tmp/ocb-family');
  assert.deepEqual(patch.plugins.entries.openclawbrain.config.scopes.agents, ['family']);
});

test('openclaw integration forces conservative privacy defaults', () => {
  const config = normalizePluginConfig({ mode: 'active', rawTranscriptUpload: true });
  assert.equal(config.mode, 'active');
  assert.equal(config.rawTranscriptUpload, false);
});
