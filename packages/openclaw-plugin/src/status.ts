import { PLUGIN_ID, PLUGIN_VERSION } from './config.js';

export function buildStatus(config: any, details: any = {}) {
  return {
    ok: true,
    plugin: PLUGIN_ID,
    pluginVersion: PLUGIN_VERSION,
    enabled: config.enabled === true,
    mode: config.mode,
    agentId: details.agentId || 'main',
    activationRoot: config.activationRoot,
    proofEvents: config.proofEvents ? 'writing' : 'disabled',
    rawTranscriptUpload: false,
    lastDecisionKind: details.lastDecisionKind || details.lastDecision || 'none',
    lastDecisionAt: details.lastDecisionAt || new Date().toISOString(),
    memory: details.memory || undefined,
    routing: details.routing || undefined,
    learning: details.learning || undefined,
    latency: details.latency || undefined,
    nativeSqlite: details.nativeSqlite || undefined,
  };
}
