import { PLUGIN_ID, PLUGIN_VERSION } from './config.js';
export function buildStatus(config, details = {}) {
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
        lastDecisionAt: details.lastDecisionAt || new Date().toISOString()
    };
}
