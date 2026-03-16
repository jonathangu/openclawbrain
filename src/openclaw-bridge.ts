/**
 * Compatibility bridge for plugin-sdk context-engine symbols.
 *
 * This module exports stable plugin-sdk surface area plus shims for
 * types that were previously exported but are now removed.
 */

export type {
  ContextEngine,
  ContextEngineInfo,
  AssembleResult,
  AssembleResultWithSystemPrompt,
  CompactResult,
  IngestResult,
  IngestBatchResult,
  BootstrapResult,
  SubagentSpawnPreparation,
  SubagentEndReason,
  ContextEngineFactory,
  OpenClawPluginToolContext,
  AgentMessage,
} from "./openclaw-sdk-compat.js";

export { type PluginRuntime, type RuntimeLogger } from "openclaw/plugin-sdk";
