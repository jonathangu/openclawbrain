/**
 * SDK compatibility shims for OpenClawBrain.
 * 
 * The OpenClaw plugin-sdk has evolved and some types that were previously
 * exported are no longer available. This module provides shims to maintain
 * compatibility with the existing OpenClawBrain implementation.
 * 
 * These shims reflect the types OpenClawBrain expects from the SDK based on
 * its context-engine plugin interface.
 */

import type { PluginRuntime, RuntimeLogger } from "openclaw/plugin-sdk";
import type {
  BrainDropReason,
  BrainDropStage,
  BrainFitStrategy,
  BrainFittingDropReason,
  BrainInterruptionMetadata,
  BrainInterruptionStage,
} from "./brain-core/types.js";

// Re-export what's actually available from the SDK
export { PluginRuntime, type RuntimeLogger } from "openclaw/plugin-sdk";

// ── Shim types for removed/moved SDK exports ─────────────────────────────────

/** Brain assembly decision metadata */
export interface BrainDecisionMetadata {
  mode?: string;
  episodeId?: string | null;
  traceId?: string | null;
  footer?: string | null;
  interruption?: BrainInterruptionMetadata | null;
  queryInterrupted?: boolean | null;
  interruptionStage?: BrainInterruptionStage | null;
  interruptionReason?: string | null;
  servedPartial?: boolean | null;
  compileElapsedMs?: number | null;
  compileDeadlineMs?: number | null;
  compileDeadlineHit?: boolean | null;
  brainDropReason?: BrainDropReason | null;
  brainDropStage?: BrainDropStage | null;
  budgetFraction?: number | null;
  maxContextChars?: number | null;
  queryBudgetChars?: number | null;
  injectedChars?: number | null;
  droppedChars?: number | null;
  contextClipped?: boolean;
  fitStrategy?: BrainFitStrategy | null;
  retrievedNodeCount?: number | null;
  fittedNodeCount?: number | null;
  droppedNodeCount?: number | null;
  fittingDropReasons?: Partial<Record<BrainFittingDropReason, number>> | null;
}

/** Result from assemble - matches what the code expects */
export interface AssembleResult {
	messages: ContextEngineMessage[];
	truncated?: boolean;
	estimatedTokens?: number;
	brainDecision?: BrainDecisionMetadata;
}

/** Assemble result with optional system prompt addition */
export type AssembleResultWithSystemPrompt = AssembleResult & { systemPromptAddition?: string };

/** Result from bootstrap */
export interface BootstrapResult {
	ok?: boolean;
	bootstrapped?: boolean;
	importedMessages?: number;
	reason?: string;
}

/** Result from compact */
export interface CompactResult {
	ok?: boolean;
	compacted?: boolean;
	reason?: string;
	result?: {
		tokensBefore?: number;
		tokensAfter?: number;
		details?: Record<string, unknown>;
	};
}

/** Result from ingest */
export interface IngestResult {
	ok?: boolean;
	ingested?: boolean;
}

/** Result from ingest batch */
export interface IngestBatchResult {
	ok?: boolean;
	ingestedCount?: number;
}

/** Shim for removed ContextEngine type - represents the plugin's context assembly interface */
export interface ContextEngine {
	readonly info: ContextEngineInfo;
	bootstrap(params: { sessionId: string; sessionFile: string }): Promise<BootstrapResult>;
	assemble(params: { sessionId: string; messages: unknown[]; tokenBudget?: number; maxContextChars?: number }): Promise<AssembleResult>;
	ingest(params: { sessionId: string; message: unknown; isHeartbeat?: boolean; brainEpisodeId?: string }): Promise<IngestResult>;
	compact(params: { sessionId: string; sessionFile: string; tokenBudget?: number; currentTokenCount?: number; compactionTarget?: string; customInstructions?: string; legacyParams?: Record<string, unknown>; force?: boolean }): Promise<CompactResult>;
}

/** Engine identification */
export interface ContextEngineInfo {
	id: string;
	name: string;
	version?: string;
	ownsCompaction?: boolean;
	rollback?: boolean;
}

/** Bootstrap configuration */
export interface ContextEngineConfig {
	stateDir: string;
	workspaceDir: string;
	env: Record<string, string>;
}

/** Result from bootstrap */
export interface ContextEngineBootstrapResult {
	ok: true;
}

/** Context for assembly */
export interface ContextEngineAssembleContext {
	sessionKey: string;
	messages: ContextEngineMessage[];
	tokenBudget?: number;
	maxContextChars?: number;
}

/** Message shape for context engine */
export interface ContextEngineMessage {
	role: string;
	content?: unknown;
	name?: string;
	toolCallId?: string;
}

/** Batch for ingestion */
export interface ContextEngineIngestBatch {
	sessionKey: string;
	messages: ContextEngineMessage[];
}

/** Result from ingest */
export interface ContextEngineIngestResult {
	ok: true;
}

/** Result from compact */
export interface ContextEngineCompactResult {
	ok: true;
}

/** Factory for context engine (shim for removed registerContextEngine) */
export type ContextEngineFactory = (runtime: PluginRuntime) => ContextEngine;

// ── Subagent types (shims for removed SDK exports) ────────────────────────────

export type SubagentEndReason = "completed" | "error" | "cancelled" | "deleted" | "released" | "swept";

export interface SubagentSpawnPreparation {
	sessionKey?: string;
	prompt?: string;
	tools?: unknown[];
	rollback?: boolean | (() => void);
}

// ── OpenClawPluginToolContext (shim for removed SDK export) ─────────────────

export interface OpenClawPluginToolContext {
	sessionKey: string;
	stateDir: string;
	workspaceDir: string;
	runtime: PluginRuntime;
}

// ── AgentMessage helper type ───────────────────────────────────────────────

/** Agent message shape used in the codebase */
export interface AgentMessage {
	role: string;
	content?: unknown;
	name?: string;
	toolCallId?: string;
	toolName?: string;
	isError?: boolean;
	timestamp?: number;
}
