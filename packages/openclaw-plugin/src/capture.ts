import { hashText, latestUserTextFromEvent, redactText, safeString } from './redact.js';

export interface TurnEventPacket {
  agentId: string;
  sessionId?: string;
  sessionKey?: string;
  turnId?: string;
  runId?: string;
  sourceHook: string;
  latestUserMessage: string;
  redactedLatestUserMessage: string;
  recentAssistantMessage?: string;
  toolObservations: Array<{
    toolName: string;
    ok: boolean;
    durationMs?: number;
    argsSummary?: string;
    resultSummary?: string;
    errorClass?: string;
  }>;
  recentInjections: Array<{ injectionId: string; memoryId: string; outcome?: string }>;
  metadata: Record<string, unknown>;
}

export class CaptureOrchestrator {
  fromBeforePromptBuild(event: any = {}, config: any = {}) {
    return basePacket(event, config, 'before_prompt_build');
  }

  fromAgentEnd(event: any = {}, config: any = {}) {
    return basePacket(event, config, 'agent_end');
  }

  fromAfterToolCall(event: any = {}, config: any = {}) {
    const packet = basePacket(event, config, 'after_tool_call');
    packet.toolObservations = [sanitizeToolEvent(event, config)];
    return packet;
  }
}

export function sanitizeToolEvent(event: any = {}, config: any = {}) {
  return {
    toolName: safeString(event.toolName ?? event.tool_name ?? event.name ?? 'unknown_tool') || 'unknown_tool',
    ok: event.ok !== false && event.success !== false,
    durationMs: Number.isFinite(Number(event.durationMs ?? event.duration_ms)) ? Number(event.durationMs ?? event.duration_ms) : undefined,
    argsSummary: redactText(JSON.stringify(event.args ?? event.arguments ?? null), config.maxContextChars || 3000),
    resultSummary: redactText(JSON.stringify(event.result ?? event.output ?? null), config.maxContextChars || 3000),
    errorClass: safeString(event.error?.name ?? event.errorClass ?? event.error_class ?? ''),
  };
}

function basePacket(event: any = {}, config: any = {}, sourceHook: string): TurnEventPacket {
  const latestUserMessage = latestUserTextFromEvent(event);
  const ctx = event.ctx || {};
  return {
    agentId: safeString(ctx.agentId ?? event.agentId ?? event.agent_id ?? 'main') || 'main',
    sessionId: safeString(ctx.sessionId ?? event.sessionId ?? event.session_id ?? ''),
    sessionKey: safeString(ctx.sessionKey ?? event.sessionKey ?? event.session_key ?? ''),
    turnId: safeString(event.turnId ?? event.turn_id ?? ctx.turnId ?? ''),
    runId: safeString(ctx.runId ?? event.runId ?? event.run_id ?? ''),
    sourceHook,
    latestUserMessage,
    redactedLatestUserMessage: redactText(latestUserMessage, config.maxContextChars || 3000),
    recentAssistantMessage: redactText(safeString(event.assistantMessage ?? event.assistant_message ?? ''), config.maxContextChars || 3000),
    toolObservations: [],
    recentInjections: Array.isArray(event.recentInjections)
      ? event.recentInjections.map((it: any) => ({
          injectionId: safeString(it.injectionId ?? it.injection_id ?? ''),
          memoryId: safeString(it.memoryId ?? it.memory_id ?? ''),
          outcome: safeString(it.outcome ?? ''),
        })).filter((it: any) => it.injectionId && it.memoryId)
      : [],
    metadata: {
      promptHash: hashText(latestUserMessage),
      turnType: safeString(event.turnType ?? event.turn_type ?? ''),
      profileId: safeString(ctx.profile ?? event.profile ?? ''),
    },
  };
}
