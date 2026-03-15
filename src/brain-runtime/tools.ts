/**
 * Agent tools: brain_teach, brain_status, brain_trace.
 *
 * Follows lossless-claw's tool pattern:
 * - Typebox schemas for input validation
 * - jsonResult() for output formatting
 * - createX() factory functions
 */

import { Type } from "@sinclair/typebox";

// Use LCM's existing tool helpers
const jsonResult = (payload: unknown) => ({
  content: [{ type: "text" as const, text: JSON.stringify(payload, null, 2) }],
  details: payload,
});

// ─── Types ───

interface BrainToolDeps {
  teach: (
    instruction: string,
    kind?: string,
    tags?: string[],
  ) => Promise<{ nodeId: string; packVersion?: number | null }>;
  status: () => Promise<Record<string, unknown>>;
  getTrace: (traceId?: string) => Promise<Record<string, unknown> | null>;
}

// ─── brain_teach ───

const BrainTeachSchema = Type.Object({
  instruction: Type.String({
    description: "What to remember or correct. Examples: 'use gh pr create, not hub', 'for deployment errors, always check CI logs first'",
  }),
  kind: Type.Optional(Type.String({
    description: 'Node kind: "correction", "toolcard", or "workflow". Default: "correction".',
    enum: ["correction", "toolcard", "workflow"],
  })),
  tags: Type.Optional(Type.Array(Type.String(), {
    description: "Tags for filtering and retrieval",
  })),
});

export function createBrainTeachTool(deps: BrainToolDeps) {
  return {
    name: "brain_teach",
    label: "Brain Teach",
    description: "Teach the brain a correction, pattern, or preference. Creates a high-trust human node that will be surfaced in future context.",
    parameters: BrainTeachSchema,
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const instruction = params.instruction as string;
      const kind = (params.kind as string) ?? "correction";
      const tags = (params.tags as string[]) ?? [];

      const result = await deps.teach(instruction, kind, tags);
      return jsonResult({
        success: true,
        nodeId: result.nodeId,
        packVersion: result.packVersion ?? null,
        message: `Brain will remember: "${instruction.slice(0, 80)}${instruction.length > 80 ? "..." : ""}"`,
      });
    },
  };
}

// ─── brain_status ───

const BrainStatusSchema = Type.Object({});

export function createBrainStatusTool(deps: BrainToolDeps) {
  return {
    name: "brain_status",
    label: "Brain Status",
    description: "Show brain health: node/edge counts, pack version, learning stats, recent traces.",
    parameters: BrainStatusSchema,
    async execute() {
      const status = await deps.status();
      return jsonResult(status);
    },
  };
}

// ─── brain_trace ───

const BrainTraceSchema = Type.Object({
  traceId: Type.Optional(Type.String({
    description: "Specific trace ID (bt_...). Default: most recent trace.",
  })),
});

export function createBrainTraceTool(deps: BrainToolDeps) {
  return {
    name: "brain_trace",
    label: "Brain Trace",
    description: "Show the detailed decision trace for the most recent (or specified) brain query. Shows seed ranking, traversal steps, candidate probabilities, fired nodes, and vetoed nodes.",
    parameters: BrainTraceSchema,
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const traceId = params.traceId as string | undefined;
      const trace = await deps.getTrace(traceId);

      if (!trace) {
        return jsonResult({ error: "No trace found", traceId });
      }

      return jsonResult(trace);
    },
  };
}
