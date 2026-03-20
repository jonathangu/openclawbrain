/**
 * Off-path async teacher for traced route-decision evaluation.
 *
 * CRITICAL RULE: Teacher sees ONLY the persisted trace slice.
 * It evaluates the routing decision, not the overall task outcome.
 * No cheating with extra context.
 */

import type { BrainGraph } from "./graph.js";
import type {
  DecisionRouteTrace,
  DecisionTrace,
  DecisionTraceInjectedNodeSummary,
} from "./types.js";

export type BrainTeacherCompletion = (params: {
  provider?: string;
  model: string;
  apiKey?: string;
  messages: Array<{ role: string; content: unknown }>;
  system?: string;
  maxTokens: number;
  temperature?: number;
}) => Promise<{ content?: Array<{ text?: string }> }>;

export type BrainTeacherResolveModel = () => { provider: string; model: string };
export type BrainTeacherGetApiKey = (provider: string, model: string) => Promise<string | undefined>;

export interface TeacherLabelInputV1 {
  version: 1;
  traceId: string;
  episodeId: string | null;
  queryText: string;
  routeDecision: {
    requestDigest: string;
    conversationId: number | null;
    activePackId: string | null;
    routerIdentity: string;
    candidateNodeIds: string[];
    selectedNodeIds: string[];
    selectedPathNodeIds: string[];
    sourceSummary: DecisionRouteTrace["sourceSummary"];
    selectionMetadata: DecisionRouteTrace["selectionMetadata"];
  };
  selectedContext: DecisionTraceInjectedNodeSummary[];
}

export interface TeacherLabelResultV1 {
  version: 1;
  traceId: string;
  episodeId: string | null;
  requestDigest: string;
  score: number;
  reason: string;
  input: TeacherLabelInputV1;
}

const TEACHER_SYSTEM_PROMPT =
  "You are evaluating a traced context routing decision. Score the selected context shown in the trace slice for relevance, completeness, and conciseness. Do not assume access to hidden router state or unseen candidate content. Return ONLY a JSON object: {\"score\": <number from -1.0 to 1.0>, \"reason\": \"<brief explanation>\"}";

function cloneInjectedSummary(summary: DecisionTraceInjectedNodeSummary): DecisionTraceInjectedNodeSummary {
  return {
    nodeId: summary.nodeId,
    kind: summary.kind,
    trust: summary.trust,
    sourceUri: summary.sourceUri,
    tags: [...summary.tags],
    tokenCount: summary.tokenCount,
    contentPreview: summary.contentPreview,
  };
}

export function materializeTeacherLabelInput(trace: DecisionTrace): TeacherLabelInputV1 | null {
  const routeTrace = trace.routeTrace ?? null;
  if (!routeTrace) {
    return null;
  }

  if (typeof trace.queryText !== "string" || trace.queryText.trim().length === 0) {
    return null;
  }

  if (routeTrace.selectedNodeIds.length === 0 || routeTrace.injectedNodeSummaries.length === 0) {
    return null;
  }

  return {
    version: 1,
    traceId: trace.id,
    episodeId: trace.episodeId,
    queryText: trace.queryText,
    routeDecision: {
      requestDigest: routeTrace.requestDigest,
      conversationId: routeTrace.conversationId,
      activePackId: routeTrace.activePackId,
      routerIdentity: routeTrace.routerIdentity,
      candidateNodeIds: [...routeTrace.candidateNodeIds],
      selectedNodeIds: [...routeTrace.selectedNodeIds],
      selectedPathNodeIds: [...routeTrace.selectedPathNodeIds],
      sourceSummary: {
        injectedCount: routeTrace.sourceSummary.injectedCount,
        kinds: { ...routeTrace.sourceSummary.kinds },
        trusts: { ...routeTrace.sourceSummary.trusts },
        sourceUris: [...routeTrace.sourceSummary.sourceUris],
      },
      selectionMetadata: {
        ...routeTrace.selectionMetadata,
      },
    },
    selectedContext: routeTrace.injectedNodeSummaries.map(cloneInjectedSummary),
  };
}

export function isTeacherEligibleTrace(trace: DecisionTrace): boolean {
  return materializeTeacherLabelInput(trace) !== null;
}

export class BrainTeacher {
  constructor(
    private complete: BrainTeacherCompletion,
    private resolveModel: BrainTeacherResolveModel,
    private getApiKey: BrainTeacherGetApiKey,
    private graph: BrainGraph,
    private log: { info: (msg: string) => void; error: (msg: string) => void },
  ) {
    void this.graph;
  }

  async evaluateTrace(trace: DecisionTrace): Promise<TeacherLabelResultV1 | null> {
    const input = materializeTeacherLabelInput(trace);
    if (!input) {
      return null;
    }

    const prompt = `Evaluate this traced route decision. The JSON below is the entire teacher-visible surface.\n\n${JSON.stringify(input, null, 2)}`;

    try {
      const { provider, model } = this.resolveModel();
      const apiKey = await this.getApiKey(provider, model);

      const result = await this.complete({
        provider,
        model,
        apiKey,
        system: TEACHER_SYSTEM_PROMPT,
        messages: [{ role: "user", content: prompt }],
        maxTokens: 200,
        temperature: 0.1,
      });

      const text = result.content
        ?.map((b: { text?: string }) => b.text ?? "")
        .join("") ?? "";

      const jsonMatch = text.match(/\{[\s\S]*"score"[\s\S]*\}/);
      if (!jsonMatch) {
        this.log.error(`[brain] Teacher returned non-JSON for trace ${trace.id}: ${text.slice(0, 100)}`);
        return {
          version: 1,
          traceId: trace.id,
          episodeId: trace.episodeId,
          requestDigest: input.routeDecision.requestDigest,
          score: 0,
          reason: "failed to parse teacher response",
          input,
        };
      }

      const parsed = JSON.parse(jsonMatch[0]);
      const score = Math.max(-1, Math.min(1, Number(parsed.score) || 0));
      const reason = String(parsed.reason || "teacher evaluation");

      this.log.info(`[brain] Teacher scored trace ${trace.id}: ${score.toFixed(2)} (${reason})`);
      return {
        version: 1,
        traceId: trace.id,
        episodeId: trace.episodeId,
        requestDigest: input.routeDecision.requestDigest,
        score,
        reason,
        input,
      };
    } catch (err) {
      this.log.error(`[brain] Teacher evaluation failed for trace ${trace.id}: ${(err as Error).message}`);
      return {
        version: 1,
        traceId: trace.id,
        episodeId: trace.episodeId,
        requestDigest: input.routeDecision.requestDigest,
        score: 0,
        reason: "teacher evaluation failed",
        input,
      };
    }
  }
}
