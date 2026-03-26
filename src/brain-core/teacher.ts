/**
 * Off-path async teacher for full-turn reward evaluation.
 *
 * The teacher sees the persisted observation only:
 * - user query
 * - selected brain context
 * - route metadata
 * - actual assistant response
 * - tool outcomes
 * - next user turn when present
 */

import type { BrainGraph } from "./graph.js";
import type {
  BrainObservation,
  BrainObservationBindingMode,
  BrainObservationRouteMetadata,
  BrainObservationToolResult,
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

export interface TeacherLabelInputV2 {
  version: 2;
  observationId: string;
  episodeId: string;
  traceId: string | null;
  conversationId: number | null;
  queryText: string;
  selectedContext: DecisionTraceInjectedNodeSummary[];
  routeMetadata: BrainObservationRouteMetadata;
  assistantResponse: string;
  toolResults: BrainObservationToolResult[];
  nextUserTurn: string | null;
}

export interface TeacherLabelResultV2 {
  version: 2;
  observationId: string;
  episodeId: string;
  traceId: string | null;
  serveDecisionRecordId: string | null;
  selectionDigest: string | null;
  turnCompileEventId: string | null;
  decisionRecordedAt: string | null;
  activePackId: string | null;
  activePackEventExportDigest: string | null;
  activePackGraphChecksum: string | null;
  activePackRouterChecksum: string | null;
  activePackBuiltAt: string | null;
  bindingMode: BrainObservationBindingMode;
  retrievalRelevance: number;
  agentUsage: number;
  outcomeSupport: number;
  finalScore: number;
  confidence: number;
  reason: string;
  input: TeacherLabelInputV2;
}

const TEACHER_SYSTEM_PROMPT =
  "You are evaluating a persisted OpenClawBrain turn observation. Score only from the provided observation. " +
  "Judge (1) retrieval relevance of the selected context to the user query, " +
  "(2) agent usage of that context and any tools, and " +
  "(3) whether the observed outcome supports rewarding the route. " +
  "If the next user turn is missing or ambiguous, lower confidence rather than inventing certainty. " +
  "Return ONLY JSON: " +
  "{\"retrieval_relevance\": <number -1.0..1.0>, \"agent_usage\": <number -1.0..1.0>, \"outcome_support\": <number -1.0..1.0>, \"final_score\": <number -1.0..1.0>, \"confidence\": <number 0.0..1.0>, \"reason\": \"<brief explanation>\"}";

function clampSigned(value: unknown): number {
  return Math.max(-1, Math.min(1, Number(value) || 0));
}

function clampUnit(value: unknown): number {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function cloneContext(summary: DecisionTraceInjectedNodeSummary): DecisionTraceInjectedNodeSummary {
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

function cloneToolResult(result: BrainObservationToolResult): BrainObservationToolResult {
  return {
    sourceRole: result.sourceRole,
    toolCallId: result.toolCallId,
    toolName: result.toolName,
    input: result.input,
    output: result.output,
    isError: result.isError,
    excerpt: result.excerpt,
  };
}

function cloneServedArtifact(
  artifact: BrainObservationRouteMetadata["servedArtifact"],
): BrainObservationRouteMetadata["servedArtifact"] {
  return artifact ? JSON.parse(JSON.stringify(artifact)) as BrainObservationRouteMetadata["servedArtifact"] : null;
}

function resolveObservationBindingMode(observation: BrainObservation): BrainObservationBindingMode {
  if (observation.routeMetadata.serveDecisionRecordId) {
    return "exact_decision_id";
  }
  if (observation.routeMetadata.selectionDigest && observation.routeMetadata.activePackGraphChecksum) {
    return "exact_selection_digest";
  }
  if (observation.routeMetadata.turnCompileEventId) {
    return "turn_compile_event_id";
  }
  if (observation.traceId) {
    return "trace_id";
  }
  return "unbound";
}

export function materializeTeacherLabelInput(observation: BrainObservation): TeacherLabelInputV2 | null {
  if (typeof observation.queryText !== "string" || observation.queryText.trim().length === 0) {
    return null;
  }

  return {
    version: 2,
    observationId: observation.id,
    episodeId: observation.episodeId,
    traceId: observation.traceId,
    conversationId: observation.conversationId,
    queryText: observation.queryText,
    selectedContext: observation.retrievedContext.map(cloneContext),
    routeMetadata: {
      requestDigest: observation.routeMetadata.requestDigest,
      activePackId: observation.routeMetadata.activePackId,
      routerIdentity: observation.routeMetadata.routerIdentity,
      serveDecisionRecordId: observation.routeMetadata.serveDecisionRecordId,
      selectionDigest: observation.routeMetadata.selectionDigest,
      turnCompileEventId: observation.routeMetadata.turnCompileEventId,
      decisionRecordedAt: observation.routeMetadata.decisionRecordedAt,
      activePackEventExportDigest: observation.routeMetadata.activePackEventExportDigest,
      activePackGraphChecksum: observation.routeMetadata.activePackGraphChecksum,
      activePackRouterChecksum: observation.routeMetadata.activePackRouterChecksum,
      activePackBuiltAt: observation.routeMetadata.activePackBuiltAt,
      servedArtifact: cloneServedArtifact(observation.routeMetadata.servedArtifact),
      candidateNodeIds: [...observation.routeMetadata.candidateNodeIds],
      selectedNodeIds: [...observation.routeMetadata.selectedNodeIds],
      selectedTraversalNodeIds: [...observation.routeMetadata.selectedTraversalNodeIds],
      selectedPathNodeIds: [...observation.routeMetadata.selectedPathNodeIds],
      selectedSeedNodeIds: [...observation.routeMetadata.selectedSeedNodeIds],
      sourceSummary: observation.routeMetadata.sourceSummary
        ? {
            injectedCount: observation.routeMetadata.sourceSummary.injectedCount,
            kinds: { ...observation.routeMetadata.sourceSummary.kinds },
            trusts: { ...observation.routeMetadata.sourceSummary.trusts },
            sourceUris: [...observation.routeMetadata.sourceSummary.sourceUris],
          }
        : null,
      selectionMetadata: observation.routeMetadata.selectionMetadata
        ? {
            ...observation.routeMetadata.selectionMetadata,
          }
        : null,
    },
    assistantResponse: observation.assistantResponse,
    toolResults: observation.toolResults.map(cloneToolResult),
    nextUserTurn: observation.followUpText,
  };
}

export function isTeacherEligibleObservation(observation: BrainObservation): boolean {
  return materializeTeacherLabelInput(observation) !== null;
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

  async evaluateObservation(observation: BrainObservation): Promise<TeacherLabelResultV2 | null> {
    const input = materializeTeacherLabelInput(observation);
    if (!input) {
      return null;
    }
    const bindingMode = resolveObservationBindingMode(observation);
    const provenance = {
      serveDecisionRecordId: observation.routeMetadata.serveDecisionRecordId,
      selectionDigest: observation.routeMetadata.selectionDigest,
      turnCompileEventId: observation.routeMetadata.turnCompileEventId,
      decisionRecordedAt: observation.routeMetadata.decisionRecordedAt,
      activePackId: observation.routeMetadata.activePackId,
      activePackEventExportDigest: observation.routeMetadata.activePackEventExportDigest,
      activePackGraphChecksum: observation.routeMetadata.activePackGraphChecksum,
      activePackRouterChecksum: observation.routeMetadata.activePackRouterChecksum,
      activePackBuiltAt: observation.routeMetadata.activePackBuiltAt,
      bindingMode,
    };

    const prompt = `Evaluate this persisted OpenClawBrain turn observation.\n\n${JSON.stringify(input, null, 2)}`;

    try {
      const { provider, model } = this.resolveModel();
      const apiKey = await this.getApiKey(provider, model);
      const result = await this.complete({
        provider,
        model,
        apiKey,
        system: TEACHER_SYSTEM_PROMPT,
        messages: [{ role: "user", content: prompt }],
        maxTokens: 300,
        temperature: 0.1,
      });

      const text = result.content
        ?.map((block: { text?: string }) => block.text ?? "")
        .join("") ?? "";
      const jsonMatch = text.match(/\{[\s\S]*"reason"[\s\S]*\}/);
      if (!jsonMatch) {
        this.log.error(`[brain] Teacher returned non-JSON for observation ${observation.id}: ${text.slice(0, 100)}`);
        return {
          version: 2,
          observationId: observation.id,
          episodeId: observation.episodeId,
          traceId: observation.traceId,
          ...provenance,
          retrievalRelevance: 0,
          agentUsage: 0,
          outcomeSupport: 0,
          finalScore: 0,
          confidence: 0,
          reason: "failed to parse teacher response",
          input,
        };
      }

      const parsed = JSON.parse(jsonMatch[0]) as Record<string, unknown>;
      const retrievalRelevance = clampSigned(parsed.retrieval_relevance);
      const agentUsage = clampSigned(parsed.agent_usage);
      const outcomeSupport = clampSigned(parsed.outcome_support);
      const finalScore = clampSigned(
        parsed.final_score
          ?? ((retrievalRelevance + agentUsage + outcomeSupport) / 3),
      );
      const confidence = clampUnit(parsed.confidence ?? 0.5);
      const reason = String(parsed.reason || "teacher evaluation");

      this.log.info(
        `[brain] Teacher scored observation ${observation.id}: ${finalScore.toFixed(2)} (${reason})`,
      );
      return {
        version: 2,
        observationId: observation.id,
        episodeId: observation.episodeId,
        traceId: observation.traceId,
        ...provenance,
        retrievalRelevance,
        agentUsage,
        outcomeSupport,
        finalScore,
        confidence,
        reason,
        input,
      };
    } catch (error) {
      this.log.error(
        `[brain] Teacher evaluation failed for observation ${observation.id}: ${(error as Error).message}`,
      );
      return {
        version: 2,
        observationId: observation.id,
        episodeId: observation.episodeId,
        traceId: observation.traceId,
        ...provenance,
        retrievalRelevance: 0,
        agentUsage: 0,
        outcomeSupport: 0,
        finalScore: 0,
        confidence: 0,
        reason: "teacher evaluation failed",
        input,
      };
    }
  }
}
