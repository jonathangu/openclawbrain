/**
 * SQLite persistence for the brain's learned retrieval graph.
 *
 * Follows the same patterns as LCM's ConversationStore and SummaryStore:
 * - Constructor takes DatabaseSync
 * - Snake_case DB rows → camelCase TypeScript objects
 * - Prepared statements for performance
 */

import type { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type {
  BrainAgentIdentity,
  BrainNode,
  BrainEdge,
  EdgeKind,
  Episode,
  Label,
  RewardSource,
  Pack,
  MutationProposal,
  MutationBundleRecord,
  MutationBundleStatus,
  MutationStatus,
  BundleEvaluationVerdict,
  DecisionTrace,
  DecisionRouteTrace,
  BrainEvidence,
  BrainEvidenceKind,
  BrainEvidenceResolution,
  ResolvedLabel,
  SeedWeight,
  StopLocalWeight,
  TraceSupervisionRecord,
  BrainObservation,
  BrainObservationBindingMode,
  BrainObservationRouteMetadata,
  BrainObservationStatus,
  BrainObservationTeacherEvaluation,
  BrainObservationToolResult,
  BrainContextUsefulnessEvaluationV1,
  BrainContextUsefulnessSignal,
  BrainContextUsefulnessRecord,
  BrainContextUsefulnessSignalsV1,
  ContextFeedbackAgentCoverageSummary,
  ContextFeedbackFocusAction,
  ContextFeedbackSummary,
  ContextFeedbackVerdict,
  ContextUsefulnessLatestVerdict,
  ContextUsefulnessSummary,
  DecisionTraceInjectedNodeSummary,
  LearningJournalEventType,
  LearningJournalRecord,
  MutationProposedJournalPayload,
  BundleEvaluationStartedJournalPayload,
  BundleEvaluationCompletedJournalPayload,
  PromotionJournalPayload,
  TeacherFeedbackRichness,
} from "../brain-core/types.js";
import { classifyObservationBindingQuality, resolveObservationBindingMode } from "../brain-core/types.js";
import type {
  TeacherCanaryRolloutPlanV1,
  TeacherProposal,
  TeacherProposalDiffV1,
  TeacherProposalProofBundleV1,
  TeacherProposalReplayGateV1,
  TeacherProposalReplaySummaryV1,
  TeacherProposalSummaryV1,
} from "../brain-core/teacher-v3-contracts.js";
import {
  describeTeacherCanaryActivationGuardV1,
  diffTeacherProposalV1,
  normalizeTeacherProposalV1,
  summarizeTeacherProposalV1,
} from "../brain-core/teacher-v3-contracts.js";
import { usefulnessThresholds } from "../brain-core/usefulness.js";
import { summarizeRecentDecisionTraces, type RecentDecisionTraceSummary } from "../brain-core/trace.js";

// ═══════════════════════════════════════════
// Embedding serialization
// ═══════════════════════════════════════════

function serializeEmbedding(emb: Float32Array | null): Buffer | null {
  if (!emb) return null;
  return Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength);
}

function deserializeEmbedding(blob: Buffer | Uint8Array | null): Float32Array | null {
  if (!blob) return null;
  const buf = blob instanceof Buffer ? blob : Buffer.from(blob);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function parseJsonValue<T>(value: unknown, fallback: T): T {
  if (typeof value !== "string" || value.trim().length === 0) {
    return fallback;
  }
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function toOptionalString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toFiniteNumber(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function toObservationBindingMode(value: unknown): BrainObservationBindingMode | null {
  switch (value) {
    case "exact_decision_id":
    case "exact_selection_digest":
    case "turn_compile_event_id":
    case "trace_id":
    case "legacy_heuristic":
    case "unbound":
      return value;
    default:
      return null;
  }
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function hasNonEmptyToolResults(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value !== "string") {
    return false;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 && trimmed !== "[]";
}

function classifyTeacherFeedbackRichnessFromFields(params: {
  followUpText: unknown;
  toolResults: unknown;
}): TeacherFeedbackRichness {
  const hasFollowUp = toOptionalString(params.followUpText) !== null;
  const hasToolResults = hasNonEmptyToolResults(params.toolResults);
  if (hasFollowUp && hasToolResults) {
    return "followup_and_tool";
  }
  if (hasFollowUp) {
    return "followup_only";
  }
  if (hasToolResults) {
    return "tool_only";
  }
  return "sparse";
}

function isObservationReadyForTeacher(params: {
  status: unknown;
  followUpText: unknown;
  toolResults: unknown;
  createdAt: unknown;
  readyBefore: number;
}): boolean {
  if (params.status === "pending_teacher") {
    return true;
  }
  if (toOptionalString(params.followUpText) !== null) {
    return true;
  }
  if (hasNonEmptyToolResults(params.toolResults)) {
    return true;
  }
  return toFiniteNumber(params.createdAt) <= params.readyBefore;
}

function toBrainAgentIdentity(value: unknown): BrainAgentIdentity | null {
  const record = toRecord(value);
  const agentId = toOptionalString(record?.agentId);
  const lane = toOptionalString(record?.lane);
  if (!agentId || !lane) {
    return null;
  }
  return { agentId, lane };
}

function formatAgentIdentity(identity: BrainAgentIdentity): string {
  return identity.lane === "main"
    ? identity.agentId
    : `${identity.agentId}:${identity.lane}`;
}

function cloneJsonRecord(value: unknown): Record<string, unknown> | null {
  const record = toRecord(value);
  return record ? JSON.parse(JSON.stringify(record)) as Record<string, unknown> : null;
}

function normalizeObservationSourceSummary(
  value: unknown,
): BrainObservationRouteMetadata["sourceSummary"] {
  const record = toRecord(value);
  if (!record) {
    return null;
  }

  return {
    injectedCount: Number.isFinite(Number(record.injectedCount)) ? Number(record.injectedCount) : 0,
    kinds: (toRecord(record.kinds) ?? {}) as NonNullable<BrainObservationRouteMetadata["sourceSummary"]>["kinds"],
    trusts: (toRecord(record.trusts) ?? {}) as NonNullable<BrainObservationRouteMetadata["sourceSummary"]>["trusts"],
    sourceUris: toStringArray(record.sourceUris),
    sourceRefs: toStringArray(record.sourceRefs),
  };
}

function normalizeObservationSelectionMetadata(
  value: unknown,
): BrainObservationRouteMetadata["selectionMetadata"] {
  const record = toRecord(value);
  return record ? { ...record } as BrainObservationRouteMetadata["selectionMetadata"] : null;
}

function normalizeObservationRouteMetadata(
  value: unknown,
  options: { traceId?: string | null } = {},
): BrainObservationRouteMetadata {
  const record = toRecord(value) ?? {};
  const serveDecisionRecordId = toOptionalString(record.serveDecisionRecordId);
  const selectionDigest = toOptionalString(record.selectionDigest);
  const turnCompileEventId = toOptionalString(record.turnCompileEventId);
  const activePackGraphChecksum = toOptionalString(record.activePackGraphChecksum);
  return {
    requestDigest: toOptionalString(record.requestDigest),
    agentIdentity: toBrainAgentIdentity(record.agentIdentity),
    activePackId: toOptionalString(record.activePackId),
    routerIdentity: toOptionalString(record.routerIdentity),
    persistenceMode:
      record.persistenceMode === "redacted_with_operator_audit"
        ? "redacted_with_operator_audit"
        : (record.persistenceMode === "redacted" ? "redacted" : null),
    bindingMode: resolveObservationBindingMode({
      bindingMode: toObservationBindingMode(record.bindingMode),
      serveDecisionRecordId,
      selectionDigest,
      activePackGraphChecksum,
      turnCompileEventId,
      traceId: options.traceId ?? null,
    }),
    serveDecisionRecordId,
    selectionDigest,
    turnCompileEventId,
    decisionRecordedAt: toOptionalString(record.decisionRecordedAt),
    activePackEventExportDigest: toOptionalString(record.activePackEventExportDigest),
    activePackGraphChecksum,
    activePackRouterChecksum: toOptionalString(record.activePackRouterChecksum),
    activePackBuiltAt: toOptionalString(record.activePackBuiltAt),
    servedArtifact: cloneJsonRecord(record.servedArtifact),
    candidateNodeIds: toStringArray(record.candidateNodeIds),
    selectedNodeIds: toStringArray(record.selectedNodeIds),
    selectedTraversalNodeIds: toStringArray(record.selectedTraversalNodeIds),
    selectedPathNodeIds: toStringArray(record.selectedPathNodeIds),
    selectedSeedNodeIds: toStringArray(record.selectedSeedNodeIds),
    sourceSummary: normalizeObservationSourceSummary(record.sourceSummary),
    operatorAudit: cloneJsonRecord(record.operatorAudit) as BrainObservationRouteMetadata["operatorAudit"],
    selectionMetadata: normalizeObservationSelectionMetadata(record.selectionMetadata),
  };
}

function observationRouteMetadataFromRow(row: Record<string, unknown>): BrainObservationRouteMetadata {
  const traceId = toOptionalString(row.trace_id);
  const routeMetadata = normalizeObservationRouteMetadata(
    parseJsonValue(row.route_metadata_json, {}),
    { traceId },
  );
  return normalizeObservationRouteMetadata({
    ...routeMetadata,
    bindingMode: toObservationBindingMode(row.binding_mode) ?? routeMetadata.bindingMode,
    serveDecisionRecordId: toOptionalString(row.serve_decision_record_id) ?? routeMetadata.serveDecisionRecordId,
    selectionDigest: toOptionalString(row.selection_digest) ?? routeMetadata.selectionDigest,
    turnCompileEventId: toOptionalString(row.turn_compile_event_id) ?? routeMetadata.turnCompileEventId,
    decisionRecordedAt: toOptionalString(row.decision_recorded_at) ?? routeMetadata.decisionRecordedAt,
    activePackId: toOptionalString(row.active_pack_id) ?? routeMetadata.activePackId,
    activePackEventExportDigest:
      toOptionalString(row.active_pack_event_export_digest) ?? routeMetadata.activePackEventExportDigest,
    activePackGraphChecksum:
      toOptionalString(row.active_pack_graph_checksum) ?? routeMetadata.activePackGraphChecksum,
    activePackRouterChecksum:
      toOptionalString(row.active_pack_router_checksum) ?? routeMetadata.activePackRouterChecksum,
    activePackBuiltAt: toOptionalString(row.active_pack_built_at) ?? routeMetadata.activePackBuiltAt,
  }, { traceId });
}

function normalizeObservationTeacherEvaluation(
  value: unknown,
  params: {
    observationId: string;
    episodeId: string;
    traceId: string | null;
    routeMetadata: BrainObservationRouteMetadata;
  },
): BrainObservationTeacherEvaluation | null {
  const record = toRecord(value);
  if (!record) {
    return null;
  }

  const serveDecisionRecordId =
    params.routeMetadata.serveDecisionRecordId ?? toOptionalString(record.serveDecisionRecordId);
  const selectionDigest =
    params.routeMetadata.selectionDigest ?? toOptionalString(record.selectionDigest);
  const turnCompileEventId =
    params.routeMetadata.turnCompileEventId ?? toOptionalString(record.turnCompileEventId);
  const decisionRecordedAt =
    params.routeMetadata.decisionRecordedAt ?? toOptionalString(record.decisionRecordedAt);
  const activePackId =
    params.routeMetadata.activePackId ?? toOptionalString(record.activePackId);
  const activePackEventExportDigest =
    params.routeMetadata.activePackEventExportDigest ?? toOptionalString(record.activePackEventExportDigest);
  const activePackGraphChecksum =
    params.routeMetadata.activePackGraphChecksum ?? toOptionalString(record.activePackGraphChecksum);
  const activePackRouterChecksum =
    params.routeMetadata.activePackRouterChecksum ?? toOptionalString(record.activePackRouterChecksum);
  const activePackBuiltAt =
    params.routeMetadata.activePackBuiltAt ?? toOptionalString(record.activePackBuiltAt);
  const recordBindingMode = toObservationBindingMode(record.bindingMode);
  const routeBindingMode = params.routeMetadata.bindingMode;

  return {
    version: 2,
    observationId: toOptionalString(record.observationId) ?? params.observationId,
    episodeId: toOptionalString(record.episodeId) ?? params.episodeId,
    traceId: params.traceId ?? toOptionalString(record.traceId),
    serveDecisionRecordId,
    selectionDigest,
    turnCompileEventId,
    decisionRecordedAt,
    activePackId,
    activePackEventExportDigest,
    activePackGraphChecksum,
    activePackRouterChecksum,
    activePackBuiltAt,
    bindingMode: resolveObservationBindingMode({
      bindingMode: routeBindingMode === "unbound" ? (recordBindingMode ?? routeBindingMode) : routeBindingMode,
      serveDecisionRecordId,
      selectionDigest,
      activePackGraphChecksum,
      turnCompileEventId,
      traceId: params.traceId ?? toOptionalString(record.traceId),
    }),
    retrievalRelevance: toFiniteNumber(record.retrievalRelevance),
    agentUsage: toFiniteNumber(record.agentUsage),
    outcomeSupport: toFiniteNumber(record.outcomeSupport),
    finalScore: toFiniteNumber(record.finalScore),
    confidence: toFiniteNumber(record.confidence),
    reason: typeof record.reason === "string" ? record.reason : "",
  };
}

function normalizeContextUsefulnessSignal(
  value: unknown,
  fallbackClass: string,
  fallbackDetail: string,
): BrainContextUsefulnessSignal {
  const record = toRecord(value) ?? {};
  return {
    class: typeof record.class === "string" && record.class.trim().length > 0 ? record.class : fallbackClass,
    score: toFiniteNumber(record.score),
    evidence: toStringArray(record.evidence),
    detail: typeof record.detail === "string" && record.detail.trim().length > 0
      ? record.detail
      : fallbackDetail,
  };
}

function normalizeContextUsefulnessSignals(
  value: unknown,
): BrainContextUsefulnessSignalsV1 {
  const record = toRecord(value) ?? {};
  const authorityGate = toRecord(record.authorityGate) ?? {};
  return {
    followUp: normalizeContextUsefulnessSignal(record.followUp, "missing", "no follow-up signal") as BrainContextUsefulnessSignalsV1["followUp"],
    toolOutcome: normalizeContextUsefulnessSignal(record.toolOutcome, "missing", "no tool outcome signal") as BrainContextUsefulnessSignalsV1["toolOutcome"],
    routeIntegrity: normalizeContextUsefulnessSignal(record.routeIntegrity, "fallback", "no route integrity signal") as BrainContextUsefulnessSignalsV1["routeIntegrity"],
    teacherAlignment: record.teacherAlignment
      ? normalizeContextUsefulnessSignal(record.teacherAlignment, "calibration", "teacher calibration signal") as BrainContextUsefulnessSignalsV1["teacherAlignment"]
      : null,
    authorityGate: {
      blocked: Boolean(authorityGate.blocked),
      reason: typeof authorityGate.reason === "string" && authorityGate.reason.trim().length > 0
        ? authorityGate.reason
        : null,
    },
  };
}

function toContextUsefulness(row: Record<string, unknown>): BrainContextUsefulnessRecord {
  const signals = normalizeContextUsefulnessSignals(parseJsonValue(row.signal_json, {}));
  return {
    id: row.id as string,
    observationId: row.observation_id as string,
    episodeId: row.episode_id as string,
    traceId: toOptionalString(row.trace_id),
    conversationId: typeof row.conversation_id === "number" ? row.conversation_id : null,
    bindingMode: toObservationBindingMode(row.binding_mode),
    followUpText: toOptionalString(row.follow_up_text),
    toolResults: parseJsonValue<BrainObservationToolResult[]>(row.tool_results_json, []),
    signals,
    finalScore: toFiniteNumber(row.final_score),
    confidence: toFiniteNumber(row.confidence),
    verdict: (row.verdict as BrainContextUsefulnessRecord["verdict"]) ?? "irrelevant",
    reason: toOptionalString(row.reason),
    createdAt: toFiniteNumber(row.created_at),
    updatedAt: toFiniteNumber(row.updated_at),
    evaluatedAt: toFiniteNumber(row.evaluated_at),
  };
}

const OBSERVATION_BINDING_MODES: BrainObservationBindingMode[] = [
  "exact_decision_id",
  "exact_selection_digest",
  "turn_compile_event_id",
  "trace_id",
  "legacy_heuristic",
  "unbound",
];

function emptyObservationBindingModeCounts(): Record<BrainObservationBindingMode, number> {
  return {
    exact_decision_id: 0,
    exact_selection_digest: 0,
    turn_compile_event_id: 0,
    trace_id: 0,
    legacy_heuristic: 0,
    unbound: 0,
  };
}

const CONTEXT_FEEDBACK_HELPFUL_MIN = 0.25;
const CONTEXT_FEEDBACK_HARMFUL_MAX = -0.25;

function classifyContextFeedbackVerdict(score: number): ContextFeedbackVerdict {
  if (score >= CONTEXT_FEEDBACK_HELPFUL_MIN) {
    return "helpful";
  }
  if (score <= CONTEXT_FEEDBACK_HARMFUL_MAX) {
    return "harmful";
  }
  return "irrelevant";
}

function toContextFeedbackFocusAction(params: {
  routeTraceCount: number;
  harmfulCount: number;
  pendingFollowupCount: number;
  pendingTeacherCount: number;
  supervisedTraceCount: number;
}): { action: ContextFeedbackFocusAction; detail: string } {
  if (params.routeTraceCount === 0) {
    return {
      action: "monitor",
      detail: "no traced routes recorded yet",
    };
  }
  if (params.harmfulCount > 0) {
    return {
      action: "review_harmful_context",
      detail: `${params.harmfulCount} traced route verdict(s) are harmful; inspect the latest harmful context first`,
    };
  }
  if (params.pendingFollowupCount > 0) {
    return {
      action: "capture_follow_up",
      detail: `${params.pendingFollowupCount} observation(s) still need operator follow-up before teacher scoring`,
    };
  }
  if (params.pendingTeacherCount > 0) {
    return {
      action: "wait_for_teacher",
      detail: `${params.pendingTeacherCount} observation(s) are ready for teacher scoring`,
    };
  }
  if (params.supervisedTraceCount < params.routeTraceCount) {
    return {
      action: "increase_feedback_coverage",
      detail: `feedback covers ${params.supervisedTraceCount}/${params.routeTraceCount} traced route(s)`,
    };
  }
  return {
    action: "monitor",
    detail: "feedback loop is closed on every traced route",
  };
}

export interface LearningJournalInsert {
  eventType: LearningJournalEventType;
  mutationId?: string | null;
  mutationIds?: string[];
  bundleId?: string | null;
  packVersion?: number | null;
  payload: LearningJournalRecord["payload"];
  createdAt?: number;
}

export interface LearningJournalQuery {
  limit?: number;
  eventTypes?: LearningJournalEventType[];
  mutationId?: string;
  bundleId?: string;
  since?: number;
}

// ═══════════════════════════════════════════
// BrainStore
// ═══════════════════════════════════════════

export class BrainStore {
  constructor(
    private db: DatabaseSync,
    private options: { brainRoot?: string } = {},
  ) {
    if (this.options.brainRoot) {
      mkdirSync(this.options.brainRoot, { recursive: true });
      mkdirSync(this.getPacksDir(), { recursive: true });
    }
  }

  // ─── Nodes ───

  insertNode(node: BrainNode): void {
    this.db.prepare(`
      INSERT INTO brain_nodes (id, kind, content, embedding, source_uri, trust, tags, token_count, metadata, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      node.id, node.kind, node.content,
      serializeEmbedding(node.embedding),
      node.sourceUri, node.trust,
      JSON.stringify(node.tags), node.tokenCount,
      JSON.stringify(node.metadata),
      node.createdAt, node.updatedAt,
    );
  }

  getNode(id: string): BrainNode | null {
    const row = this.db.prepare(`SELECT * FROM brain_nodes WHERE id = ?`).get(id) as Record<string, unknown> | undefined;
    return row ? this.toNode(row) : null;
  }

  getAllNodes(): BrainNode[] {
    const rows = this.db.prepare(`SELECT * FROM brain_nodes`).all() as Record<string, unknown>[];
    return rows.map((r) => this.toNode(r));
  }

  updateNodeEmbedding(id: string, embedding: Float32Array): void {
    this.db.prepare(`UPDATE brain_nodes SET embedding = ?, updated_at = ? WHERE id = ?`)
      .run(serializeEmbedding(embedding), Date.now(), id);
  }

  clearGraph(): void {
    this.db.exec(`
      DELETE FROM brain_tool_action_priors;
      DELETE FROM brain_stop_local_weights;
      DELETE FROM brain_seed_weights;
      DELETE FROM brain_edges;
      DELETE FROM brain_nodes;
    `);
  }

  deleteNode(id: string): void {
    this.db.prepare(`DELETE FROM brain_tool_action_priors WHERE source_node_id = ? OR tool_node_id = ?`).run(id, id);
    this.db.prepare(`DELETE FROM brain_stop_local_weights WHERE source_node_id = ?`).run(id);
    this.db.prepare(`DELETE FROM brain_seed_weights WHERE node_id = ?`).run(id);
    this.db.prepare(`DELETE FROM brain_edges WHERE source = ? OR target = ?`).run(id, id);
    this.db.prepare(`DELETE FROM brain_nodes WHERE id = ?`).run(id);
  }

  private toNode(row: Record<string, unknown>): BrainNode {
    return {
      id: row.id as string,
      kind: row.kind as BrainNode["kind"],
      content: row.content as string,
      embedding: deserializeEmbedding(row.embedding as Buffer | null),
      sourceUri: (row.source_uri as string) || null,
      trust: row.trust as BrainNode["trust"],
      tags: JSON.parse((row.tags as string) || "[]"),
      tokenCount: (row.token_count as number) || 0,
      metadata: JSON.parse((row.metadata as string) || "{}"),
      createdAt: row.created_at as number,
      updatedAt: row.updated_at as number,
    };
  }

  // ─── Edges ───

  insertEdge(edge: BrainEdge): void {
    this.db.prepare(`
      INSERT OR REPLACE INTO brain_edges (source, target, kind, weight, prior, metadata, decayed_at, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      edge.source, edge.target, edge.kind,
      edge.weight, edge.prior,
      JSON.stringify(edge.metadata),
      edge.decayedAt, edge.createdAt,
    );
  }

  getOutgoingEdges(source: string): BrainEdge[] {
    const rows = this.db.prepare(`SELECT * FROM brain_edges WHERE source = ?`).all(source) as Record<string, unknown>[];
    return rows.map((r) => this.toEdge(r));
  }

  updateEdgeWeight(source: string, target: string, kind: EdgeKind, weight: number): void {
    this.db.prepare(`UPDATE brain_edges SET weight = ? WHERE source = ? AND target = ? AND kind = ?`)
      .run(weight, source, target, kind);
  }

  deleteEdge(source: string, target: string, kind: EdgeKind): void {
    this.db.prepare(`DELETE FROM brain_edges WHERE source = ? AND target = ? AND kind = ?`)
      .run(source, target, kind);
  }

  decayAllWeights(rate: number): void {
    const now = Date.now();
    this.db.prepare(`
      UPDATE brain_edges SET weight = weight * ? + prior * (1.0 - ?), decayed_at = ?
    `).run(rate, rate, now);
  }

  private toEdge(row: Record<string, unknown>): BrainEdge {
    return {
      source: row.source as string,
      target: row.target as string,
      kind: row.kind as EdgeKind,
      weight: row.weight as number,
      prior: row.prior as number,
      metadata: JSON.parse((row.metadata as string) || "{}"),
      decayedAt: row.decayed_at as number,
      createdAt: row.created_at as number,
    };
  }

  // ─── Seed Weights ───

  setSeedWeight(nodeId: string, weight: number): void {
    this.db.prepare(`
      INSERT INTO brain_seed_weights (node_id, weight, updated_at)
      VALUES (?, ?, ?)
      ON CONFLICT(node_id) DO UPDATE SET weight = excluded.weight, updated_at = excluded.updated_at
    `).run(nodeId, weight, Date.now());
  }

  getSeedWeight(nodeId: string): SeedWeight | null {
    const row = this.db.prepare(`SELECT * FROM brain_seed_weights WHERE node_id = ?`).get(nodeId) as Record<string, unknown> | undefined;
    if (!row) {
      return null;
    }
    return {
      nodeId: row.node_id as string,
      weight: row.weight as number,
      updatedAt: row.updated_at as number,
    };
  }

  getSeedWeights(nodeIds: string[]): Record<string, number> {
    if (nodeIds.length === 0) {
      return {};
    }
    const placeholders = nodeIds.map(() => "?").join(", ");
    const rows = this.db.prepare(`SELECT node_id, weight FROM brain_seed_weights WHERE node_id IN (${placeholders})`).all(...nodeIds) as Array<{ node_id: string; weight: number }>;
    const weights: Record<string, number> = {};
    for (const row of rows) {
      weights[row.node_id] = row.weight;
    }
    return weights;
  }

  getAllSeedWeights(): SeedWeight[] {
    const rows = this.db.prepare(`SELECT * FROM brain_seed_weights`).all() as Record<string, unknown>[];
    return rows.map((row) => ({
      nodeId: row.node_id as string,
      weight: row.weight as number,
      updatedAt: row.updated_at as number,
    }));
  }

  // ─── STOP_LOCAL Weights ───

  setStopLocalWeight(sourceNodeId: string, weight: number): void {
    this.db.prepare(`
      INSERT INTO brain_stop_local_weights (source_node_id, weight, updated_at)
      VALUES (?, ?, ?)
      ON CONFLICT(source_node_id) DO UPDATE SET weight = excluded.weight, updated_at = excluded.updated_at
    `).run(sourceNodeId, weight, Date.now());
  }

  getStopLocalWeight(sourceNodeId: string): StopLocalWeight | null {
    const row = this.db.prepare(`SELECT * FROM brain_stop_local_weights WHERE source_node_id = ?`).get(sourceNodeId) as Record<string, unknown> | undefined;
    if (!row) {
      return null;
    }
    return {
      sourceNodeId: row.source_node_id as string,
      weight: row.weight as number,
      updatedAt: row.updated_at as number,
    };
  }

  getAllStopLocalWeights(): StopLocalWeight[] {
    const rows = this.db.prepare(`SELECT * FROM brain_stop_local_weights`).all() as Record<string, unknown>[];
    return rows.map((row) => ({
      sourceNodeId: row.source_node_id as string,
      weight: row.weight as number,
      updatedAt: row.updated_at as number,
    }));
  }

  // ─── Tool Action Priors ───

  setToolActionPrior(sourceNodeId: string, toolNodeId: string, weight: number): void {
    this.db.prepare(`
      INSERT INTO brain_tool_action_priors (source_node_id, tool_node_id, weight, updated_at)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(source_node_id, tool_node_id) DO UPDATE SET weight = excluded.weight, updated_at = excluded.updated_at
    `).run(sourceNodeId, toolNodeId, weight, Date.now());
  }

  getToolActionPrior(sourceNodeId: string, toolNodeId: string): { sourceNodeId: string; toolNodeId: string; weight: number; updatedAt: number } | null {
    const row = this.db.prepare(`
      SELECT * FROM brain_tool_action_priors WHERE source_node_id = ? AND tool_node_id = ?
    `).get(sourceNodeId, toolNodeId) as Record<string, unknown> | undefined;
    if (!row) {
      return null;
    }
    return {
      sourceNodeId: row.source_node_id as string,
      toolNodeId: row.tool_node_id as string,
      weight: row.weight as number,
      updatedAt: row.updated_at as number,
    };
  }

  getAllToolActionPriors(): Array<{ sourceNodeId: string; toolNodeId: string; weight: number; updatedAt: number }> {
    const rows = this.db.prepare(`SELECT * FROM brain_tool_action_priors ORDER BY source_node_id, tool_node_id`).all() as Record<string, unknown>[];
    return rows.map((row) => ({
      sourceNodeId: row.source_node_id as string,
      toolNodeId: row.tool_node_id as string,
      weight: row.weight as number,
      updatedAt: row.updated_at as number,
    }));
  }

  // ─── Episodes ───

  insertEpisode(episode: Episode): void {
    this.db.prepare(`
      INSERT INTO brain_episodes (id, conversation_id, query_text, query_embedding, trajectory, fired_nodes, vetoed_nodes, context_chars, reward, reward_source, pack_version, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      episode.id, episode.conversationId, episode.queryText,
      serializeEmbedding(episode.queryEmbedding),
      JSON.stringify(episode.trajectory),
      JSON.stringify(episode.firedNodes),
      JSON.stringify(episode.vetoedNodes),
      episode.contextChars, episode.reward, episode.rewardSource,
      episode.packVersion, episode.createdAt,
    );
  }

  getEpisode(id: string): Episode | null {
    const row = this.db.prepare(`SELECT * FROM brain_episodes WHERE id = ?`).get(id) as Record<string, unknown> | undefined;
    return row ? this.toEpisode(row) : null;
  }

  getRecentEpisodes(limit: number): Episode[] {
    const rows = this.db.prepare(`SELECT * FROM brain_episodes ORDER BY created_at DESC LIMIT ?`).all(limit) as Record<string, unknown>[];
    return rows.map((r) => this.toEpisode(r));
  }

  getRecentEpisodesForConversation(conversationId: number, limit: number): Episode[] {
    const rows = this.db.prepare(`
      SELECT * FROM brain_episodes
      WHERE conversation_id = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(conversationId, limit) as Record<string, unknown>[];
    return rows.map((r) => this.toEpisode(r));
  }

  getEpisodesForUpdate(limit: number): Episode[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_episodes
      WHERE updated = 0
        AND (
          reward IS NOT NULL
          OR EXISTS (
            SELECT 1
            FROM brain_trace_supervision
            WHERE brain_trace_supervision.episode_id = brain_episodes.id
              AND brain_trace_supervision.resolution = 'promoted_to_label'
          )
        )
      ORDER BY created_at ASC
      LIMIT ?
    `).all(limit) as Record<string, unknown>[];
    return rows.map((r) => this.toEpisode(r));
  }

  getUnlabeledEpisodes(limit: number): Episode[] {
    const rows = this.db.prepare(`
      SELECT * FROM brain_episodes WHERE reward IS NULL ORDER BY created_at ASC LIMIT ?
    `).all(limit) as Record<string, unknown>[];
    return rows.map((r) => this.toEpisode(r));
  }

  setEpisodeReward(id: string, reward: number, source: RewardSource): void {
    this.db.prepare(`UPDATE brain_episodes SET reward = ?, reward_source = ? WHERE id = ?`)
      .run(reward, source, id);
  }

  markEpisodeUpdated(id: string): void {
    this.db.prepare(`UPDATE brain_episodes SET updated = 1 WHERE id = ?`).run(id);
  }

  private toEpisode(row: Record<string, unknown>): Episode {
    return {
      id: row.id as string,
      conversationId: (row.conversation_id as number) ?? null,
      queryText: (row.query_text as string) || "",
      queryEmbedding: deserializeEmbedding(row.query_embedding as Buffer | null),
      trajectory: JSON.parse((row.trajectory as string) || "[]"),
      firedNodes: JSON.parse((row.fired_nodes as string) || "[]"),
      vetoedNodes: JSON.parse((row.vetoed_nodes as string) || "[]"),
      contextChars: (row.context_chars as number) || 0,
      reward: (row.reward as number) ?? null,
      rewardSource: (row.reward_source as RewardSource) ?? null,
      packVersion: (row.pack_version as number) ?? null,
      createdAt: row.created_at as number,
    };
  }

  // ─── Labels ───

  insertLabel(params: { episodeId: string; source: RewardSource; value: number; confidence?: number; reason?: string }): Label {
    const id = `bl_${randomUUID().slice(0, 8)}`;
    const now = Date.now();
    this.db.prepare(`
      INSERT INTO brain_labels (id, episode_id, source, value, confidence, reason, applied, created_at)
      VALUES (?, ?, ?, ?, ?, ?, 0, ?)
    `).run(id, params.episodeId, params.source, params.value, params.confidence ?? 1.0, params.reason ?? null, now);
    return {
      id, episodeId: params.episodeId, source: params.source,
      value: params.value, confidence: params.confidence ?? 1.0,
      reason: params.reason ?? null, applied: false, createdAt: now,
    };
  }

  getPendingLabels(): Label[] {
    const rows = this.db.prepare(`SELECT * FROM brain_labels WHERE applied = 0 ORDER BY created_at ASC`).all() as Record<string, unknown>[];
    return rows.map((r) => ({
      id: r.id as string,
      episodeId: r.episode_id as string,
      source: r.source as RewardSource,
      value: r.value as number,
      confidence: (r.confidence as number) ?? 1.0,
      reason: (r.reason as string) ?? null,
      applied: false,
      createdAt: r.created_at as number,
    }));
  }

  countPendingLabelsBySource(): Record<RewardSource, number> {
    const rows = this.db.prepare(`
      SELECT source, COUNT(*) as count
      FROM brain_labels
      WHERE applied = 0
      GROUP BY source
    `).all() as Array<{ source: RewardSource; count: number }>;

    const counts: Record<RewardSource, number> = {
      human: 0,
      self: 0,
      scanner: 0,
      teacher: 0,
    };
    for (const row of rows) {
      counts[row.source] = row.count;
    }
    return counts;
  }

  markLabelApplied(id: string): void {
    this.db.prepare(`UPDATE brain_labels SET applied = 1 WHERE id = ?`).run(id);
  }

  // ─── Raw Evidence + Resolved Labels ───

  insertEvidence(params: {
    episodeId: string;
    conversationId?: number | null;
    source: RewardSource;
    kind: BrainEvidenceKind;
    value: number;
    confidence?: number;
    reason?: string;
    contentSnippet?: string;
    metadata?: Record<string, unknown>;
  }): BrainEvidence {
    const id = `be_${randomUUID().slice(0, 8)}`;
    const now = Date.now();
    this.db.prepare(`
      INSERT INTO brain_evidence (id, episode_id, conversation_id, source, kind, value, confidence, reason, content_snippet, metadata, resolved, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
    `).run(
      id,
      params.episodeId,
      params.conversationId ?? null,
      params.source,
      params.kind,
      params.value,
      params.confidence ?? 1.0,
      params.reason ?? null,
      params.contentSnippet ?? null,
      JSON.stringify(params.metadata ?? {}),
      now,
    );
    return {
      id,
      episodeId: params.episodeId,
      conversationId: params.conversationId ?? null,
      source: params.source,
      kind: params.kind,
      value: params.value,
      confidence: params.confidence ?? 1.0,
      reason: params.reason ?? null,
      contentSnippet: params.contentSnippet ?? null,
      metadata: params.metadata ?? {},
      resolved: false,
      createdAt: now,
    };
  }

  getPendingEvidence(limit = 100): BrainEvidence[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_evidence
      WHERE resolved = 0
      ORDER BY created_at ASC
      LIMIT ?
    `).all(limit) as Record<string, unknown>[];
    return rows.map((row) => this.toEvidence(row));
  }

  countPendingEvidenceBySource(): Record<RewardSource, number> {
    const rows = this.db.prepare(`
      SELECT source, COUNT(*) as count
      FROM brain_evidence
      WHERE resolved = 0
      GROUP BY source
    `).all() as Array<{ source: RewardSource; count: number }>;

    const counts: Record<RewardSource, number> = {
      human: 0,
      self: 0,
      scanner: 0,
      teacher: 0,
    };
    for (const row of rows) {
      counts[row.source] = row.count;
    }
    return counts;
  }

  resolveEvidence(params: {
    evidenceId: string;
    episodeId: string;
    source: RewardSource;
    value: number;
    confidence: number;
    resolution: BrainEvidenceResolution;
    labelId?: string | null;
    note?: string | null;
  }): ResolvedLabel {
    const id = `br_${randomUUID().slice(0, 8)}`;
    const now = Date.now();
    this.db.prepare(`UPDATE brain_evidence SET resolved = 1 WHERE id = ?`).run(params.evidenceId);
    this.db.prepare(`
      INSERT INTO brain_resolved_labels (id, evidence_id, episode_id, source, value, confidence, resolution, label_id, note, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      params.evidenceId,
      params.episodeId,
      params.source,
      params.value,
      params.confidence,
      params.resolution,
      params.labelId ?? null,
      params.note ?? null,
      now,
    );
    return {
      id,
      evidenceId: params.evidenceId,
      episodeId: params.episodeId,
      source: params.source,
      value: params.value,
      confidence: params.confidence,
      resolution: params.resolution,
      labelId: params.labelId ?? null,
      note: params.note ?? null,
      createdAt: now,
    };
  }

  getResolvedLabelsForEpisode(episodeId: string, limit = 20): ResolvedLabel[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_resolved_labels
      WHERE episode_id = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(episodeId, limit) as Record<string, unknown>[];
    return rows.map((row) => ({
      id: row.id as string,
      evidenceId: row.evidence_id as string,
      episodeId: row.episode_id as string,
      source: row.source as RewardSource,
      value: row.value as number,
      confidence: (row.confidence as number) ?? 1.0,
      resolution: row.resolution as BrainEvidenceResolution,
      labelId: (row.label_id as string) ?? null,
      note: (row.note as string) ?? null,
      createdAt: row.created_at as number,
    }));
  }

  insertTraceSupervision(params: {
    traceId: string;
    episodeId: string;
    conversationId?: number | null;
    source: RewardSource;
    kind: BrainEvidenceKind;
    value: number;
    confidence?: number;
    reason?: string | null;
    contentSnippet?: string | null;
    resolution: BrainEvidenceResolution;
    labelId?: string | null;
    evidenceId?: string | null;
    metadata?: Record<string, unknown>;
    createdAt?: number;
  }): TraceSupervisionRecord {
    const id = `ts_${randomUUID().slice(0, 8)}`;
    const createdAt = params.createdAt ?? Date.now();
    this.db.prepare(`
      INSERT INTO brain_trace_supervision (
        id,
        trace_id,
        episode_id,
        conversation_id,
        source,
        kind,
        value,
        confidence,
        reason,
        content_snippet,
        resolution,
        label_id,
        evidence_id,
        metadata,
        created_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      params.traceId,
      params.episodeId,
      params.conversationId ?? null,
      params.source,
      params.kind,
      params.value,
      params.confidence ?? 1.0,
      params.reason ?? null,
      params.contentSnippet ?? null,
      params.resolution,
      params.labelId ?? null,
      params.evidenceId ?? null,
      JSON.stringify(params.metadata ?? {}),
      createdAt,
    );
    return {
      id,
      traceId: params.traceId,
      episodeId: params.episodeId,
      conversationId: params.conversationId ?? null,
      source: params.source,
      kind: params.kind,
      value: params.value,
      confidence: params.confidence ?? 1.0,
      reason: params.reason ?? null,
      contentSnippet: params.contentSnippet ?? null,
      resolution: params.resolution,
      labelId: params.labelId ?? null,
      evidenceId: params.evidenceId ?? null,
      metadata: params.metadata ?? {},
      createdAt,
    };
  }

  getTraceSupervision(traceId: string, limit = 20): TraceSupervisionRecord[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_trace_supervision
      WHERE trace_id = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(traceId, limit) as Record<string, unknown>[];
    return rows.map((row) => this.toTraceSupervision(row));
  }

  getTraceSupervisionForEpisode(episodeId: string, limit = 50): TraceSupervisionRecord[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_trace_supervision
      WHERE episode_id = ?
      ORDER BY created_at ASC
      LIMIT ?
    `).all(episodeId, limit) as Record<string, unknown>[];
    return rows.map((row) => this.toTraceSupervision(row));
  }

  countTraceSupervision(): number {
    const row = this.db.prepare(`SELECT COUNT(*) as count FROM brain_trace_supervision`).get() as { count: number };
    return row.count ?? 0;
  }

  // ─── Durable Turn Observations ───

  insertObservation(params: {
    episodeId: string;
    conversationId?: number | null;
    traceId?: string | null;
    queryText: string;
    retrievedContext: DecisionTraceInjectedNodeSummary[];
    routeMetadata: BrainObservationRouteMetadata;
    assistantResponse?: string | null;
    toolResults?: BrainObservationToolResult[];
    followUpText?: string | null;
    status?: BrainObservationStatus;
    createdAt?: number;
    updatedAt?: number;
  }): BrainObservation {
    const existing = this.getObservationForEpisode(params.episodeId);
    if (existing) {
      return existing;
    }

    const id = `bo_${randomUUID().slice(0, 8)}`;
    const createdAt = params.createdAt ?? Date.now();
    const updatedAt = params.updatedAt ?? createdAt;
    const routeMetadata = normalizeObservationRouteMetadata(params.routeMetadata, {
      traceId: params.traceId ?? null,
    });
    this.db.prepare(`
      INSERT INTO brain_observations (
        id,
        episode_id,
        conversation_id,
        trace_id,
        binding_mode,
        serve_decision_record_id,
        selection_digest,
        turn_compile_event_id,
        decision_recorded_at,
        active_pack_id,
        active_pack_event_export_digest,
        active_pack_graph_checksum,
        active_pack_router_checksum,
        active_pack_built_at,
        query_text,
        retrieved_context_json,
        route_metadata_json,
        assistant_response,
        tool_results_json,
        follow_up_text,
        status,
        created_at,
        updated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      params.episodeId,
      params.conversationId ?? null,
      params.traceId ?? null,
      routeMetadata.bindingMode,
      routeMetadata.serveDecisionRecordId,
      routeMetadata.selectionDigest,
      routeMetadata.turnCompileEventId,
      routeMetadata.decisionRecordedAt,
      routeMetadata.activePackId,
      routeMetadata.activePackEventExportDigest,
      routeMetadata.activePackGraphChecksum,
      routeMetadata.activePackRouterChecksum,
      routeMetadata.activePackBuiltAt,
      params.queryText,
      JSON.stringify(params.retrievedContext ?? []),
      JSON.stringify(routeMetadata),
      params.assistantResponse ?? "",
      JSON.stringify(params.toolResults ?? []),
      params.followUpText ?? null,
      params.status ?? "pending_followup",
      createdAt,
      updatedAt,
    );

    return {
      id,
      episodeId: params.episodeId,
      conversationId: params.conversationId ?? null,
      traceId: params.traceId ?? null,
      queryText: params.queryText,
      retrievedContext: params.retrievedContext ?? [],
      routeMetadata,
      assistantResponse: params.assistantResponse ?? "",
      toolResults: params.toolResults ?? [],
      followUpText: params.followUpText ?? null,
      phase1Score: null,
      phase2Score: null,
      finalScore: null,
      confidence: null,
      reason: null,
      status: params.status ?? "pending_followup",
      teacherEvaluation: null,
      createdAt,
      updatedAt,
      evaluatedAt: null,
    };
  }

  getObservation(id: string): BrainObservation | null {
    const row = this.db.prepare(`SELECT * FROM brain_observations WHERE id = ?`).get(id) as Record<string, unknown> | undefined;
    return row ? this.toObservation(row) : null;
  }

  getObservationForEpisode(episodeId: string): BrainObservation | null {
    const row = this.db.prepare(`
      SELECT *
      FROM brain_observations
      WHERE episode_id = ?
      LIMIT 1
    `).get(episodeId) as Record<string, unknown> | undefined;
    return row ? this.toObservation(row) : null;
  }

  getPendingObservations(limit: number, readyBefore: number): BrainObservation[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_observations
      WHERE status IN ('pending_followup', 'pending_teacher')
        AND (
          status = 'pending_teacher'
          OR follow_up_text IS NOT NULL
          OR COALESCE(tool_results_json, '[]') <> '[]'
          OR created_at <= ?
        )
      ORDER BY created_at ASC
      LIMIT ?
    `).all(readyBefore, limit) as Record<string, unknown>[];
    return rows.map((row) => this.toObservation(row));
  }

  countPendingObservations(): number {
    const row = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_observations
      WHERE status IN ('pending_followup', 'pending_teacher')
    `).get() as { count: number };
    return row.count ?? 0;
  }

  countObservationsByStatus(): Record<BrainObservationStatus, number> {
    const rows = this.db.prepare(`
      SELECT status, COUNT(*) as count
      FROM brain_observations
      GROUP BY status
    `).all() as Array<{ status: BrainObservationStatus; count: number }>;
    const counts: Record<BrainObservationStatus, number> = {
      pending_followup: 0,
      pending_teacher: 0,
      completed: 0,
    };
    for (const row of rows) {
      counts[row.status] = row.count;
    }
    return counts;
  }

  getTeacherQueueSummary(
    readyBefore: number,
    budgetPerTick: number,
    sampleLimit = 12,
  ): {
    budgetPerTick: number;
    delayMs: number;
    pendingCount: number;
    pendingFollowupCount: number;
    pendingTeacherCount: number;
    readyCount: number;
    delayedCount: number;
    budgetDeferredCount: number;
    sparseReadyCount: number;
    richReadyCount: number;
    sample: Array<{
      observationId: string;
      episodeId: string;
      traceId: string | null;
      status: BrainObservationStatus;
      gate: "ready" | "delayed" | "budget_deferred";
      reason: string;
      feedbackRichness: TeacherFeedbackRichness;
      createdAt: number;
    }>;
    detail: string;
  } {
    const pendingRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_observations
      WHERE status IN ('pending_followup', 'pending_teacher')
    `).get() as { count: number };
    const readyRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_observations
      WHERE status IN ('pending_followup', 'pending_teacher')
        AND (
          status = 'pending_teacher'
          OR follow_up_text IS NOT NULL
          OR COALESCE(tool_results_json, '[]') <> '[]'
          OR created_at <= ?
        )
    `).get(readyBefore) as { count: number };
    const sparseReadyRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_observations
      WHERE status IN ('pending_followup', 'pending_teacher')
        AND (
          status = 'pending_teacher'
          OR follow_up_text IS NOT NULL
          OR COALESCE(tool_results_json, '[]') <> '[]'
          OR created_at <= ?
        )
        AND follow_up_text IS NULL
        AND COALESCE(tool_results_json, '[]') = '[]'
    `).get(readyBefore) as { count: number };
    const sampleRows = this.db.prepare(`
      SELECT id, episode_id, trace_id, status, follow_up_text, tool_results_json, created_at
      FROM brain_observations
      WHERE status IN ('pending_followup', 'pending_teacher')
      ORDER BY created_at ASC
      LIMIT ?
    `).all(Math.max(sampleLimit, budgetPerTick + 2)) as Array<{
      id: string;
      episode_id: string;
      trace_id: string | null;
      status: BrainObservationStatus;
      follow_up_text: string | null;
      tool_results_json: string | null;
      created_at: number;
    }>;

    const pendingStatusCounts = this.countObservationsByStatus();
    const pendingCount = pendingRow.count ?? 0;
    const pendingFollowupCount = pendingStatusCounts.pending_followup;
    const pendingTeacherCount = pendingStatusCounts.pending_teacher;
    const readyCount = readyRow.count ?? 0;
    const sparseReadyCount = sparseReadyRow.count ?? 0;
    const delayedCount = Math.max(0, pendingCount - readyCount);
    const budgetDeferredCount = Math.max(0, readyCount - budgetPerTick);
    const richReadyCount = Math.max(0, readyCount - sparseReadyCount);
    const sample = sampleRows.map((row, index) => {
      const feedbackRichness = classifyTeacherFeedbackRichnessFromFields({
        followUpText: row.follow_up_text,
        toolResults: row.tool_results_json,
      });
      const ready = isObservationReadyForTeacher({
        status: row.status,
        followUpText: row.follow_up_text,
        toolResults: row.tool_results_json,
        createdAt: row.created_at,
        readyBefore,
      });
      const gate: "ready" | "delayed" | "budget_deferred" =
        ready && index >= budgetPerTick
          ? "budget_deferred"
          : ready
            ? "ready"
            : "delayed";
      let reason = "ready for teacher scoring";
      if (gate === "delayed") {
        reason = "waiting for follow-up or the teacher delay window";
      } else if (gate === "budget_deferred") {
        reason = "ready, but deferred behind older observations by the per-tick teacher budget";
      } else if (feedbackRichness === "tool_only") {
        reason = "tool results make the observation ready without waiting for follow-up";
      } else if (feedbackRichness === "followup_only") {
        reason = "operator follow-up makes the observation ready";
      } else if (feedbackRichness === "followup_and_tool") {
        reason = "follow-up and tool results make the observation ready";
      } else if (feedbackRichness === "sparse") {
        reason = "teacher delay elapsed, so sparse feedback is now eligible for scoring";
      }

      return {
        observationId: row.id,
        episodeId: row.episode_id,
        traceId: toOptionalString(row.trace_id),
        status: row.status,
        gate,
        reason,
        feedbackRichness,
        createdAt: Number(row.created_at ?? 0),
      };
    }).slice(0, sampleLimit);

    const detail =
      pendingCount === 0
        ? "no teacher observations are pending"
        : `pending_followup=${pendingFollowupCount}, pending_teacher=${pendingTeacherCount}; ready=${readyCount}, delayed=${delayedCount}, budget_deferred=${budgetDeferredCount}, sparse_ready=${sparseReadyCount}, rich_ready=${richReadyCount}`;

    return {
      budgetPerTick,
      delayMs: Math.max(0, Date.now() - readyBefore),
      pendingCount,
      pendingFollowupCount,
      pendingTeacherCount,
      readyCount,
      delayedCount,
      budgetDeferredCount,
      sparseReadyCount,
      richReadyCount,
      sample,
      detail,
    };
  }

  getObservationAttributionSummary(): {
    totalObservationCount: number;
    completedObservationCount: number;
    completedWithoutEvaluationCount: number;
    teacherEvaluationCount: number;
    nonExactCount: number;
    bindingModes: Record<BrainObservationBindingMode, number>;
    attributionQuality: {
      exact: number;
      fallback: number;
      unbound: number;
    };
    latestAmbiguous: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: BrainObservationBindingMode;
      attributionQuality: "fallback" | "unbound";
      feedbackRichness: TeacherFeedbackRichness;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
    } | null;
    latestUnmatched: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: BrainObservationBindingMode;
      attributionQuality: "fallback" | "unbound";
      feedbackRichness: TeacherFeedbackRichness;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
    } | null;
    latestNonExact: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: BrainObservationBindingMode;
      attributionQuality: "fallback" | "unbound";
      feedbackRichness: TeacherFeedbackRichness;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
    } | null;
    detail: string;
  } {
    const pendingStatusCounts = this.countObservationsByStatus();
    const rows = this.db.prepare(`
      SELECT id, episode_id, trace_id, status, follow_up_text, tool_results_json, teacher_evaluation_json, evaluated_at, created_at
      FROM brain_observations
      ORDER BY evaluated_at DESC, created_at DESC
    `).all() as Array<{
      id: string;
      episode_id: string;
      trace_id: string | null;
      status?: BrainObservationStatus;
      follow_up_text?: string | null;
      tool_results_json?: string | null;
      teacher_evaluation_json?: string | null;
      evaluated_at?: number | null;
      created_at?: number;
    }>;
    const bindingModes = emptyObservationBindingModeCounts();
    const attributionQuality = {
      exact: 0,
      fallback: 0,
      unbound: 0,
    };
    let completedObservationCount = 0;
    let completedWithoutEvaluationCount = 0;
    let teacherEvaluationCount = 0;
    let latestAmbiguous: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: BrainObservationBindingMode;
      attributionQuality: "fallback" | "unbound";
      feedbackRichness: TeacherFeedbackRichness;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
    } | null = null;
    let latestUnmatched: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: BrainObservationBindingMode;
      attributionQuality: "fallback" | "unbound";
      feedbackRichness: TeacherFeedbackRichness;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
    } | null = null;
    let latestNonExact: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: BrainObservationBindingMode;
      attributionQuality: "fallback" | "unbound";
      feedbackRichness: TeacherFeedbackRichness;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
    } | null = null;

    for (const row of rows) {
      if (row.status === "completed") {
        completedObservationCount += 1;
      }
      const evaluation = row.teacher_evaluation_json
        ? parseJsonValue<BrainObservationTeacherEvaluation | null>(row.teacher_evaluation_json, null)
        : null;
      const bindingMode = evaluation?.bindingMode;
      if (!bindingMode || !OBSERVATION_BINDING_MODES.includes(bindingMode)) {
        if (row.status === "completed") {
          completedWithoutEvaluationCount += 1;
        }
        continue;
      }
      teacherEvaluationCount += 1;
      bindingModes[bindingMode] += 1;
      const quality = classifyObservationBindingQuality(bindingMode);
      attributionQuality[quality] += 1;
      const sample = {
        observationId: row.id,
        episodeId: row.episode_id,
        traceId: toOptionalString(row.trace_id),
        bindingMode,
        attributionQuality: quality,
        feedbackRichness: classifyTeacherFeedbackRichnessFromFields({
          followUpText: row.follow_up_text,
          toolResults: row.tool_results_json,
        }),
        confidence: evaluation?.confidence ?? null,
        reason: evaluation?.reason ?? null,
        evaluatedAt: row.evaluated_at === null ? null : Number(row.evaluated_at),
      };
      if (quality !== "exact" && !latestNonExact) {
        latestNonExact = sample as typeof latestNonExact extends null ? never : NonNullable<typeof latestNonExact>;
      }
      if (quality === "fallback" && !latestAmbiguous) {
        latestAmbiguous = sample as typeof latestAmbiguous extends null ? never : NonNullable<typeof latestAmbiguous>;
      }
      if (quality === "unbound" && !latestUnmatched) {
        latestUnmatched = sample as typeof latestUnmatched extends null ? never : NonNullable<typeof latestUnmatched>;
      }
    }

    const nonExactCount = attributionQuality.fallback + attributionQuality.unbound;
    const detail = `teacher attribution counts: evaluated=${teacherEvaluationCount}/${rows.length}, exact=${attributionQuality.exact}, fallback=${attributionQuality.fallback}, unbound=${attributionQuality.unbound}, pending_followup=${pendingStatusCounts.pending_followup}, pending_teacher=${pendingStatusCounts.pending_teacher}, completed_without_evaluation=${completedWithoutEvaluationCount}`;

    return {
      totalObservationCount: rows.length,
      completedObservationCount,
      completedWithoutEvaluationCount,
      teacherEvaluationCount,
      nonExactCount,
      bindingModes,
      attributionQuality,
      latestAmbiguous,
      latestUnmatched,
      latestNonExact,
      detail,
    };
  }

  getContextFeedbackSummary(): ContextFeedbackSummary {
    const traceRows = this.db.prepare(`
      SELECT id, created_at, route_trace_json
      FROM brain_traces
    `).all() as Array<{
      id: string;
      created_at: number;
      route_trace_json?: string;
    }>;
    const routeTraceCount = traceRows.length;
    const observationRows = this.db.prepare(`
      SELECT id, status
      FROM brain_observations
    `).all() as Array<{ id: string; status: BrainObservationStatus }>;
    let completedObservationCount = 0;
    let pendingFollowupCount = 0;
    let pendingTeacherCount = 0;

    for (const row of observationRows) {
      if (row.status === "completed") {
        completedObservationCount += 1;
      } else if (row.status === "pending_followup") {
        pendingFollowupCount += 1;
      } else if (row.status === "pending_teacher") {
        pendingTeacherCount += 1;
      }
    }

    const verdictCounts: Record<ContextFeedbackVerdict, number> = {
      helpful: 0,
      irrelevant: 0,
      harmful: 0,
    };
    const traceAgentIdentityById = new Map<string, BrainAgentIdentity>();
    const agentCoverage = new Map<string, {
      agentIdentity: BrainAgentIdentity;
      routeTraceCount: number;
      supervisedTraceCount: number;
      verdictCounts: Record<ContextFeedbackVerdict, number>;
      latestTraceAt: number | null;
      latestVerdictAt: number | null;
    }>();
    let identifiedRouteTraceCount = 0;
    for (const row of traceRows) {
      const routeTrace = parseJsonValue<DecisionRouteTrace | null>(row.route_trace_json, null);
      const agentIdentity = toBrainAgentIdentity(routeTrace?.agentIdentity);
      if (!agentIdentity) {
        continue;
      }
      identifiedRouteTraceCount += 1;
      traceAgentIdentityById.set(row.id, agentIdentity);
      const key = `${agentIdentity.agentId}::${agentIdentity.lane}`;
      const entry = agentCoverage.get(key) ?? {
        agentIdentity,
        routeTraceCount: 0,
        supervisedTraceCount: 0,
        verdictCounts: {
          helpful: 0,
          irrelevant: 0,
          harmful: 0,
        },
        latestTraceAt: null,
        latestVerdictAt: null,
      };
      entry.routeTraceCount += 1;
      entry.latestTraceAt = Math.max(entry.latestTraceAt ?? 0, row.created_at ?? 0) || null;
      agentCoverage.set(key, entry);
    }
    let latest: ContextFeedbackSummary["latest"] = null;
    const latestVerdictByTrace = new Map<string, true>();
    const supervisionRows = this.db.prepare(`
      SELECT *
      FROM brain_trace_supervision
      WHERE resolution = 'promoted_to_label'
      ORDER BY created_at DESC
    `).all() as Record<string, unknown>[];

    for (const row of supervisionRows) {
      const record = this.toTraceSupervision(row);
      if (!record.traceId || latestVerdictByTrace.has(record.traceId)) {
        continue;
      }
      latestVerdictByTrace.set(record.traceId, true);

      const verdict = classifyContextFeedbackVerdict(record.value);
      verdictCounts[verdict] += 1;
      const metadata = toRecord(record.metadata);
      const teacherEvaluation = toRecord(metadata?.teacherEvaluation);
      const agentIdentity =
        toBrainAgentIdentity(metadata?.agentIdentity)
        ?? traceAgentIdentityById.get(record.traceId)
        ?? null;
      const bindingMode =
        toObservationBindingMode(metadata?.bindingMode)
        ?? toObservationBindingMode(teacherEvaluation?.bindingMode);

      const current = {
        traceId: record.traceId,
        episodeId: record.episodeId,
        observationId: toOptionalString(metadata?.observationId),
        agentIdentity,
        source: record.source,
        verdict,
        score: record.value,
        confidence: record.confidence,
        reason: record.reason,
        bindingMode,
        createdAt: record.createdAt,
      };
      if (!latest) {
        latest = current;
      }

      if (agentIdentity) {
        const key = `${agentIdentity.agentId}::${agentIdentity.lane}`;
        const entry = agentCoverage.get(key) ?? {
          agentIdentity,
          routeTraceCount: 0,
          supervisedTraceCount: 0,
          verdictCounts: {
            helpful: 0,
            irrelevant: 0,
            harmful: 0,
          },
          latestTraceAt: null,
          latestVerdictAt: null,
        };
        entry.supervisedTraceCount += 1;
        entry.verdictCounts[verdict] += 1;
        entry.latestVerdictAt = Math.max(entry.latestVerdictAt ?? 0, record.createdAt ?? 0) || null;
        agentCoverage.set(key, entry);
      }
    }

    const supervisedTraceCount = latestVerdictByTrace.size;
    const unsupervisedTraceCount = Math.max(0, routeTraceCount - supervisedTraceCount);
    const unidentifiedRouteTraceCount = Math.max(0, routeTraceCount - identifiedRouteTraceCount);
    const agents: ContextFeedbackAgentCoverageSummary[] = [...agentCoverage.values()]
      .map((entry) => {
        const supervisionCoverage = entry.routeTraceCount > 0
          ? entry.supervisedTraceCount / entry.routeTraceCount
          : 0;
        const unsupervisedCount = Math.max(0, entry.routeTraceCount - entry.supervisedTraceCount);
        return {
          agentIdentity: entry.agentIdentity,
          routeTraceCount: entry.routeTraceCount,
          supervisedTraceCount: entry.supervisedTraceCount,
          unsupervisedTraceCount: unsupervisedCount,
          supervisionCoverage,
          verdictCounts: entry.verdictCounts,
          latestTraceAt: entry.latestTraceAt,
          latestVerdictAt: entry.latestVerdictAt,
          detail:
            `${formatAgentIdentity(entry.agentIdentity)}: feedback covers ${entry.supervisedTraceCount}/${entry.routeTraceCount} traced route(s); `
            + `${entry.verdictCounts.helpful} helpful, ${entry.verdictCounts.irrelevant} irrelevant, ${entry.verdictCounts.harmful} harmful`,
        };
      })
      .sort((left, right) => {
        if (right.routeTraceCount !== left.routeTraceCount) {
          return right.routeTraceCount - left.routeTraceCount;
        }
        if (right.supervisedTraceCount !== left.supervisedTraceCount) {
          return right.supervisedTraceCount - left.supervisedTraceCount;
        }
        return formatAgentIdentity(left.agentIdentity).localeCompare(formatAgentIdentity(right.agentIdentity));
      });
    const focus = toContextFeedbackFocusAction({
      routeTraceCount,
      harmfulCount: verdictCounts.harmful,
      pendingFollowupCount,
      pendingTeacherCount,
      supervisedTraceCount,
    });

    return {
      scoreBands: {
        helpfulMin: CONTEXT_FEEDBACK_HELPFUL_MIN,
        harmfulMax: CONTEXT_FEEDBACK_HARMFUL_MAX,
      },
      verdictCounts,
      coverage: {
        routeTraceCount,
        identifiedRouteTraceCount,
        unidentifiedRouteTraceCount,
        agentIdentityCoverage: routeTraceCount > 0 ? identifiedRouteTraceCount / routeTraceCount : 0,
        observationCount: observationRows.length,
        completedObservationCount,
        supervisedTraceCount,
        unsupervisedTraceCount,
        observationCoverage: routeTraceCount > 0 ? observationRows.length / routeTraceCount : 0,
        supervisionCoverage: routeTraceCount > 0 ? supervisedTraceCount / routeTraceCount : 0,
        pendingFollowupCount,
        pendingTeacherCount,
      },
      agents,
      latest,
      focus,
      detail:
        routeTraceCount === 0
          ? "no traced routes recorded yet"
          : `${verdictCounts.helpful} helpful, ${verdictCounts.irrelevant} irrelevant, ${verdictCounts.harmful} harmful traced route verdicts; feedback covers ${supervisedTraceCount}/${routeTraceCount} traced route(s); agent identity covers ${identifiedRouteTraceCount}/${routeTraceCount} traced route(s); helpful >= ${CONTEXT_FEEDBACK_HELPFUL_MIN}, harmful <= ${CONTEXT_FEEDBACK_HARMFUL_MAX}`,
    };
  }

  attachObservationFollowUp(
    conversationId: number,
    followUpText: string,
    episodeId?: string | null,
  ): BrainObservation | null {
    const normalized = followUpText.trim();
    if (!normalized) {
      return null;
    }

    const normalizedEpisodeId = typeof episodeId === "string" ? episodeId.trim() : "";
    const row =
      normalizedEpisodeId.length > 0
        ? this.db.prepare(`
      SELECT id
      FROM brain_observations
      WHERE episode_id = ?
        AND conversation_id = ?
        AND status IN ('pending_followup', 'pending_teacher')
        AND follow_up_text IS NULL
      LIMIT 1
    `).get(normalizedEpisodeId, conversationId) as { id?: string } | undefined
        : undefined;
    const fallbackRow = this.db.prepare(`
      SELECT id
      FROM brain_observations
      WHERE conversation_id = ?
        AND status IN ('pending_followup', 'pending_teacher')
        AND follow_up_text IS NULL
      ORDER BY created_at DESC
      LIMIT 1
    `).get(conversationId) as { id?: string } | undefined;
    const targetRow = row?.id ? row : fallbackRow;
    if (!targetRow?.id) {
      return null;
    }

    const now = Date.now();
    this.db.prepare(`
      UPDATE brain_observations
      SET follow_up_text = ?, status = 'pending_teacher', updated_at = ?
      WHERE id = ?
    `).run(normalized, now, targetRow.id);
    return this.getObservation(targetRow.id);
  }

  completeObservationEvaluation(params: {
    observationId: string;
    phase1Score?: number | null;
    phase2Score?: number | null;
    finalScore?: number | null;
    confidence?: number | null;
    reason?: string | null;
    status?: BrainObservationStatus;
    teacherEvaluation?: BrainObservationTeacherEvaluation | null;
    evaluatedAt?: number | null;
  }): BrainObservation | null {
    const now = Date.now();
    this.db.prepare(`
      UPDATE brain_observations
      SET phase1_score = ?,
          phase2_score = ?,
          final_score = ?,
          confidence = ?,
          reason = ?,
          status = ?,
          teacher_evaluation_json = ?,
          updated_at = ?,
          evaluated_at = ?
      WHERE id = ?
    `).run(
      params.phase1Score ?? null,
      params.phase2Score ?? null,
      params.finalScore ?? null,
      params.confidence ?? null,
      params.reason ?? null,
      params.status ?? "completed",
      params.teacherEvaluation ? JSON.stringify(params.teacherEvaluation) : null,
      now,
      params.evaluatedAt ?? now,
      params.observationId,
    );
    return this.getObservation(params.observationId);
  }

  insertContextUsefulnessEvaluation(params: {
    observation: BrainObservation;
    evaluation: BrainContextUsefulnessEvaluationV1;
  }): BrainContextUsefulnessRecord | null {
    const existing = this.getContextUsefulnessForObservation(params.observation.id);
    if (existing) {
      return existing;
    }

    const now = params.evaluation.computedAt;
    const id = `uc_${randomUUID().slice(0, 8)}`;
    this.db.prepare(`
      INSERT INTO brain_context_usefulness (
        id,
        observation_id,
        episode_id,
        trace_id,
        conversation_id,
        binding_mode,
        follow_up_text,
        tool_results_json,
        signal_json,
        final_score,
        confidence,
        verdict,
        reason,
        created_at,
        updated_at,
        evaluated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      params.evaluation.observationId,
      params.evaluation.episodeId,
      params.evaluation.traceId,
      params.evaluation.conversationId,
      params.evaluation.bindingMode,
      params.observation.followUpText,
      JSON.stringify(params.observation.toolResults ?? []),
      JSON.stringify(params.evaluation.signals),
      params.evaluation.finalScore,
      params.evaluation.confidence,
      params.evaluation.verdict,
      params.evaluation.reason,
      now,
      now,
      now,
    );

    return this.getContextUsefulnessForObservation(params.observation.id);
  }

  getContextUsefulnessForObservation(observationId: string): BrainContextUsefulnessRecord | null {
    const row = this.db.prepare(`
      SELECT *
      FROM brain_context_usefulness
      WHERE observation_id = ?
      LIMIT 1
    `).get(observationId) as Record<string, unknown> | undefined;
    return row ? toContextUsefulness(row) : null;
  }

  getPendingContextUsefulnessObservations(limit = 20): BrainObservation[] {
    const rows = this.db.prepare(`
      SELECT o.*
      FROM brain_observations o
      LEFT JOIN brain_context_usefulness u ON u.observation_id = o.id
      WHERE u.id IS NULL
        AND o.follow_up_text IS NOT NULL
        AND o.status IN ('pending_teacher', 'completed')
      ORDER BY o.updated_at ASC, o.created_at ASC
      LIMIT ?
    `).all(limit) as Record<string, unknown>[];
    return rows.map((row) => this.toObservation(row));
  }

  countContextUsefulness(): number {
    const row = this.db.prepare(`SELECT COUNT(*) as count FROM brain_context_usefulness`).get() as { count: number };
    return row.count ?? 0;
  }

  countContextUsefulnessByVerdict(): Record<BrainContextUsefulnessRecord["verdict"], number> {
    const rows = this.db.prepare(`
      SELECT verdict, COUNT(*) as count
      FROM brain_context_usefulness
      GROUP BY verdict
    `).all() as Array<{ verdict: BrainContextUsefulnessRecord["verdict"]; count: number }>;
    const counts: Record<BrainContextUsefulnessRecord["verdict"], number> = {
      helpful: 0,
      irrelevant: 0,
      harmful: 0,
    };
    for (const row of rows) {
      if (row.verdict in counts) {
        counts[row.verdict] = row.count;
      }
    }
    return counts;
  }

  getContextUsefulnessSummary(): ContextUsefulnessSummary {
    const thresholds = usefulnessThresholds();
    const observationRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_observations
    `).get() as { count: number };
    const readyRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_observations
      WHERE follow_up_text IS NOT NULL
        AND status IN ('pending_teacher', 'completed')
    `).get() as { count: number };
    const completedRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_observations
      WHERE status = 'completed'
    `).get() as { count: number };
    const pendingRows = this.db.prepare(`
      SELECT status, COUNT(*) as count
      FROM brain_observations
      GROUP BY status
    `).all() as Array<{ status: BrainObservationStatus; count: number }>;
    let pendingFollowupCount = 0;
    let pendingTeacherCount = 0;
    for (const row of pendingRows) {
      if (row.status === "pending_followup") {
        pendingFollowupCount = row.count;
      } else if (row.status === "pending_teacher") {
        pendingTeacherCount = row.count;
      }
    }
    const verdictCounts = this.countContextUsefulnessByVerdict();
    const latestRow = this.db.prepare(`
      SELECT *
      FROM brain_context_usefulness
      ORDER BY evaluated_at DESC, created_at DESC
      LIMIT 1
    `).get() as Record<string, unknown> | undefined;
    const latest = latestRow ? (() => {
      const record = toContextUsefulness(latestRow);
      return {
        observationId: record.observationId,
        episodeId: record.episodeId,
        traceId: record.traceId,
        verdict: record.verdict,
        score: record.finalScore,
        confidence: record.confidence,
        reason: record.reason,
        bindingMode: record.bindingMode,
        createdAt: record.createdAt,
      } satisfies ContextUsefulnessLatestVerdict;
    })() : null;
    const scoredObservationCount = this.countContextUsefulness();
    const readyObservationCount = readyRow.count ?? 0;
    const observationCount = observationRow.count ?? 0;

    return {
      scoreBands: thresholds,
      verdictCounts,
      coverage: {
        observationCount,
        readyObservationCount,
        scoredObservationCount,
        completedObservationCount: completedRow.count ?? 0,
        observationCoverage: observationCount > 0 ? scoredObservationCount / observationCount : 0,
        readyCoverage: readyObservationCount > 0 ? scoredObservationCount / readyObservationCount : 0,
        pendingFollowupCount,
        pendingTeacherCount,
      },
      latest,
      detail:
        scoredObservationCount === 0
          ? "no shadow usefulness scores recorded yet"
          : `${verdictCounts.helpful} helpful, ${verdictCounts.irrelevant} irrelevant, ${verdictCounts.harmful} harmful shadow verdicts; ${scoredObservationCount}/${readyObservationCount} ready observation(s) scored`,
    };
  }

  private toEvidence(row: Record<string, unknown>): BrainEvidence {
    return {
      id: row.id as string,
      episodeId: row.episode_id as string,
      conversationId: (row.conversation_id as number) ?? null,
      source: row.source as RewardSource,
      kind: row.kind as BrainEvidenceKind,
      value: row.value as number,
      confidence: (row.confidence as number) ?? 1.0,
      reason: (row.reason as string) ?? null,
      contentSnippet: (row.content_snippet as string) ?? null,
      metadata: parseJsonValue(row.metadata, {}),
      resolved: !!(row.resolved as number),
      createdAt: row.created_at as number,
    };
  }

  // ─── Packs ───

  insertPack(params: { nodeCount: number; edgeCount: number; healthJson: string }): Pack {
    const now = Date.now();
    this.db.prepare(`
      INSERT INTO brain_packs (node_count, edge_count, health_json, created_at) VALUES (?, ?, ?, ?)
    `).run(params.nodeCount, params.edgeCount, params.healthJson, now);

    const row = this.db.prepare(`SELECT last_insert_rowid() as version`).get() as { version: number };
    return {
      version: row.version, nodeCount: params.nodeCount, edgeCount: params.edgeCount,
      healthJson: params.healthJson, promotedAt: null, rolledBack: false, createdAt: now,
    };
  }

  getCurrentPack(): Pack | null {
    const row = this.db.prepare(`
      SELECT * FROM brain_packs WHERE promoted_at IS NOT NULL AND rolled_back = 0 ORDER BY version DESC LIMIT 1
    `).get() as Record<string, unknown> | undefined;
    return row ? this.toPack(row) : null;
  }

  getRecentPromotedPacks(limit = 5): Pack[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_packs
      WHERE promoted_at IS NOT NULL
      ORDER BY promoted_at DESC, version DESC
      LIMIT ?
    `).all(limit) as Record<string, unknown>[];
    return rows.map((row) => this.toPack(row));
  }

  promotePack(version: number): void {
    this.db.prepare(`UPDATE brain_packs SET promoted_at = ? WHERE version = ?`).run(Date.now(), version);
    if (this.options.brainRoot) {
      writeFileSync(this.getCurrentPackFile(), String(version), "utf8");
    }
  }

  rollbackPack(version: number): void {
    this.db.prepare(`UPDATE brain_packs SET rolled_back = 1 WHERE version = ?`).run(version);
    if (!this.options.brainRoot) {
      return;
    }

    const row = this.db.prepare(`
      SELECT version
      FROM brain_packs
      WHERE promoted_at IS NOT NULL AND rolled_back = 0 AND version < ?
      ORDER BY version DESC
      LIMIT 1
    `).get(version) as { version?: number } | undefined;

    if (typeof row?.version === "number") {
      writeFileSync(this.getCurrentPackFile(), String(row.version), "utf8");
      return;
    }

    writeFileSync(this.getCurrentPackFile(), "", "utf8");
  }

  // ─── Mutations ───

  insertMutation(mutation: MutationProposal): void {
    this.db.prepare(`
      INSERT INTO brain_mutations (id, kind, proposal, evidence, expected_gain, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(
      mutation.id, mutation.kind, JSON.stringify(mutation.proposal),
      mutation.evidence ? JSON.stringify(mutation.evidence) : null,
      mutation.expectedGain, mutation.status, mutation.createdAt,
    );
  }

  resolveMutation(id: string, status: MutationStatus): void {
    this.db.prepare(`UPDATE brain_mutations SET status = ?, resolved_at = ? WHERE id = ?`)
      .run(status, Date.now(), id);
  }

  insertMutationBundle(params: {
    id: string;
    mutationIds: string[];
    bundleSize: number;
    status?: MutationBundleStatus;
    baseScore?: number | null;
    candidateScore?: number | null;
    expectedGain: number;
    rejectionReason?: string | null;
    verdict?: BundleEvaluationVerdict | null;
    createdAt: number;
    resolvedAt?: number | null;
  }): void {
    this.db.prepare(`
      INSERT INTO brain_mutation_bundles (
        id,
        mutation_ids,
        bundle_size,
        status,
        base_score,
        candidate_score,
        expected_gain,
        rejection_reason,
        verdict_json,
        created_at,
        resolved_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      params.id,
      JSON.stringify(params.mutationIds),
      params.bundleSize,
      params.status ?? "pending",
      params.baseScore ?? null,
      params.candidateScore ?? null,
      params.expectedGain,
      params.rejectionReason ?? null,
      params.verdict ? JSON.stringify(params.verdict) : null,
      params.createdAt,
      params.resolvedAt ?? null,
    );
  }

  resolveMutationBundle(params: {
    id: string;
    status: MutationBundleStatus;
    baseScore: number | null;
    candidateScore: number | null;
    rejectionReason?: string | null;
    verdict?: BundleEvaluationVerdict | null;
    resolvedAt?: number | null;
  }): void {
    this.db.prepare(`
      UPDATE brain_mutation_bundles
      SET status = ?,
          base_score = ?,
          candidate_score = ?,
          rejection_reason = ?,
          verdict_json = ?,
          resolved_at = ?
      WHERE id = ?
    `).run(
      params.status,
      params.baseScore,
      params.candidateScore,
      params.rejectionReason ?? null,
      params.verdict ? JSON.stringify(params.verdict) : null,
      params.resolvedAt ?? Date.now(),
      params.id,
    );
  }

  updateMutationBundle(
    id: string,
    params: {
      status: MutationBundleStatus;
      baseScore?: number | null;
      candidateScore?: number | null;
      rejectionReason?: string | null;
      verdict?: BundleEvaluationVerdict | null;
      resolvedAt?: number | null;
    },
  ): void {
    this.resolveMutationBundle({
      id,
      status: params.status,
      baseScore: params.baseScore ?? null,
      candidateScore: params.candidateScore ?? null,
      rejectionReason: params.rejectionReason ?? null,
      verdict: params.verdict ?? null,
      resolvedAt: params.resolvedAt ?? null,
    });
  }

  getMutationsByStatus(status: MutationStatus, limit = 50): MutationProposal[] {
    const rows = this.db.prepare(`
      SELECT * FROM brain_mutations
      WHERE status = ?
      ORDER BY created_at ASC
      LIMIT ?
    `).all(status, limit) as Record<string, unknown>[];
    return rows.map((row) => ({
      id: row.id as string,
      kind: row.kind as MutationProposal["kind"],
      proposal: JSON.parse((row.proposal as string) || "{}"),
      evidence: row.evidence ? JSON.parse(row.evidence as string) : null,
      expectedGain: (row.expected_gain as number) ?? null,
      status: row.status as MutationStatus,
      createdAt: row.created_at as number,
      resolvedAt: (row.resolved_at as number) ?? null,
    }));
  }

  getRecentMutationsByStatus(status: MutationStatus, limit = 10): MutationProposal[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_mutations
      WHERE status = ?
      ORDER BY COALESCE(resolved_at, created_at) DESC, created_at DESC
      LIMIT ?
    `).all(status, limit) as Record<string, unknown>[];
    return rows.map((row) => ({
      id: row.id as string,
      kind: row.kind as MutationProposal["kind"],
      proposal: JSON.parse((row.proposal as string) || "{}"),
      evidence: row.evidence ? JSON.parse(row.evidence as string) : null,
      expectedGain: (row.expected_gain as number) ?? null,
      status: row.status as MutationStatus,
      createdAt: row.created_at as number,
      resolvedAt: (row.resolved_at as number) ?? null,
    }));
  }

  countMutationsByStatus(): Record<MutationStatus, number> {
    const rows = this.db.prepare(`
      SELECT status, COUNT(*) as count
      FROM brain_mutations
      GROUP BY status
    `).all() as Array<{ status: MutationStatus; count: number }>;

    const counts: Record<MutationStatus, number> = {
      pending: 0,
      validated: 0,
      promoted: 0,
      rejected: 0,
    };
    for (const row of rows) {
      counts[row.status] = row.count;
    }
    return counts;
  }

  getMutationBundle(id: string): MutationBundleRecord | null {
    const row = this.db.prepare(`SELECT * FROM brain_mutation_bundles WHERE id = ?`).get(id) as Record<string, unknown> | undefined;
    return row ? this.toMutationBundle(row) : null;
  }

  getMutationBundlesByStatus(status: MutationBundleStatus, limit = 50): MutationBundleRecord[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_mutation_bundles
      WHERE status = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(status, limit) as Record<string, unknown>[];
    return rows.map((row) => this.toMutationBundle(row));
  }

  getRecentMutationBundles(limit = 20): MutationBundleRecord[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_mutation_bundles
      ORDER BY created_at DESC
      LIMIT ?
    `).all(limit) as Record<string, unknown>[];
    return rows.map((row) => this.toMutationBundle(row));
  }

  // ─── Teacher Proposals ───

  private validateTeacherCanaryActivation(proposal: TeacherProposal): void {
    const guard = describeTeacherCanaryActivationGuardV1({
      proposalId: proposal.proposalId,
      proposalClass: proposal.proposalClass,
      rollbackKey: proposal.rollbackKey,
      canaryRollout: proposal.canaryRollout ?? null,
      replaySummary: proposal.replaySummary ?? null,
      proofBundle: proposal.proofBundle ?? null,
    });

    if (guard.requested && !guard.allowed) {
      throw new Error('cannot activate Teacher v3 canary for ' + proposal.proposalId + ': ' + guard.blockers.join('; '));
    }
  }

  insertTeacherProposal(proposal: TeacherProposal): void {
    const now = Date.now();
    this.validateTeacherCanaryActivation(proposal);
    const record = this.toTeacherProposalRow(proposal, now, now);
    this.db.prepare(`
      INSERT INTO brain_teacher_proposals (
        proposal_id,
        proposal_class,
        proposal_kind,
        lane,
        status,
        lifecycle_state,
        schema_version,
        idempotency_key,
        rollback_key,
        scope,
        base_pack_version,
        base_graph_hash,
        producer_version,
        producer_build_id,
        prompt_hash,
        template_id,
        profile,
        source_bundle_id,
        replay_suite_ids,
        freshness_ts,
        proposal_json,
        created_at,
        updated_at,
        resolved_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      record.proposalId,
      record.proposalClass,
      record.proposalKind,
      record.lane,
      record.status,
      record.lifecycleState,
      record.schemaVersion,
      record.idempotencyKey,
      record.rollbackKey,
      record.scope,
      record.basePackVersion,
      record.baseGraphHash,
      record.producerVersion,
      record.producerBuildId,
      record.promptHash,
      record.templateId,
      record.profile,
      record.sourceBundleId,
      record.replaySuiteIds,
      record.freshnessTs,
      record.proposalJson,
      record.createdAt,
      record.updatedAt,
      record.resolvedAt,
    );
  }

  updateTeacherProposalStatus(params: {
    proposalId: string;
    status: TeacherProposal["status"];
    resolvedAt?: string | null;
    proofBundle?: TeacherProposalProofBundleV1 | null;
    replayGate?: TeacherProposalReplayGateV1 | null;
    canaryRollout?: TeacherCanaryRolloutPlanV1 | null;
    replaySummary?: TeacherProposalReplaySummaryV1 | null;
  }): void {
    const existing = this.getTeacherProposal(params.proposalId);
    if (!existing) {
      throw new Error(`teacher proposal not found: ${params.proposalId}`);
    }

    const terminalStatuses: TeacherProposal["status"][] = ["promoted", "rejected", "expired", "rolled_back"];
    const resolvedAt = params.resolvedAt ?? (terminalStatuses.includes(params.status) ? new Date().toISOString() : existing.resolvedAt ?? null);
    const updated: TeacherProposal = {
      ...existing,
      status: params.status,
      resolvedAt: resolvedAt ?? undefined,
      freshnessTs: resolvedAt ?? existing.freshnessTs ?? existing.createdAt,
      proofBundle: params.proofBundle === undefined ? existing.proofBundle : params.proofBundle ?? undefined,
      replayGate: params.replayGate === undefined ? existing.replayGate : params.replayGate ?? undefined,
      canaryRollout: params.canaryRollout === undefined ? existing.canaryRollout : params.canaryRollout ?? undefined,
      replaySummary: params.replaySummary === undefined ? existing.replaySummary : params.replaySummary ?? undefined,
    };
    this.validateTeacherCanaryActivation(updated);
    const existingCreatedAt = Number.isFinite(Date.parse(existing.createdAt)) ? Date.parse(existing.createdAt) : Date.now();
    const record = this.toTeacherProposalRow(updated, existingCreatedAt, Date.now());

    this.db.prepare(`
      UPDATE brain_teacher_proposals
      SET proposal_class = ?,
          proposal_kind = ?,
          lane = ?,
          status = ?,
          lifecycle_state = ?,
          schema_version = ?,
          idempotency_key = ?,
          rollback_key = ?,
          scope = ?,
          base_pack_version = ?,
          base_graph_hash = ?,
          producer_version = ?,
          producer_build_id = ?,
          prompt_hash = ?,
          template_id = ?,
          profile = ?,
          source_bundle_id = ?,
          replay_suite_ids = ?,
          freshness_ts = ?,
          proposal_json = ?,
          updated_at = ?,
          resolved_at = ?
      WHERE proposal_id = ?
    `).run(
      record.proposalClass,
      record.proposalKind,
      record.lane,
      record.status,
      record.lifecycleState,
      record.schemaVersion,
      record.idempotencyKey,
      record.rollbackKey,
      record.scope,
      record.basePackVersion,
      record.baseGraphHash,
      record.producerVersion,
      record.producerBuildId,
      record.promptHash,
      record.templateId,
      record.profile,
      record.sourceBundleId,
      record.replaySuiteIds,
      record.freshnessTs,
      record.proposalJson,
      record.updatedAt,
      record.resolvedAt,
      record.proposalId,
    );
  }

  getTeacherProposal(proposalId: string): TeacherProposal | null {
    const row = this.db.prepare(`SELECT * FROM brain_teacher_proposals WHERE proposal_id = ?`).get(proposalId) as Record<string, unknown> | undefined;
    return row ? this.toTeacherProposal(row) : null;
  }

  getTeacherProposalByIdempotencyKey(idempotencyKey: string): TeacherProposal | null {
    const row = this.db.prepare(`SELECT * FROM brain_teacher_proposals WHERE idempotency_key = ?`).get(idempotencyKey) as Record<string, unknown> | undefined;
    return row ? this.toTeacherProposal(row) : null;
  }

  getTeacherProposalsByClass(proposalClass: TeacherProposal["proposalClass"], limit = 50): TeacherProposal[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_teacher_proposals
      WHERE proposal_class = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(proposalClass, limit) as Record<string, unknown>[];
    return rows.map((row) => this.toTeacherProposal(row));
  }

  getTeacherProposalsByKind(proposalKind: string, limit = 50): TeacherProposal[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_teacher_proposals
      WHERE proposal_kind = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(proposalKind, limit) as Record<string, unknown>[];
    return rows.map((row) => this.toTeacherProposal(row));
  }

  getTeacherProposalsByLifecycleState(lifecycleState: string, limit = 50): TeacherProposal[] {
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_teacher_proposals
      WHERE lifecycle_state = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).all(lifecycleState, limit) as Record<string, unknown>[];
    return rows.map((row) => this.toTeacherProposal(row));
  }

  summarizeTeacherProposal(proposalId: string): TeacherProposalSummaryV1 | null {
    const proposal = this.getTeacherProposal(proposalId);
    return proposal ? summarizeTeacherProposalV1(proposal) : null;
  }

  diffTeacherProposals(leftProposalId: string, rightProposalId: string): TeacherProposalDiffV1 | null {
    const left = this.getTeacherProposal(leftProposalId);
    const right = this.getTeacherProposal(rightProposalId);
    if (!left || !right) {
      return null;
    }
    return diffTeacherProposalV1(left, right);
  }

  appendLearningJournal(params: LearningJournalInsert): LearningJournalRecord {
    const id = `bj_${randomUUID().slice(0, 8)}`;
    const createdAt = params.createdAt ?? Date.now();
    this.db.prepare(`
      INSERT INTO brain_learning_journal (id, event_type, mutation_id, mutation_ids, bundle_id, pack_version, payload, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      params.eventType,
      params.mutationId ?? null,
      JSON.stringify(params.mutationIds ?? []),
      params.bundleId ?? null,
      params.packVersion ?? null,
      JSON.stringify(params.payload),
      createdAt,
    );
    return this.toLearningJournalRecord({
      id,
      event_type: params.eventType,
      mutation_id: params.mutationId ?? null,
      mutation_ids: JSON.stringify(params.mutationIds ?? []),
      bundle_id: params.bundleId ?? null,
      pack_version: params.packVersion ?? null,
      payload: JSON.stringify(params.payload),
      created_at: createdAt,
    });
  }

  getLearningJournal(query: LearningJournalQuery = {}): LearningJournalRecord[] {
    const filters: string[] = [];
    const args: Array<number | string> = [];

    if (query.eventTypes && query.eventTypes.length > 0) {
      filters.push(`event_type IN (${query.eventTypes.map(() => "?").join(", ")})`);
      args.push(...query.eventTypes);
    }
    if (query.bundleId) {
      filters.push("bundle_id = ?");
      args.push(query.bundleId);
    }
    if (typeof query.since === "number") {
      filters.push("created_at >= ?");
      args.push(query.since);
    }

    const whereClause = filters.length > 0 ? `WHERE ${filters.join(" AND ")}` : "";
    const rows = this.db.prepare(`
      SELECT *
      FROM brain_learning_journal
      ${whereClause}
      ORDER BY created_at ASC
    `).all(...args) as Record<string, unknown>[];

    let records = rows.map((row) => this.toLearningJournalRecord(row));
    if (query.mutationId) {
      const mutationId = query.mutationId;
      records = records.filter((record) =>
        record.mutationId === mutationId || record.mutationIds.includes(mutationId),
      );
    }
    if (typeof query.limit === "number" && query.limit >= 0) {
      records = records.slice(Math.max(0, records.length - query.limit));
    }
    return records;
  }

  countOrphanedTraceRows(): number {
    const row = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM brain_traces t
      LEFT JOIN brain_episodes e ON e.id = t.episode_id
      WHERE t.episode_id IS NOT NULL AND e.id IS NULL
    `).get() as { count: number };
    return row.count ?? 0;
  }

  // ─── Traces ───

  insertTrace(trace: DecisionTrace): void {
    this.db.prepare(`
      INSERT INTO brain_traces (id, episode_id, pack_version, query_text, seed_scores, trajectory, fired_nodes, vetoed_nodes, context_chars, footer, route_trace_json, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      trace.id, trace.episodeId, trace.packVersion, trace.queryText,
      JSON.stringify(trace.seedScores), JSON.stringify(trace.trajectory),
      JSON.stringify(trace.firedNodes), JSON.stringify(trace.vetoedNodes),
      trace.contextChars, trace.footer, JSON.stringify(trace.routeTrace ?? null), trace.createdAt,
    );
  }

  updateTraceSelectionMetadata(
    traceId: string,
    selectionMetadata: Partial<DecisionRouteTrace["selectionMetadata"]>,
  ): void {
    const row = this.db.prepare(`
      SELECT route_trace_json
      FROM brain_traces
      WHERE id = ?
    `).get(traceId) as { route_trace_json?: string } | undefined;
    if (!row || row.route_trace_json === undefined) {
      return;
    }

    const routeTrace = JSON.parse(row.route_trace_json || "null") as DecisionRouteTrace | null;
    if (!routeTrace?.selectionMetadata) {
      return;
    }

    this.db.prepare(`
      UPDATE brain_traces
      SET route_trace_json = ?
      WHERE id = ?
    `).run(
      JSON.stringify({
        ...routeTrace,
        selectionMetadata: {
          ...routeTrace.selectionMetadata,
          ...selectionMetadata,
        },
      }),
      traceId,
    );
  }

  getRecentTraces(limit: number): DecisionTrace[] {
    const rows = this.db.prepare(`SELECT * FROM brain_traces ORDER BY created_at DESC LIMIT ?`).all(limit) as Record<string, unknown>[];
    return rows.map((r) => this.toTrace(r));
  }

  getRecentDecisionSummary(limit: number): RecentDecisionTraceSummary {
    return summarizeRecentDecisionTraces(this.getRecentTraces(limit), limit);
  }

  countTraces(): number {
    const row = this.db.prepare(`SELECT COUNT(*) as count FROM brain_traces`).get() as { count: number };
    return row.count ?? 0;
  }

  getTrace(id: string): DecisionTrace | null {
    const row = this.db.prepare(`SELECT * FROM brain_traces WHERE id = ?`).get(id) as Record<string, unknown> | undefined;
    return row ? this.toTrace(row) : null;
  }

  getLatestTraceForEpisode(episodeId: string): DecisionTrace | null {
    const row = this.db.prepare(`
      SELECT *
      FROM brain_traces
      WHERE episode_id = ?
      ORDER BY created_at DESC
      LIMIT 1
    `).get(episodeId) as Record<string, unknown> | undefined;
    return row ? this.toTrace(row) : null;
  }

  getTraceForEpisode(episodeId: string): DecisionTrace | null {
    return this.getLatestTraceForEpisode(episodeId);
  }

  private toTrace(row: Record<string, unknown>): DecisionTrace {
    return {
      id: row.id as string,
      episodeId: (row.episode_id as string) ?? null,
      packVersion: (row.pack_version as number) ?? null,
      queryText: (row.query_text as string) || "",
      seedScores: parseJsonValue(row.seed_scores, []),
      trajectory: parseJsonValue(row.trajectory, []),
      firedNodes: parseJsonValue(row.fired_nodes, []),
      vetoedNodes: parseJsonValue(row.vetoed_nodes, []),
      contextChars: (row.context_chars as number) || 0,
      footer: (row.footer as string) || "",
      routeTrace: row.route_trace_json === undefined
        ? null
        : parseJsonValue(row.route_trace_json, null),
      supervision: [],
      createdAt: row.created_at as number,
    };
  }

  private toTraceSupervision(row: Record<string, unknown>): TraceSupervisionRecord {
    return {
      id: row.id as string,
      traceId: row.trace_id as string,
      episodeId: row.episode_id as string,
      conversationId: (row.conversation_id as number) ?? null,
      source: row.source as RewardSource,
      kind: row.kind as BrainEvidenceKind,
      value: row.value as number,
      confidence: (row.confidence as number) ?? 1.0,
      reason: (row.reason as string) ?? null,
      contentSnippet: (row.content_snippet as string) ?? null,
      resolution: row.resolution as BrainEvidenceResolution,
      labelId: (row.label_id as string) ?? null,
      evidenceId: (row.evidence_id as string) ?? null,
      metadata: parseJsonValue(row.metadata, {}),
      createdAt: row.created_at as number,
    };
  }

  private toObservation(row: Record<string, unknown>): BrainObservation {
    const traceId = toOptionalString(row.trace_id);
    const routeMetadata = observationRouteMetadataFromRow(row);
    return {
      id: row.id as string,
      episodeId: row.episode_id as string,
      conversationId: (row.conversation_id as number) ?? null,
      traceId,
      queryText: (row.query_text as string) ?? "",
      retrievedContext: parseJsonValue(row.retrieved_context_json, []),
      routeMetadata,
      assistantResponse: (row.assistant_response as string) ?? "",
      toolResults: parseJsonValue(row.tool_results_json, []),
      followUpText: (row.follow_up_text as string) ?? null,
      phase1Score: row.phase1_score === null ? null : Number(row.phase1_score),
      phase2Score: row.phase2_score === null ? null : Number(row.phase2_score),
      finalScore: row.final_score === null ? null : Number(row.final_score),
      confidence: row.confidence === null ? null : Number(row.confidence),
      reason: (row.reason as string) ?? null,
      status: row.status as BrainObservationStatus,
      teacherEvaluation: row.teacher_evaluation_json
        ? normalizeObservationTeacherEvaluation(parseJsonValue(row.teacher_evaluation_json, null), {
            observationId: row.id as string,
            episodeId: row.episode_id as string,
            traceId,
            routeMetadata,
          })
        : null,
      createdAt: Number(row.created_at ?? 0),
      updatedAt: Number(row.updated_at ?? 0),
      evaluatedAt: row.evaluated_at === null ? null : Number(row.evaluated_at),
    };
  }

  private toPack(row: Record<string, unknown>): Pack {
    return {
      version: row.version as number,
      nodeCount: row.node_count as number,
      edgeCount: row.edge_count as number,
      healthJson: row.health_json as string,
      promotedAt: (row.promoted_at as number) ?? null,
      rolledBack: !!(row.rolled_back as number),
      createdAt: row.created_at as number,
    };
  }

  private toTeacherProposal(row: Record<string, unknown>): TeacherProposal {
    const proposal = parseJsonValue<TeacherProposal | null>(row.proposal_json, null);
    if (!proposal) {
      throw new Error(`invalid teacher proposal payload for ${row.proposal_id as string}`);
    }
    return normalizeTeacherProposalV1(proposal);
  }

  private toTeacherProposalRow(
    proposal: TeacherProposal,
    createdAt: number,
    updatedAt: number,
  ): {
    schemaVersion: 1;
    proposalId: string;
    proposalClass: TeacherProposal["proposalClass"];
    proposalKind: string | null;
    lane: string | null;
    status: TeacherProposal["status"];
    lifecycleState: string | null;
    replaySuiteIds: string;
    freshnessTs: string | null;
    idempotencyKey: string;
    rollbackKey: string;
    scope: string;
    basePackVersion: number | null;
    baseGraphHash: string | null;
    producerVersion: string;
    producerBuildId: string | null;
    promptHash: string | null;
    templateId: string | null;
    profile: string | null;
    sourceBundleId: string | null;
    proposalJson: string;
    createdAt: number;
    updatedAt: number;
    resolvedAt: number | null;
  } {
    const toEpochMs = (value: string | undefined): number | null => {
      if (!value) {
        return null;
      }
      const parsed = Date.parse(value);
      return Number.isFinite(parsed) ? parsed : null;
    };

    const createdAtValue = toEpochMs(proposal.createdAt) ?? createdAt;
    const resolvedAtValue = toEpochMs(proposal.resolvedAt);
    const normalizedProposal = normalizeTeacherProposalV1(proposal);
    return {
      schemaVersion: normalizedProposal.schemaVersion ?? 1,
      proposalId: proposal.proposalId,
      proposalClass: proposal.proposalClass,
      proposalKind: normalizedProposal.proposalKind ?? null,
      lane: normalizedProposal.lane ?? null,
      status: normalizedProposal.status,
      lifecycleState: normalizedProposal.lifecycleState ?? null,
      replaySuiteIds: JSON.stringify(normalizedProposal.replaySuiteIds),
      freshnessTs: normalizedProposal.freshnessTs ?? null,
      idempotencyKey: normalizedProposal.lineage.idempotencyKey,
      rollbackKey: normalizedProposal.rollbackKey,
      scope: normalizedProposal.lineage.scope,
      basePackVersion: normalizedProposal.lineage.basePackVersion ?? null,
      baseGraphHash: normalizedProposal.lineage.baseGraphHash ?? null,
      producerVersion: normalizedProposal.lineage.producerVersion,
      producerBuildId: normalizedProposal.lineage.producerBuildId ?? null,
      promptHash: normalizedProposal.lineage.promptHash ?? null,
      templateId: normalizedProposal.lineage.templateId ?? null,
      profile: normalizedProposal.lineage.profile ?? null,
      sourceBundleId: normalizedProposal.lineage.sourceBundleId ?? null,
      proposalJson: JSON.stringify(normalizedProposal),
      createdAt: createdAtValue,
      updatedAt,
      resolvedAt: resolvedAtValue,
    };
  }

  private toMutationBundle(row: Record<string, unknown>): MutationBundleRecord {
    return {
      id: row.id as string,
      mutationIds: JSON.parse((row.mutation_ids as string) || "[]"),
      bundleSize: Number(row.bundle_size ?? 0),
      status: row.status as MutationBundleStatus,
      baseScore: row.base_score === null ? null : Number(row.base_score),
      candidateScore: row.candidate_score === null ? null : Number(row.candidate_score),
      expectedGain: Number(row.expected_gain ?? 0),
      rejectionReason: (row.rejection_reason as string) ?? null,
      verdict: row.verdict_json
        ? JSON.parse(row.verdict_json as string) as BundleEvaluationVerdict
        : null,
      createdAt: Number(row.created_at ?? 0),
      resolvedAt: row.resolved_at === null ? null : Number(row.resolved_at),
    };
  }

  private toLearningJournalRecord(row: Record<string, unknown>): LearningJournalRecord {
    const base = {
      id: row.id as string,
      mutationId: (row.mutation_id as string) ?? null,
      mutationIds: JSON.parse((row.mutation_ids as string) || "[]"),
      bundleId: (row.bundle_id as string) ?? null,
      packVersion: (row.pack_version as number) ?? null,
      createdAt: row.created_at as number,
    };
    const payloadRaw = JSON.parse((row.payload as string) || "{}");

    switch (row.event_type) {
      case "mutation_proposed":
        return {
          ...base,
          eventType: "mutation_proposed",
          payload: payloadRaw as MutationProposedJournalPayload,
        };
      case "bundle_evaluation_started":
        return {
          ...base,
          eventType: "bundle_evaluation_started",
          payload: payloadRaw as BundleEvaluationStartedJournalPayload,
        };
      case "bundle_evaluation_completed":
        return {
          ...base,
          eventType: "bundle_evaluation_completed",
          payload: payloadRaw as BundleEvaluationCompletedJournalPayload,
        };
      case "promotion_accepted":
        return {
          ...base,
          eventType: "promotion_accepted",
          payload: payloadRaw as PromotionJournalPayload,
        };
      case "promotion_rejected":
        return {
          ...base,
          eventType: "promotion_rejected",
          payload: payloadRaw as PromotionJournalPayload,
        };
      default:
        throw new Error(`Unknown learning journal event type: ${String(row.event_type)}`);
    }
  }

  // ─── Training State ───

  getTrainingState(key: string): string | null {
    const row = this.db.prepare(`SELECT value FROM brain_training_state WHERE key = ?`).get(key) as { value: string } | undefined;
    return row?.value ?? null;
  }

  getTrainingStateJson<T>(key: string): T | null {
    const raw = this.getTrainingState(key)?.trim();
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw) as T;
    } catch {
      return null;
    }
  }

  setTrainingState(key: string, value: string | number): void {
    this.db.prepare(`INSERT OR REPLACE INTO brain_training_state (key, value) VALUES (?, ?)`)
      .run(key, String(value));
  }

  setTrainingStateJson(key: string, value: unknown | null): void {
    this.setTrainingState(key, value === null ? "" : JSON.stringify(value));
  }

  // ─── Bulk load into BrainGraph ───

  loadAllEdges(): BrainEdge[] {
    const rows = this.db.prepare(`SELECT * FROM brain_edges`).all() as Record<string, unknown>[];
    return rows.map((r) => this.toEdge(r));
  }

  loadAllSeedWeights(): SeedWeight[] {
    return this.getAllSeedWeights();
  }

  loadAllStopLocalWeights(): StopLocalWeight[] {
    return this.getAllStopLocalWeights();
  }

  loadAllToolActionPriors(): Array<{ sourceNodeId: string; toolNodeId: string; weight: number; updatedAt: number }> {
    return this.getAllToolActionPriors();
  }

  getCurrentPackVersion(): number | null {
    if (!this.options.brainRoot) {
      return null;
    }
    const currentFile = this.getCurrentPackFile();
    if (!existsSync(currentFile)) {
      return null;
    }
    const value = readFileSync(currentFile, "utf8").trim();
    if (!value) {
      return null;
    }
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }

  writePackSnapshot(params: {
    version: number;
    nodes: BrainNode[];
    edges: BrainEdge[];
    seedWeights?: SeedWeight[];
    stopLocalWeights?: StopLocalWeight[];
    toolActionPriors?: Array<{ sourceNodeId: string; toolNodeId: string; weight: number; updatedAt: number }>;
    metadata: Record<string, unknown>;
  }): string {
    if (!this.options.brainRoot) {
      throw new Error("brainRoot is required to write pack snapshots");
    }

    const packDir = this.getPackDir(params.version);
    mkdirSync(packDir, { recursive: true });

    const manifest = {
      version: params.version,
      nodeCount: params.nodes.length,
      edgeCount: params.edges.length,
      createdAt: Date.now(),
    };
    writeFileSync(join(packDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
    writeFileSync(join(packDir, "metadata.json"), JSON.stringify(params.metadata, null, 2), "utf8");
    writeFileSync(
      join(packDir, "nodes.jsonl"),
      `${params.nodes.map((node) => JSON.stringify({
        ...node,
        embedding: node.embedding ? Array.from(node.embedding) : null,
      })).join("\n")}\n`,
      "utf8",
    );
    writeFileSync(
      join(packDir, "edges.jsonl"),
      `${params.edges.map((edge) => JSON.stringify(edge)).join("\n")}\n`,
      "utf8",
    );
    writeFileSync(
      join(packDir, "seed-weights.jsonl"),
      `${(params.seedWeights ?? []).map((seedWeight) => JSON.stringify(seedWeight)).join("\n")}${(params.seedWeights ?? []).length > 0 ? "\n" : ""}`,
      "utf8",
    );
    writeFileSync(
      join(packDir, "stop-local-weights.jsonl"),
      `${(params.stopLocalWeights ?? []).map((stopLocalWeight) => JSON.stringify(stopLocalWeight)).join("\n")}${(params.stopLocalWeights ?? []).length > 0 ? "\n" : ""}`,
      "utf8",
    );
    writeFileSync(
      join(packDir, "tool-action-priors.jsonl"),
      `${(params.toolActionPriors ?? []).map((toolActionPrior) => JSON.stringify(toolActionPrior)).join("\n")}${(params.toolActionPriors ?? []).length > 0 ? "\n" : ""}`,
      "utf8",
    );

    return packDir;
  }

  readPackSnapshot(version: number): {
    nodes: BrainNode[];
    edges: BrainEdge[];
    seedWeights: SeedWeight[];
    stopLocalWeights: StopLocalWeight[];
    toolActionPriors: Array<{ sourceNodeId: string; toolNodeId: string; weight: number; updatedAt: number }>;
    metadata: Record<string, unknown>;
  } | null {
    if (!this.options.brainRoot) {
      return null;
    }

    const packDir = this.getPackDir(version);
    const nodesFile = join(packDir, "nodes.jsonl");
    const edgesFile = join(packDir, "edges.jsonl");
    const seedWeightsFile = join(packDir, "seed-weights.jsonl");
    const stopLocalWeightsFile = join(packDir, "stop-local-weights.jsonl");
    const toolActionPriorsFile = join(packDir, "tool-action-priors.jsonl");
    const metadataFile = join(packDir, "metadata.json");
    if (!existsSync(nodesFile) || !existsSync(edgesFile)) {
      return null;
    }

    const parseJsonl = (value: string): Record<string, unknown>[] =>
      value
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line) as Record<string, unknown>);

    const nodes = parseJsonl(readFileSync(nodesFile, "utf8")).map((row) => ({
      ...(row as unknown as BrainNode),
      embedding: Array.isArray(row.embedding)
        ? new Float32Array((row.embedding as number[]).map((value) => Number(value)))
        : null,
    }));
    const edges = parseJsonl(readFileSync(edgesFile, "utf8")) as unknown as BrainEdge[];
    const seedWeights = existsSync(seedWeightsFile)
      ? parseJsonl(readFileSync(seedWeightsFile, "utf8")).map((row) => ({
          nodeId: row.nodeId as string,
          weight: Number(row.weight ?? 0),
          updatedAt: Number(row.updatedAt ?? 0),
        }))
      : [];
    const stopLocalWeights = existsSync(stopLocalWeightsFile)
      ? parseJsonl(readFileSync(stopLocalWeightsFile, "utf8")).map((row) => ({
          sourceNodeId: row.sourceNodeId as string,
          weight: Number(row.weight ?? 0),
          updatedAt: Number(row.updatedAt ?? 0),
        }))
      : [];
    const toolActionPriors = existsSync(toolActionPriorsFile)
      ? parseJsonl(readFileSync(toolActionPriorsFile, "utf8")).map((row) => ({
          sourceNodeId: row.sourceNodeId as string,
          toolNodeId: row.toolNodeId as string,
          weight: Number(row.weight ?? 0),
          updatedAt: Number(row.updatedAt ?? 0),
        }))
      : [];
    const metadata = existsSync(metadataFile)
      ? JSON.parse(readFileSync(metadataFile, "utf8")) as Record<string, unknown>
      : {};

    return { nodes, edges, seedWeights, stopLocalWeights, toolActionPriors, metadata };
  }

  private getPacksDir(): string {
    if (!this.options.brainRoot) {
      throw new Error("brainRoot is not configured");
    }
    return join(this.options.brainRoot, "packs");
  }

  private getCurrentPackFile(): string {
    if (!this.options.brainRoot) {
      throw new Error("brainRoot is not configured");
    }
    return join(this.options.brainRoot, "current");
  }

  private getPackDir(version: number): string {
    return join(this.getPacksDir(), `v${version.toString().padStart(6, "0")}`);
  }
}
