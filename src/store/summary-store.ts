import type { DatabaseSync } from "node:sqlite";
import { sanitizeFts5Query } from "./fts5-sanitize.js";
import { buildLikeSearchPlan, createFallbackSnippet } from "./full-text-fallback.js";

export type SummaryKind = "leaf" | "condensed";
export type SummaryLineageRole = "support" | "episode" | "snapshot";
export type SummaryTruthBasis = "canonical" | "derived" | "open";
export type SummaryFreshnessState =
  | "fresh"
  | "stale_source"
  | "stale_branch"
  | "stale_pack"
  | "superseded"
  | "tombstoned";
export type ContextItemType = "message" | "summary";

export type MarbleKind = "replay_stub" | "typed_extraction" | "operational_summary";
export type MarbleFreshnessState =
  | "fresh"
  | "stale_source"
  | "stale_policy"
  | "stale_pack"
  | "superseded"
  | "tombstoned";
export type MarbleSourceKind = "message" | "summary" | "file" | "tool_result" | "trace";

export type CreateMarbleSourceInput = {
  sourceKind: MarbleSourceKind;
  sourceId: string;
  sourceSubId?: string | null;
  sourceDigest: string;
  sourceProvenanceRef: string;
  sourceUri?: string | null;
  ordinal?: number;
};

export type CreateMarbleInput = {
  marbleId: string;
  conversationId: number;
  marbleKind: MarbleKind;
  compressionVersion: number;
  renderVersion: number;
  content: string;
  payloadJson: string;
  tokenCount: number;
  confidence: number;
  freshnessState: MarbleFreshnessState;
  sourceFingerprint: string;
  contentHash: string;
  provenanceRef: string;
  sourceArtifactTokenCount?: number;
  derivedFromMarbleId?: string | null;
  invalidatedAt?: Date | null;
  invalidationReason?: string | null;
  createdAt?: Date;
  updatedAt?: Date;
  sources?: CreateMarbleSourceInput[];
};

export type MarbleRecord = {
  marbleId: string;
  conversationId: number;
  marbleKind: MarbleKind;
  compressionVersion: number;
  renderVersion: number;
  content: string;
  payloadJson: string;
  tokenCount: number;
  confidence: number;
  freshnessState: MarbleFreshnessState;
  sourceFingerprint: string;
  contentHash: string;
  provenanceRef: string;
  sourceCount: number;
  sourceArtifactTokenCount: number;
  derivedFromMarbleId: string | null;
  invalidatedAt: Date | null;
  invalidationReason: string | null;
  createdAt: Date;
  updatedAt: Date;
};

export type MarbleSourceRecord = {
  marbleId: string;
  sourceKind: MarbleSourceKind;
  sourceId: string;
  sourceSubId: string | null;
  sourceDigest: string;
  sourceProvenanceRef: string;
  sourceUri: string | null;
  ordinal: number;
};

export type MarbleSearchInput = {
  conversationId?: number;
  query: string;
  mode: "regex" | "full_text";
  since?: Date;
  before?: Date;
  limit?: number;
};

export type MarbleSearchResult = {
  marbleId: string;
  conversationId: number;
  marbleKind: MarbleKind;
  freshnessState: MarbleFreshnessState;
  provenanceRef: string;
  sourceFingerprint: string;
  sourceCount: number;
  sourceArtifactTokenCount: number;
  snippet: string;
  createdAt: Date;
  rank?: number;
};

export type CreateSummaryInput = {
  summaryId: string;
  conversationId: number;
  kind: SummaryKind;
  depth?: number;
  content: string;
  tokenCount: number;
  fileIds?: string[];
  earliestAt?: Date;
  latestAt?: Date;
  descendantCount?: number;
  descendantTokenCount?: number;
  sourceMessageTokenCount?: number;
};

export type CreateSummaryLineageInput = {
  summaryId: string;
  conversationId: number;
  branchId: string;
  episodeId: string;
  summaryRole: SummaryLineageRole;
  truthBasis: SummaryTruthBasis;
  freshnessState?: SummaryFreshnessState;
  parentBranchId?: string | null;
  typedMemoryRefs?: string[];
  snapshotId?: string | null;
  forkReason?: string | null;
  invalidatedAt?: Date | null;
  invalidationReason?: string | null;
  createdAt?: Date;
};

export type SummaryLineageRecord = {
  summaryId: string;
  conversationId: number;
  branchId: string;
  episodeId: string;
  summaryRole: SummaryLineageRole;
  truthBasis: SummaryTruthBasis;
  freshnessState: SummaryFreshnessState;
  parentBranchId: string | null;
  typedMemoryRefs: string[];
  snapshotId: string | null;
  forkReason: string | null;
  invalidatedAt: Date | null;
  invalidationReason: string | null;
  createdAt: Date;
};

export type CreateBranchSnapshotInput = {
  snapshotId: string;
  conversationId: number;
  branchId: string;
  episodeId: string;
  activeSummaryId?: string | null;
  contextOrdinal: number;
  packVersion?: number | null;
  summarySpineIds?: string[];
  typedMemoryRefs?: string[];
  openQuestionRefs?: string[];
  stateJson: string;
  createdAt?: Date;
};

export type BranchSnapshotRecord = {
  snapshotId: string;
  conversationId: number;
  branchId: string;
  episodeId: string;
  activeSummaryId: string | null;
  contextOrdinal: number;
  packVersion: number | null;
  summarySpineIds: string[];
  typedMemoryRefs: string[];
  openQuestionRefs: string[];
  stateJson: string;
  createdAt: Date;
};

export type SummaryRecord = {
  summaryId: string;
  conversationId: number;
  kind: SummaryKind;
  depth: number;
  content: string;
  tokenCount: number;
  fileIds: string[];
  earliestAt: Date | null;
  latestAt: Date | null;
  descendantCount: number;
  descendantTokenCount: number;
  sourceMessageTokenCount: number;
  createdAt: Date;
};

export type SummarySubtreeNodeRecord = SummaryRecord & {
  depthFromRoot: number;
  parentSummaryId: string | null;
  path: string;
  childCount: number;
  freshnessState?: SummaryFreshnessState;
};

export type ContextItemRecord = {
  conversationId: number;
  ordinal: number;
  itemType: ContextItemType;
  messageId: number | null;
  summaryId: string | null;
  createdAt: Date;
};

export type SummarySearchInput = {
  conversationId?: number;
  query: string;
  mode: "regex" | "full_text";
  since?: Date;
  before?: Date;
  limit?: number;
};

export type SummarySearchResult = {
  summaryId: string;
  conversationId: number;
  kind: SummaryKind;
  freshnessState?: SummaryFreshnessState;
  snippet: string;
  createdAt: Date;
  rank?: number;
};

export type CreateLargeFileInput = {
  fileId: string;
  conversationId: number;
  fileName?: string;
  mimeType?: string;
  byteSize?: number;
  storageUri: string;
  explorationSummary?: string;
};

export type LargeFileRecord = {
  fileId: string;
  conversationId: number;
  fileName: string | null;
  mimeType: string | null;
  byteSize: number | null;
  storageUri: string;
  explorationSummary: string | null;
  createdAt: Date;
};

// ── DB row shapes (snake_case) ────────────────────────────────────────────────

interface SummaryRow {
  summary_id: string;
  conversation_id: number;
  kind: SummaryKind;
  depth: number;
  content: string;
  token_count: number;
  file_ids: string;
  earliest_at: string | null;
  latest_at: string | null;
  descendant_count: number | null;
  descendant_token_count: number | null;
  source_message_token_count: number | null;
  created_at: string;
}

interface SummaryLineageRow {
  summary_id: string;
  conversation_id: number;
  branch_id: string;
  episode_id: string;
  summary_role: SummaryLineageRole;
  truth_basis: SummaryTruthBasis;
  freshness_state: SummaryFreshnessState;
  parent_branch_id: string | null;
  typed_memory_refs: string;
  snapshot_id: string | null;
  fork_reason: string | null;
  invalidated_at: string | null;
  invalidation_reason: string | null;
  created_at: string;
}

interface BranchSnapshotRow {
  snapshot_id: string;
  conversation_id: number;
  branch_id: string;
  episode_id: string;
  active_summary_id: string | null;
  context_ordinal: number;
  pack_version: number | null;
  summary_spine_ids: string;
  typed_memory_refs: string;
  open_question_refs: string;
  state_json: string;
  created_at: string;
}

interface SummarySubtreeRow extends SummaryRow {
  freshness_state: SummaryFreshnessState | null;
  depth_from_root: number;
  parent_summary_id: string | null;
  path: string;
  child_count: number | null;
}

interface ContextItemRow {
  conversation_id: number;
  ordinal: number;
  item_type: ContextItemType;
  message_id: number | null;
  summary_id: string | null;
  created_at: string;
}

interface SummarySearchRow {
  summary_id: string;
  conversation_id: number;
  kind: SummaryKind;
  freshness_state: SummaryFreshnessState | null;
  snippet: string;
  rank: number;
  created_at: string;
}

interface MaxOrdinalRow {
  max_ordinal: number;
}

interface DistinctDepthRow {
  depth: number;
}

interface TokenSumRow {
  total: number;
}

interface MessageIdRow {
  message_id: number;
}

interface LargeFileRow {
  file_id: string;
  conversation_id: number;
  file_name: string | null;
  mime_type: string | null;
  byte_size: number | null;
  storage_uri: string;
  exploration_summary: string | null;
  created_at: string;
}

interface MarbleRow {
  marble_id: string;
  conversation_id: number;
  marble_kind: MarbleKind;
  compression_version: number;
  render_version: number;
  content: string;
  payload_json: string;
  token_count: number;
  confidence: number;
  freshness_state: MarbleFreshnessState;
  source_fingerprint: string;
  content_hash: string;
  provenance_ref: string;
  source_count: number | null;
  source_artifact_token_count: number | null;
  derived_from_marble_id: string | null;
  invalidated_at: string | null;
  invalidation_reason: string | null;
  created_at: string;
  updated_at: string | null;
}

interface MarbleSourceRow {
  marble_id: string;
  source_kind: MarbleSourceKind;
  source_id: string;
  source_sub_id: string | null;
  source_digest: string;
  source_provenance_ref: string;
  source_uri: string | null;
  ordinal: number;
}

interface MarbleSearchRow {
  marble_id: string;
  conversation_id: number;
  marble_kind: MarbleKind;
  freshness_state: MarbleFreshnessState;
  provenance_ref: string;
  source_fingerprint: string;
  source_count: number;
  source_artifact_token_count: number;
  snippet: string;
  rank: number;
  created_at: string;
}

// ── Row mappers ───────────────────────────────────────────────────────────────

function toSummaryRecord(row: SummaryRow): SummaryRecord {
  let fileIds: string[] = [];
  try {
    fileIds = JSON.parse(row.file_ids);
  } catch {
    // ignore malformed JSON
  }
  return {
    summaryId: row.summary_id,
    conversationId: row.conversation_id,
    kind: row.kind,
    depth: row.depth,
    content: row.content,
    tokenCount: row.token_count,
    fileIds,
    earliestAt: row.earliest_at ? new Date(row.earliest_at) : null,
    latestAt: row.latest_at ? new Date(row.latest_at) : null,
    descendantCount:
      typeof row.descendant_count === "number" &&
      Number.isFinite(row.descendant_count) &&
      row.descendant_count >= 0
        ? Math.floor(row.descendant_count)
        : 0,
    descendantTokenCount:
      typeof row.descendant_token_count === "number" &&
      Number.isFinite(row.descendant_token_count) &&
      row.descendant_token_count >= 0
        ? Math.floor(row.descendant_token_count)
        : 0,
    sourceMessageTokenCount:
      typeof row.source_message_token_count === "number" &&
      Number.isFinite(row.source_message_token_count) &&
      row.source_message_token_count >= 0
        ? Math.floor(row.source_message_token_count)
        : 0,
    createdAt: new Date(row.created_at),
  };
}

function parseStringArrayJson(value: string | null | undefined): string[] {
  if (typeof value !== "string" || value.trim().length === 0) {
    return [];
  }
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((entry): entry is string => typeof entry === "string");
  } catch {
    return [];
  }
}

function normalizeSummaryFreshnessState(value: string | null | undefined): SummaryFreshnessState {
  switch (value) {
    case "fresh":
    case "stale_source":
    case "stale_branch":
    case "stale_pack":
    case "superseded":
    case "tombstoned":
      return value;
    default:
      return "fresh";
  }
}

function dedupeStringArray(values: string[]): string[] {
  return [...new Set(values.filter((value) => typeof value === "string" && value.length > 0))];
}

function makeDefaultSummaryLineage(summary: SummaryRecord): SummaryLineageRecord {
  return {
    summaryId: summary.summaryId,
    conversationId: summary.conversationId,
    branchId: `branch_${summary.conversationId}_main`,
    episodeId: `episode_${summary.conversationId}_${summary.summaryId}`,
    summaryRole: summary.kind === "leaf" ? "support" : "episode",
    truthBasis: "derived",
    freshnessState: "fresh",
    parentBranchId: null,
    typedMemoryRefs: [],
    snapshotId: null,
    forkReason: null,
    invalidatedAt: null,
    invalidationReason: null,
    createdAt: summary.createdAt,
  };
}

function toSummaryLineageRecord(row: SummaryLineageRow): SummaryLineageRecord {
  return {
    summaryId: row.summary_id,
    conversationId: row.conversation_id,
    branchId: row.branch_id,
    episodeId: row.episode_id,
    summaryRole: row.summary_role,
    truthBasis: row.truth_basis,
    freshnessState: normalizeSummaryFreshnessState(row.freshness_state),
    parentBranchId: row.parent_branch_id,
    typedMemoryRefs: parseStringArrayJson(row.typed_memory_refs),
    snapshotId: row.snapshot_id,
    forkReason: row.fork_reason,
    invalidatedAt: normalizeNullableDate(row.invalidated_at),
    invalidationReason: row.invalidation_reason,
    createdAt: new Date(row.created_at),
  };
}

function toBranchSnapshotRecord(row: BranchSnapshotRow): BranchSnapshotRecord {
  return {
    snapshotId: row.snapshot_id,
    conversationId: row.conversation_id,
    branchId: row.branch_id,
    episodeId: row.episode_id,
    activeSummaryId: row.active_summary_id,
    contextOrdinal: Math.max(0, Math.floor(row.context_ordinal)),
    packVersion:
      typeof row.pack_version === "number" && Number.isFinite(row.pack_version)
        ? Math.max(0, Math.floor(row.pack_version))
        : null,
    summarySpineIds: parseStringArrayJson(row.summary_spine_ids),
    typedMemoryRefs: parseStringArrayJson(row.typed_memory_refs),
    openQuestionRefs: parseStringArrayJson(row.open_question_refs),
    stateJson: row.state_json,
    createdAt: new Date(row.created_at),
  };
}

function toContextItemRecord(row: ContextItemRow): ContextItemRecord {
  return {
    conversationId: row.conversation_id,
    ordinal: row.ordinal,
    itemType: row.item_type,
    messageId: row.message_id,
    summaryId: row.summary_id,
    createdAt: new Date(row.created_at),
  };
}

function toSearchResult(row: SummarySearchRow): SummarySearchResult {
  return {
    summaryId: row.summary_id,
    conversationId: row.conversation_id,
    kind: row.kind,
    freshnessState: normalizeSummaryFreshnessState(row.freshness_state),
    snippet: row.snippet,
    createdAt: new Date(row.created_at),
    rank: row.rank,
  };
}

function toLargeFileRecord(row: LargeFileRow): LargeFileRecord {
  return {
    fileId: row.file_id,
    conversationId: row.conversation_id,
    fileName: row.file_name,
    mimeType: row.mime_type,
    byteSize: row.byte_size,
    storageUri: row.storage_uri,
    explorationSummary: row.exploration_summary,
    createdAt: new Date(row.created_at),
  };
}

function normalizeNullableDate(value: string | null | undefined): Date | null {
  if (typeof value !== "string" || value.trim().length === 0) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function normalizeNonNegativeInteger(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.floor(value)
    : 0;
}

function normalizeConfidence(value: number | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

function normalizeOptionalSubId(value: string | null | undefined): string {
  return typeof value === "string" ? value : "";
}

function toMarbleRecord(row: MarbleRow): MarbleRecord {
  return {
    marbleId: row.marble_id,
    conversationId: row.conversation_id,
    marbleKind: row.marble_kind,
    compressionVersion: row.compression_version,
    renderVersion: row.render_version,
    content: row.content,
    payloadJson: row.payload_json,
    tokenCount: row.token_count,
    confidence: normalizeConfidence(row.confidence),
    freshnessState: row.freshness_state,
    sourceFingerprint: row.source_fingerprint,
    contentHash: row.content_hash,
    provenanceRef: row.provenance_ref,
    sourceCount: normalizeNonNegativeInteger(row.source_count),
    sourceArtifactTokenCount: normalizeNonNegativeInteger(row.source_artifact_token_count),
    derivedFromMarbleId: row.derived_from_marble_id,
    invalidatedAt: normalizeNullableDate(row.invalidated_at),
    invalidationReason: row.invalidation_reason,
    createdAt: new Date(row.created_at),
    updatedAt: normalizeNullableDate(row.updated_at) ?? new Date(row.created_at),
  };
}

function toMarbleSourceRecord(row: MarbleSourceRow): MarbleSourceRecord {
  return {
    marbleId: row.marble_id,
    sourceKind: row.source_kind,
    sourceId: row.source_id,
    sourceSubId: row.source_sub_id,
    sourceDigest: row.source_digest,
    sourceProvenanceRef: row.source_provenance_ref,
    sourceUri: row.source_uri,
    ordinal: normalizeNonNegativeInteger(row.ordinal),
  };
}

function toMarbleSearchResult(row: MarbleSearchRow): MarbleSearchResult {
  return {
    marbleId: row.marble_id,
    conversationId: row.conversation_id,
    marbleKind: row.marble_kind,
    freshnessState: row.freshness_state,
    provenanceRef: row.provenance_ref,
    sourceFingerprint: row.source_fingerprint,
    sourceCount: normalizeNonNegativeInteger(row.source_count),
    sourceArtifactTokenCount: normalizeNonNegativeInteger(row.source_artifact_token_count),
    snippet: row.snippet,
    createdAt: new Date(row.created_at),
    rank: row.rank,
  };
}

// ── SummaryStore ──────────────────────────────────────────────────────────────

export class SummaryStore {
  private readonly fts5Available: boolean;

  constructor(
    private db: DatabaseSync,
    options?: { fts5Available?: boolean },
  ) {
    this.fts5Available = options?.fts5Available ?? true;
  }

  // ── Summary CRUD ──────────────────────────────────────────────────────────

  async insertSummary(input: CreateSummaryInput): Promise<SummaryRecord> {
    const fileIds = JSON.stringify(input.fileIds ?? []);
    const earliestAt = input.earliestAt instanceof Date ? input.earliestAt.toISOString() : null;
    const latestAt = input.latestAt instanceof Date ? input.latestAt.toISOString() : null;
    const descendantCount =
      typeof input.descendantCount === "number" &&
      Number.isFinite(input.descendantCount) &&
      input.descendantCount >= 0
        ? Math.floor(input.descendantCount)
        : 0;
    const descendantTokenCount =
      typeof input.descendantTokenCount === "number" &&
      Number.isFinite(input.descendantTokenCount) &&
      input.descendantTokenCount >= 0
        ? Math.floor(input.descendantTokenCount)
        : 0;
    const sourceMessageTokenCount =
      typeof input.sourceMessageTokenCount === "number" &&
      Number.isFinite(input.sourceMessageTokenCount) &&
      input.sourceMessageTokenCount >= 0
        ? Math.floor(input.sourceMessageTokenCount)
        : 0;
    const depth =
      typeof input.depth === "number" && Number.isFinite(input.depth) && input.depth >= 0
        ? Math.floor(input.depth)
        : input.kind === "leaf"
          ? 0
          : 1;

    this.db
      .prepare(
        `INSERT INTO summaries (
          summary_id,
          conversation_id,
          kind,
          depth,
          content,
          token_count,
          file_ids,
          earliest_at,
          latest_at,
          descendant_count,
          descendant_token_count,
          source_message_token_count
        )
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        input.summaryId,
        input.conversationId,
        input.kind,
        depth,
        input.content,
        input.tokenCount,
        fileIds,
        earliestAt,
        latestAt,
        descendantCount,
        descendantTokenCount,
        sourceMessageTokenCount,
      );

    const row = this.db
      .prepare(
        `SELECT summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                earliest_at, latest_at, descendant_count, created_at
                , descendant_token_count, source_message_token_count
       FROM summaries WHERE summary_id = ?`,
      )
      .get(input.summaryId) as unknown as SummaryRow;

    // Index in FTS5 as best-effort; compaction flow must continue even if
    // FTS indexing fails for any reason.
    if (!this.fts5Available) {
      return toSummaryRecord(row);
    }

    try {
      this.db
        .prepare(`INSERT INTO summaries_fts(summary_id, content) VALUES (?, ?)`)
        .run(input.summaryId, input.content);
    } catch {
      // FTS indexing failed — search won't find this summary but
      // compaction and assembly will still work correctly.
    }

    return toSummaryRecord(row);
  }

  async getSummary(summaryId: string): Promise<SummaryRecord | null> {
    const row = this.db
      .prepare(
        `SELECT summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                earliest_at, latest_at, descendant_count, created_at
                , descendant_token_count, source_message_token_count
       FROM summaries WHERE summary_id = ?`,
      )
      .get(summaryId) as unknown as SummaryRow | undefined;
    return row ? toSummaryRecord(row) : null;
  }

  async getSummariesByConversation(conversationId: number): Promise<SummaryRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                earliest_at, latest_at, descendant_count, created_at
                , descendant_token_count, source_message_token_count
       FROM summaries
       WHERE conversation_id = ?
       ORDER BY created_at`,
      )
      .all(conversationId) as unknown as SummaryRow[];
    return rows.map(toSummaryRecord);
  }

  async getSummaryLineage(summaryId: string): Promise<SummaryLineageRecord | null> {
    const row = this.db
      .prepare(
        `SELECT summary_id, conversation_id, branch_id, episode_id, summary_role,
                truth_basis, freshness_state, parent_branch_id, typed_memory_refs, snapshot_id,
                fork_reason, invalidated_at, invalidation_reason, created_at
         FROM summary_lineage
         WHERE summary_id = ?`,
      )
      .get(summaryId) as unknown as SummaryLineageRow | undefined;
    if (row) {
      return toSummaryLineageRecord(row);
    }

    const summary = await this.getSummary(summaryId);
    return summary ? makeDefaultSummaryLineage(summary) : null;
  }

  async getSummaryLineageByConversation(conversationId: number): Promise<SummaryLineageRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT summary_id, conversation_id, branch_id, episode_id, summary_role,
                truth_basis, freshness_state, parent_branch_id, typed_memory_refs, snapshot_id,
                fork_reason, invalidated_at, invalidation_reason, created_at
         FROM summary_lineage
         WHERE conversation_id = ?
         ORDER BY created_at ASC`,
      )
      .all(conversationId) as unknown as SummaryLineageRow[];
    return rows.map(toSummaryLineageRecord);
  }

  async insertSummaryLineage(input: CreateSummaryLineageInput): Promise<SummaryLineageRecord> {
    const createdAt = (input.createdAt ?? new Date()).toISOString();
    const typedMemoryRefs = JSON.stringify(dedupeStringArray(input.typedMemoryRefs ?? []));
    const freshnessState = input.freshnessState ?? "fresh";
    const invalidatedAt = input.invalidatedAt ? input.invalidatedAt.toISOString() : null;
    const invalidationReason = input.invalidationReason ?? null;
    this.db
      .prepare(
        `INSERT INTO summary_lineage (
          summary_id,
          conversation_id,
          branch_id,
          episode_id,
          summary_role,
          truth_basis,
          freshness_state,
          parent_branch_id,
          typed_memory_refs,
          snapshot_id,
          fork_reason,
          invalidated_at,
          invalidation_reason,
          created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(summary_id) DO UPDATE SET
          conversation_id = excluded.conversation_id,
          branch_id = excluded.branch_id,
          episode_id = excluded.episode_id,
          summary_role = excluded.summary_role,
          truth_basis = excluded.truth_basis,
          freshness_state = excluded.freshness_state,
          parent_branch_id = excluded.parent_branch_id,
          typed_memory_refs = excluded.typed_memory_refs,
          snapshot_id = excluded.snapshot_id,
          fork_reason = excluded.fork_reason,
          invalidated_at = excluded.invalidated_at,
          invalidation_reason = excluded.invalidation_reason`,
      )
      .run(
        input.summaryId,
        input.conversationId,
        input.branchId,
        input.episodeId,
        input.summaryRole,
        input.truthBasis,
        freshnessState,
        input.parentBranchId ?? null,
        typedMemoryRefs,
        input.snapshotId ?? null,
        input.forkReason ?? null,
        invalidatedAt,
        invalidationReason,
        createdAt,
      );

    const lineage = await this.getSummaryLineage(input.summaryId);
    if (!lineage) {
      throw new Error(`Summary lineage not found after insert: ${input.summaryId}`);
    }
    return lineage;
  }

  async invalidateSummaryLineage(input: {
    summaryId: string;
    freshnessState: Exclude<SummaryFreshnessState, "fresh">;
    reason: string;
    invalidatedAt?: Date;
  }): Promise<SummaryLineageRecord | null> {
    const invalidatedAt = (input.invalidatedAt ?? new Date()).toISOString();
    this.db
      .prepare(
        `UPDATE summary_lineage
         SET freshness_state = ?, invalidated_at = ?, invalidation_reason = ?
         WHERE summary_id = ?`,
      )
      .run(input.freshnessState, invalidatedAt, input.reason, input.summaryId);
    return this.getSummaryLineage(input.summaryId);
  }

  async invalidateSummaryLineages(input: {
    summaryIds: string[];
    freshnessState: Exclude<SummaryFreshnessState, "fresh">;
    reason: string;
    invalidatedAt?: Date;
  }): Promise<SummaryLineageRecord[]> {
    const summaryIds = dedupeStringArray(input.summaryIds);
    if (summaryIds.length === 0) {
      return [];
    }

    const invalidatedAt = input.invalidatedAt ?? new Date();
    this.db.exec("BEGIN");
    try {
      const stmt = this.db.prepare(
        `UPDATE summary_lineage
         SET freshness_state = ?, invalidated_at = ?, invalidation_reason = ?
         WHERE summary_id = ?`,
      );
      for (const summaryId of summaryIds) {
        stmt.run(input.freshnessState, invalidatedAt.toISOString(), input.reason, summaryId);
      }
      this.db.exec("COMMIT");
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }

    const refreshed: SummaryLineageRecord[] = [];
    for (const summaryId of summaryIds) {
      const lineage = await this.getSummaryLineage(summaryId);
      if (lineage) {
        refreshed.push(lineage);
      }
    }
    return refreshed;
  }

  // ── Lineage ───────────────────────────────────────────────────────────────

  async linkSummaryToMessages(summaryId: string, messageIds: number[]): Promise<void> {
    if (messageIds.length === 0) {
      return;
    }

    const stmt = this.db.prepare(
      `INSERT INTO summary_messages (summary_id, message_id, ordinal)
       VALUES (?, ?, ?)
       ON CONFLICT (summary_id, message_id) DO NOTHING`,
    );

    for (let idx = 0; idx < messageIds.length; idx++) {
      stmt.run(summaryId, messageIds[idx], idx);
    }
  }

  async linkSummaryToParents(summaryId: string, parentSummaryIds: string[]): Promise<void> {
    if (parentSummaryIds.length === 0) {
      return;
    }

    const stmt = this.db.prepare(
      `INSERT INTO summary_parents (summary_id, parent_summary_id, ordinal)
       VALUES (?, ?, ?)
       ON CONFLICT (summary_id, parent_summary_id) DO NOTHING`,
    );

    for (let idx = 0; idx < parentSummaryIds.length; idx++) {
      stmt.run(summaryId, parentSummaryIds[idx], idx);
    }
  }

  async getSummaryMessages(summaryId: string): Promise<number[]> {
    const rows = this.db
      .prepare(
        `SELECT message_id FROM summary_messages
       WHERE summary_id = ?
       ORDER BY ordinal`,
      )
      .all(summaryId) as unknown as MessageIdRow[];
    return rows.map((r) => r.message_id);
  }

  async insertBranchSnapshot(input: CreateBranchSnapshotInput): Promise<BranchSnapshotRecord> {
    const createdAt = (input.createdAt ?? new Date()).toISOString();
    const summarySpineIds = JSON.stringify(dedupeStringArray(input.summarySpineIds ?? []));
    const typedMemoryRefs = JSON.stringify(dedupeStringArray(input.typedMemoryRefs ?? []));
    const openQuestionRefs = JSON.stringify(dedupeStringArray(input.openQuestionRefs ?? []));

    this.db
      .prepare(
        `INSERT INTO branch_snapshots (
          snapshot_id,
          conversation_id,
          branch_id,
          episode_id,
          active_summary_id,
          context_ordinal,
          pack_version,
          summary_spine_ids,
          typed_memory_refs,
          open_question_refs,
          state_json,
          created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(snapshot_id) DO UPDATE SET
          conversation_id = excluded.conversation_id,
          branch_id = excluded.branch_id,
          episode_id = excluded.episode_id,
          active_summary_id = excluded.active_summary_id,
          context_ordinal = excluded.context_ordinal,
          pack_version = excluded.pack_version,
          summary_spine_ids = excluded.summary_spine_ids,
          typed_memory_refs = excluded.typed_memory_refs,
          open_question_refs = excluded.open_question_refs,
          state_json = excluded.state_json`,
      )
      .run(
        input.snapshotId,
        input.conversationId,
        input.branchId,
        input.episodeId,
        input.activeSummaryId ?? null,
        Math.max(0, Math.floor(input.contextOrdinal)),
        typeof input.packVersion === "number" && Number.isFinite(input.packVersion)
          ? Math.max(0, Math.floor(input.packVersion))
          : null,
        summarySpineIds,
        typedMemoryRefs,
        openQuestionRefs,
        input.stateJson,
        createdAt,
      );

    const row = this.db
      .prepare(
        `SELECT snapshot_id, conversation_id, branch_id, episode_id, active_summary_id,
                context_ordinal, pack_version, summary_spine_ids, typed_memory_refs,
                open_question_refs, state_json, created_at
         FROM branch_snapshots
         WHERE snapshot_id = ?`,
      )
      .get(input.snapshotId) as unknown as BranchSnapshotRow | undefined;
    if (!row) {
      throw new Error(`Branch snapshot not found after insert: ${input.snapshotId}`);
    }
    return toBranchSnapshotRecord(row);
  }

  async getBranchSnapshot(snapshotId: string): Promise<BranchSnapshotRecord | null> {
    const row = this.db
      .prepare(
        `SELECT snapshot_id, conversation_id, branch_id, episode_id, active_summary_id,
                context_ordinal, pack_version, summary_spine_ids, typed_memory_refs,
                open_question_refs, state_json, created_at
         FROM branch_snapshots
         WHERE snapshot_id = ?`,
      )
      .get(snapshotId) as unknown as BranchSnapshotRow | undefined;
    return row ? toBranchSnapshotRecord(row) : null;
  }

  async getBranchSnapshots(conversationId: number): Promise<BranchSnapshotRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT snapshot_id, conversation_id, branch_id, episode_id, active_summary_id,
                context_ordinal, pack_version, summary_spine_ids, typed_memory_refs,
                open_question_refs, state_json, created_at
         FROM branch_snapshots
         WHERE conversation_id = ?
         ORDER BY created_at ASC`,
      )
      .all(conversationId) as unknown as BranchSnapshotRow[];
    return rows.map(toBranchSnapshotRecord);
  }

  async getLatestBranchSnapshot(conversationId: number): Promise<BranchSnapshotRecord | null> {
    const row = this.db
      .prepare(
        `SELECT snapshot_id, conversation_id, branch_id, episode_id, active_summary_id,
                context_ordinal, pack_version, summary_spine_ids, typed_memory_refs,
                open_question_refs, state_json, created_at
         FROM branch_snapshots
         WHERE conversation_id = ?
         ORDER BY created_at DESC, context_ordinal DESC
         LIMIT 1`,
      )
      .get(conversationId) as unknown as BranchSnapshotRow | undefined;
    return row ? toBranchSnapshotRecord(row) : null;
  }

  async getSummaryChildren(parentSummaryId: string): Promise<SummaryRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                s.file_ids, s.earliest_at, s.latest_at, s.descendant_count, s.created_at
                , s.descendant_token_count, s.source_message_token_count
       FROM summaries s
       JOIN summary_parents sp ON sp.summary_id = s.summary_id
       WHERE sp.parent_summary_id = ?
       ORDER BY sp.ordinal`,
      )
      .all(parentSummaryId) as unknown as SummaryRow[];
    return rows.map(toSummaryRecord);
  }

  async getSummaryParents(summaryId: string): Promise<SummaryRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                s.file_ids, s.earliest_at, s.latest_at, s.descendant_count, s.created_at
                , s.descendant_token_count, s.source_message_token_count
       FROM summaries s
       JOIN summary_parents sp ON sp.parent_summary_id = s.summary_id
       WHERE sp.summary_id = ?
       ORDER BY sp.ordinal`,
      )
      .all(summaryId) as unknown as SummaryRow[];
    return rows.map(toSummaryRecord);
  }

  async getSummarySubtree(summaryId: string): Promise<SummarySubtreeNodeRecord[]> {
    const rows = this.db
      .prepare(
        `WITH RECURSIVE subtree(summary_id, parent_summary_id, depth_from_root, path) AS (
           SELECT ?, NULL, 0, ''
           UNION ALL
           SELECT
             sp.summary_id,
             sp.parent_summary_id,
             subtree.depth_from_root + 1,
             CASE
               WHEN subtree.path = '' THEN printf('%04d', sp.ordinal)
               ELSE subtree.path || '.' || printf('%04d', sp.ordinal)
             END
           FROM summary_parents sp
           JOIN subtree ON sp.parent_summary_id = subtree.summary_id
         )
         SELECT
           s.summary_id,
           s.conversation_id,
           s.kind,
           COALESCE(sl.freshness_state, 'fresh') AS freshness_state,
           s.depth,
           s.content,
           s.token_count,
           s.file_ids,
           s.earliest_at,
           s.latest_at,
           s.descendant_count,
           s.descendant_token_count,
           s.source_message_token_count,
           s.created_at,
           subtree.depth_from_root,
           subtree.parent_summary_id,
           subtree.path,
           (
             SELECT COUNT(*) FROM summary_parents sp2
             WHERE sp2.parent_summary_id = s.summary_id
           ) AS child_count
         FROM subtree
         JOIN summaries s ON s.summary_id = subtree.summary_id
         LEFT JOIN summary_lineage sl ON sl.summary_id = s.summary_id
         ORDER BY subtree.depth_from_root ASC, subtree.path ASC, s.created_at ASC`,
      )
      .all(summaryId) as unknown as SummarySubtreeRow[];

    const seen = new Set<string>();
    const output: SummarySubtreeNodeRecord[] = [];
    for (const row of rows) {
      if (seen.has(row.summary_id)) {
        continue;
      }
      seen.add(row.summary_id);
      output.push({
        ...toSummaryRecord(row),
        depthFromRoot: Math.max(0, Math.floor(row.depth_from_root ?? 0)),
        parentSummaryId: row.parent_summary_id ?? null,
        path: typeof row.path === "string" ? row.path : "",
        freshnessState: normalizeSummaryFreshnessState(row.freshness_state),
        childCount:
          typeof row.child_count === "number" && Number.isFinite(row.child_count)
            ? Math.max(0, Math.floor(row.child_count))
            : 0,
      });
    }
    return output;
  }

  // ── Marbles ──────────────────────────────────────────────────────────────

  async insertMarble(input: CreateMarbleInput): Promise<MarbleRecord> {
    const sources = input.sources ?? [];
    const createdAt = (input.createdAt ?? new Date()).toISOString();
    const updatedAt = (input.updatedAt ?? input.createdAt ?? new Date()).toISOString();
    const invalidatedAt = input.invalidatedAt ? input.invalidatedAt.toISOString() : null;
    const sourceArtifactTokenCount = normalizeNonNegativeInteger(input.sourceArtifactTokenCount);

    this.db.exec("BEGIN");
    try {
      this.db
        .prepare(
          `INSERT INTO marbles (
            marble_id,
            conversation_id,
            marble_kind,
            compression_version,
            render_version,
            content,
            payload_json,
            token_count,
            confidence,
            freshness_state,
            source_fingerprint,
            content_hash,
            provenance_ref,
            source_count,
            source_artifact_token_count,
            derived_from_marble_id,
            invalidated_at,
            invalidation_reason,
            created_at,
            updated_at
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          input.marbleId,
          input.conversationId,
          input.marbleKind,
          input.compressionVersion,
          input.renderVersion,
          input.content,
          input.payloadJson,
          Math.max(0, Math.floor(input.tokenCount)),
          normalizeConfidence(input.confidence),
          input.freshnessState,
          input.sourceFingerprint,
          input.contentHash,
          input.provenanceRef,
          sources.length,
          sourceArtifactTokenCount,
          input.derivedFromMarbleId ?? null,
          invalidatedAt,
          input.invalidationReason ?? null,
          createdAt,
          updatedAt,
        );

      if (sources.length > 0) {
        const sourceStmt = this.db.prepare(
          `INSERT INTO marble_sources (
            marble_id,
            source_kind,
            source_id,
            source_sub_id,
            source_digest,
            source_provenance_ref,
            source_uri,
            ordinal
          )
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        );
        for (let idx = 0; idx < sources.length; idx++) {
          const source = sources[idx];
          sourceStmt.run(
            input.marbleId,
            source.sourceKind,
            source.sourceId,
            normalizeOptionalSubId(source.sourceSubId),
            source.sourceDigest,
            source.sourceProvenanceRef,
            source.sourceUri ?? null,
            normalizeNonNegativeInteger(source.ordinal ?? idx),
          );
        }
      }

      this.db.exec("COMMIT");
    } catch (err) {
      this.db.exec("ROLLBACK");
      throw err;
    }

    const row = this.db
      .prepare(
        `SELECT marble_id, conversation_id, marble_kind, compression_version, render_version,
                content, payload_json, token_count, confidence, freshness_state,
                source_fingerprint, content_hash, provenance_ref, source_count,
                source_artifact_token_count, derived_from_marble_id, invalidated_at,
                invalidation_reason, created_at, updated_at
         FROM marbles
         WHERE marble_id = ?`,
      )
      .get(input.marbleId) as unknown as MarbleRow | undefined;
    if (!row) {
      throw new Error(`Marble not found after insert: ${input.marbleId}`);
    }

    if (this.fts5Available) {
      try {
        this.db
          .prepare(`INSERT INTO marbles_fts(marble_id, content) VALUES (?, ?)`)
          .run(input.marbleId, input.content);
      } catch {
        // best-effort only
      }
    }

    return toMarbleRecord(row);
  }

  async getMarble(marbleId: string): Promise<MarbleRecord | null> {
    const row = this.db
      .prepare(
        `SELECT marble_id, conversation_id, marble_kind, compression_version, render_version,
                content, payload_json, token_count, confidence, freshness_state,
                source_fingerprint, content_hash, provenance_ref, source_count,
                source_artifact_token_count, derived_from_marble_id, invalidated_at,
                invalidation_reason, created_at, updated_at
         FROM marbles
         WHERE marble_id = ?`,
      )
      .get(marbleId) as unknown as MarbleRow | undefined;
    return row ? toMarbleRecord(row) : null;
  }

  async getMarblesByConversation(conversationId: number): Promise<MarbleRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT marble_id, conversation_id, marble_kind, compression_version, render_version,
                content, payload_json, token_count, confidence, freshness_state,
                source_fingerprint, content_hash, provenance_ref, source_count,
                source_artifact_token_count, derived_from_marble_id, invalidated_at,
                invalidation_reason, created_at, updated_at
         FROM marbles
         WHERE conversation_id = ?
         ORDER BY created_at`,
      )
      .all(conversationId) as unknown as MarbleRow[];
    return rows.map(toMarbleRecord);
  }

  async getMarbleSources(marbleId: string): Promise<MarbleSourceRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT marble_id, source_kind, source_id, source_sub_id, source_digest,
                source_provenance_ref, source_uri, ordinal
         FROM marble_sources
         WHERE marble_id = ?
         ORDER BY ordinal, source_kind, source_id, source_sub_id`,
      )
      .all(marbleId) as unknown as MarbleSourceRow[];
    return rows.map(toMarbleSourceRecord).map((row) => ({
      ...row,
      sourceSubId: row.sourceSubId === "" ? null : row.sourceSubId,
    }));
  }

  async invalidateMarble(input: {
    marbleId: string;
    freshnessState: Exclude<MarbleFreshnessState, "fresh">;
    reason: string;
    invalidatedAt?: Date;
  }): Promise<MarbleRecord | null> {
    const invalidatedAt = (input.invalidatedAt ?? new Date()).toISOString();
    this.db
      .prepare(
        `UPDATE marbles
         SET freshness_state = ?, invalidated_at = ?, invalidation_reason = ?, updated_at = ?
         WHERE marble_id = ?`,
      )
      .run(input.freshnessState, invalidatedAt, input.reason, invalidatedAt, input.marbleId);
    return this.getMarble(input.marbleId);
  }

  async searchMarbles(input: MarbleSearchInput): Promise<MarbleSearchResult[]> {
    const limit = input.limit ?? 50;

    if (input.mode === "full_text") {
      if (this.fts5Available) {
        try {
          return this.searchMarbleFullText(input.query, limit, input.conversationId, input.since, input.before);
        } catch {
          return this.searchMarbleLike(input.query, limit, input.conversationId, input.since, input.before);
        }
      }
      return this.searchMarbleLike(input.query, limit, input.conversationId, input.since, input.before);
    }

    return this.searchMarbleRegex(input.query, limit, input.conversationId, input.since, input.before);
  }

  private searchMarbleFullText(
    query: string,
    limit: number,
    conversationId?: number,
    since?: Date,
    before?: Date,
  ): MarbleSearchResult[] {
    const where: string[] = ["marbles_fts MATCH ?"];
    const args: Array<string | number> = [sanitizeFts5Query(query)];
    if (conversationId != null) {
      where.push("m.conversation_id = ?");
      args.push(conversationId);
    }
    if (since) {
      where.push("julianday(m.created_at) >= julianday(?)");
      args.push(since.toISOString());
    }
    if (before) {
      where.push("julianday(m.created_at) < julianday(?)");
      args.push(before.toISOString());
    }
    args.push(limit);

    const sql = `SELECT
         marbles_fts.marble_id,
         m.conversation_id,
         m.marble_kind,
         m.freshness_state,
         m.provenance_ref,
         m.source_fingerprint,
         m.source_count,
         m.source_artifact_token_count,
         snippet(marbles_fts, 1, '', '', '...', 32) AS snippet,
         rank,
         m.created_at
       FROM marbles_fts
       JOIN marbles m ON m.marble_id = marbles_fts.marble_id
       WHERE ${where.join(" AND ")}
       ORDER BY m.created_at DESC
       LIMIT ?`;
    const rows = this.db.prepare(sql).all(...args) as unknown as MarbleSearchRow[];
    return rows.map(toMarbleSearchResult);
  }

  private searchMarbleLike(
    query: string,
    limit: number,
    conversationId?: number,
    since?: Date,
    before?: Date,
  ): MarbleSearchResult[] {
    const plan = buildLikeSearchPlan("content", query);
    if (plan.terms.length === 0) {
      return [];
    }

    const where: string[] = [...plan.where];
    const args: Array<string | number> = [...plan.args];
    if (conversationId != null) {
      where.push("conversation_id = ?");
      args.push(conversationId);
    }
    if (since) {
      where.push("julianday(created_at) >= julianday(?)");
      args.push(since.toISOString());
    }
    if (before) {
      where.push("julianday(created_at) < julianday(?)");
      args.push(before.toISOString());
    }
    args.push(limit);

    const whereClause = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";
    const rows = this.db
      .prepare(
        `SELECT marble_id, conversation_id, marble_kind, freshness_state,
                provenance_ref, source_fingerprint, source_count, source_artifact_token_count,
                content, created_at
         FROM marbles
         ${whereClause}
         ORDER BY created_at DESC
         LIMIT ?`,
      )
      .all(...args) as unknown as Array<{
      marble_id: string;
      conversation_id: number;
      marble_kind: MarbleKind;
      freshness_state: MarbleFreshnessState;
      provenance_ref: string;
      source_fingerprint: string;
      source_count: number;
      source_artifact_token_count: number;
      content: string;
      created_at: string;
    }>;

    return rows.map((row) => ({
      marbleId: row.marble_id,
      conversationId: row.conversation_id,
      marbleKind: row.marble_kind,
      freshnessState: row.freshness_state,
      provenanceRef: row.provenance_ref,
      sourceFingerprint: row.source_fingerprint,
      sourceCount: row.source_count,
      sourceArtifactTokenCount: row.source_artifact_token_count,
      snippet: createFallbackSnippet(row.content, plan.terms),
      createdAt: new Date(row.created_at),
      rank: 0,
    }));
  }

  private searchMarbleRegex(
    pattern: string,
    limit: number,
    conversationId?: number,
    since?: Date,
    before?: Date,
  ): MarbleSearchResult[] {
    const re = new RegExp(pattern);

    const where: string[] = [];
    const args: Array<string | number> = [];
    if (conversationId != null) {
      where.push("conversation_id = ?");
      args.push(conversationId);
    }
    if (since) {
      where.push("julianday(created_at) >= julianday(?)");
      args.push(since.toISOString());
    }
    if (before) {
      where.push("julianday(created_at) < julianday(?)");
      args.push(before.toISOString());
    }
    const whereClause = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";
    const rows = this.db
      .prepare(
        `SELECT marble_id, conversation_id, marble_kind, freshness_state,
                provenance_ref, source_fingerprint, source_count, source_artifact_token_count,
                content, created_at
         FROM marbles
         ${whereClause}
         ORDER BY created_at DESC`,
      )
      .all(...args) as unknown as Array<{
      marble_id: string;
      conversation_id: number;
      marble_kind: MarbleKind;
      freshness_state: MarbleFreshnessState;
      provenance_ref: string;
      source_fingerprint: string;
      source_count: number;
      source_artifact_token_count: number;
      content: string;
      created_at: string;
    }>;

    const results: MarbleSearchResult[] = [];
    for (const row of rows) {
      if (results.length >= limit) {
        break;
      }
      const match = re.exec(row.content);
      if (match) {
        results.push({
          marbleId: row.marble_id,
          conversationId: row.conversation_id,
          marbleKind: row.marble_kind,
          freshnessState: row.freshness_state,
          provenanceRef: row.provenance_ref,
          sourceFingerprint: row.source_fingerprint,
          sourceCount: row.source_count,
          sourceArtifactTokenCount: row.source_artifact_token_count,
          snippet: match[0],
          createdAt: new Date(row.created_at),
          rank: 0,
        });
      }
    }
    return results;
  }

  // ── Context items ─────────────────────────────────────────────────────────

  async getContextItems(conversationId: number): Promise<ContextItemRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT conversation_id, ordinal, item_type, message_id, summary_id, created_at
       FROM context_items
       WHERE conversation_id = ?
       ORDER BY ordinal`,
      )
      .all(conversationId) as unknown as ContextItemRow[];
    return rows.map(toContextItemRecord);
  }

  async getRecentContextSummaries(conversationId: number, limit = 2): Promise<SummaryRecord[]> {
    const safeLimit = Number.isFinite(limit) ? Math.max(1, Math.floor(limit)) : 2;
    const rows = this.db
      .prepare(
        `SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                s.file_ids, s.earliest_at, s.latest_at, s.descendant_count, s.created_at,
                s.descendant_token_count, s.source_message_token_count
       FROM context_items ci
       JOIN summaries s ON s.summary_id = ci.summary_id
       WHERE ci.conversation_id = ? AND ci.item_type = 'summary'
       ORDER BY ci.ordinal DESC
       LIMIT ?`,
      )
      .all(conversationId, safeLimit) as unknown as SummaryRow[];
    return rows.map(toSummaryRecord).reverse();
  }

  async getDistinctDepthsInContext(
    conversationId: number,
    options?: { maxOrdinalExclusive?: number },
  ): Promise<number[]> {
    const maxOrdinalExclusive = options?.maxOrdinalExclusive;
    const useOrdinalBound =
      typeof maxOrdinalExclusive === "number" &&
      Number.isFinite(maxOrdinalExclusive) &&
      maxOrdinalExclusive !== Infinity;

    const sql = useOrdinalBound
      ? `SELECT DISTINCT s.depth
         FROM context_items ci
         JOIN summaries s ON s.summary_id = ci.summary_id
         WHERE ci.conversation_id = ?
           AND ci.item_type = 'summary'
           AND ci.ordinal < ?
         ORDER BY s.depth ASC`
      : `SELECT DISTINCT s.depth
         FROM context_items ci
         JOIN summaries s ON s.summary_id = ci.summary_id
         WHERE ci.conversation_id = ?
           AND ci.item_type = 'summary'
         ORDER BY s.depth ASC`;

    const rows = useOrdinalBound
      ? (this.db
          .prepare(sql)
          .all(conversationId, Math.floor(maxOrdinalExclusive)) as unknown as DistinctDepthRow[])
      : (this.db.prepare(sql).all(conversationId) as unknown as DistinctDepthRow[]);

    return rows.map((row) => row.depth);
  }

  async appendContextMessage(conversationId: number, messageId: number): Promise<void> {
    const row = this.db
      .prepare(
        `SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal
       FROM context_items WHERE conversation_id = ?`,
      )
      .get(conversationId) as unknown as MaxOrdinalRow;

    this.db
      .prepare(
        `INSERT INTO context_items (conversation_id, ordinal, item_type, message_id)
       VALUES (?, ?, 'message', ?)`,
      )
      .run(conversationId, row.max_ordinal + 1, messageId);
  }

  async appendContextMessages(conversationId: number, messageIds: number[]): Promise<void> {
    if (messageIds.length === 0) {
      return;
    }

    const row = this.db
      .prepare(
        `SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal
       FROM context_items WHERE conversation_id = ?`,
      )
      .get(conversationId) as unknown as MaxOrdinalRow;
    const baseOrdinal = row.max_ordinal + 1;

    const stmt = this.db.prepare(
      `INSERT INTO context_items (conversation_id, ordinal, item_type, message_id)
       VALUES (?, ?, 'message', ?)`,
    );
    for (let idx = 0; idx < messageIds.length; idx++) {
      stmt.run(conversationId, baseOrdinal + idx, messageIds[idx]);
    }
  }

  async appendContextSummary(conversationId: number, summaryId: string): Promise<void> {
    const row = this.db
      .prepare(
        `SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal
       FROM context_items WHERE conversation_id = ?`,
      )
      .get(conversationId) as unknown as MaxOrdinalRow;

    this.db
      .prepare(
        `INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id)
       VALUES (?, ?, 'summary', ?)`,
      )
      .run(conversationId, row.max_ordinal + 1, summaryId);
  }

  async replaceContextRangeWithSummary(input: {
    conversationId: number;
    startOrdinal: number;
    endOrdinal: number;
    summaryId: string;
  }): Promise<void> {
    const { conversationId, startOrdinal, endOrdinal, summaryId } = input;

    this.db.exec("BEGIN");
    try {
      // 1. Delete context items in the range [startOrdinal, endOrdinal]
      this.db
        .prepare(
          `DELETE FROM context_items
         WHERE conversation_id = ?
           AND ordinal >= ?
           AND ordinal <= ?`,
        )
        .run(conversationId, startOrdinal, endOrdinal);

      // 2. Insert the replacement summary item at startOrdinal
      this.db
        .prepare(
          `INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id)
         VALUES (?, ?, 'summary', ?)`,
        )
        .run(conversationId, startOrdinal, summaryId);

      // 3. Resequence all ordinals to maintain contiguity (no gaps).
      //    Fetch current items, then update ordinals in order.
      const items = this.db
        .prepare(
          `SELECT ordinal FROM context_items
         WHERE conversation_id = ?
         ORDER BY ordinal`,
        )
        .all(conversationId) as unknown as { ordinal: number }[];

      const updateStmt = this.db.prepare(
        `UPDATE context_items
         SET ordinal = ?
         WHERE conversation_id = ? AND ordinal = ?`,
      );

      // Use negative temp ordinals first to avoid unique constraint conflicts
      for (let i = 0; i < items.length; i++) {
        updateStmt.run(-(i + 1), conversationId, items[i].ordinal);
      }
      for (let i = 0; i < items.length; i++) {
        updateStmt.run(i, conversationId, -(i + 1));
      }

      this.db.exec("COMMIT");
    } catch (err) {
      this.db.exec("ROLLBACK");
      throw err;
    }
  }

  async getContextTokenCount(conversationId: number): Promise<number> {
    const row = this.db
      .prepare(
        `SELECT COALESCE(SUM(token_count), 0) AS total
       FROM (
         SELECT m.token_count
         FROM context_items ci
         JOIN messages m ON m.message_id = ci.message_id
         WHERE ci.conversation_id = ?
           AND ci.item_type = 'message'

         UNION ALL

         SELECT s.token_count
         FROM context_items ci
         JOIN summaries s ON s.summary_id = ci.summary_id
         WHERE ci.conversation_id = ?
           AND ci.item_type = 'summary'
       ) sub`,
      )
      .get(conversationId, conversationId) as unknown as TokenSumRow;
    return row?.total ?? 0;
  }

  // ── Search ────────────────────────────────────────────────────────────────

  async searchSummaries(input: SummarySearchInput): Promise<SummarySearchResult[]> {
    const limit = input.limit ?? 50;

    if (input.mode === "full_text") {
      if (this.fts5Available) {
        try {
          return this.searchFullText(
            input.query,
            limit,
            input.conversationId,
            input.since,
            input.before,
          );
        } catch {
          return this.searchLike(
            input.query,
            limit,
            input.conversationId,
            input.since,
            input.before,
          );
        }
      }
      return this.searchLike(input.query, limit, input.conversationId, input.since, input.before);
    }
    return this.searchRegex(input.query, limit, input.conversationId, input.since, input.before);
  }

  private searchFullText(
    query: string,
    limit: number,
    conversationId?: number,
    since?: Date,
    before?: Date,
  ): SummarySearchResult[] {
    const where: string[] = ["summaries_fts MATCH ?"];
    const args: Array<string | number> = [sanitizeFts5Query(query)];
    if (conversationId != null) {
      where.push("s.conversation_id = ?");
      args.push(conversationId);
    }
    if (since) {
      where.push("julianday(s.created_at) >= julianday(?)");
      args.push(since.toISOString());
    }
    if (before) {
      where.push("julianday(s.created_at) < julianday(?)");
      args.push(before.toISOString());
    }
    args.push(limit);

    const sql = `SELECT
         summaries_fts.summary_id,
         s.conversation_id,
         s.kind,
         COALESCE(sl.freshness_state, 'fresh') AS freshness_state,
         snippet(summaries_fts, 1, '', '', '...', 32) AS snippet,
         rank,
         s.created_at
       FROM summaries_fts
       JOIN summaries s ON s.summary_id = summaries_fts.summary_id
       LEFT JOIN summary_lineage sl ON sl.summary_id = s.summary_id
       WHERE ${where.join(" AND ")}
       ORDER BY s.created_at DESC
       LIMIT ?`;
    const rows = this.db.prepare(sql).all(...args) as unknown as SummarySearchRow[];
    return rows.map(toSearchResult);
  }

  private searchLike(
    query: string,
    limit: number,
    conversationId?: number,
    since?: Date,
    before?: Date,
  ): SummarySearchResult[] {
    const plan = buildLikeSearchPlan("content", query);
    if (plan.terms.length === 0) {
      return [];
    }

    const where: string[] = [...plan.where];
    const args: Array<string | number> = [...plan.args];
    if (conversationId != null) {
      where.push("s.conversation_id = ?");
      args.push(conversationId);
    }
    if (since) {
      where.push("julianday(s.created_at) >= julianday(?)");
      args.push(since.toISOString());
    }
    if (before) {
      where.push("julianday(s.created_at) < julianday(?)");
      args.push(before.toISOString());
    }
    args.push(limit);

    const whereClause = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";
    const rows = this.db
      .prepare(
        `SELECT s.summary_id, s.conversation_id, s.kind,
                COALESCE(sl.freshness_state, 'fresh') AS freshness_state,
                s.content, s.created_at
         FROM summaries s
         LEFT JOIN summary_lineage sl ON sl.summary_id = s.summary_id
         ${whereClause}
         ORDER BY s.created_at DESC
         LIMIT ?`,
      )
      .all(...args) as unknown as Array<{
      summary_id: string;
      conversation_id: number;
      kind: SummaryKind;
      freshness_state: SummaryFreshnessState | null;
      content: string;
      created_at: string;
    }>;

    return rows.map((row) => ({
      summaryId: row.summary_id,
      conversationId: row.conversation_id,
      kind: row.kind,
      freshnessState: normalizeSummaryFreshnessState(row.freshness_state),
      snippet: createFallbackSnippet(row.content, plan.terms),
      createdAt: new Date(row.created_at),
      rank: 0,
    }));
  }

  private searchRegex(
    pattern: string,
    limit: number,
    conversationId?: number,
    since?: Date,
    before?: Date,
  ): SummarySearchResult[] {
    const re = new RegExp(pattern);

    const where: string[] = [];
    const args: Array<string | number> = [];
    if (conversationId != null) {
      where.push("s.conversation_id = ?");
      args.push(conversationId);
    }
    if (since) {
      where.push("julianday(s.created_at) >= julianday(?)");
      args.push(since.toISOString());
    }
    if (before) {
      where.push("julianday(s.created_at) < julianday(?)");
      args.push(before.toISOString());
    }
    const whereClause = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";
    const rows = this.db
      .prepare(
        `SELECT s.summary_id, s.conversation_id, s.kind,
                COALESCE(sl.freshness_state, 'fresh') AS freshness_state,
                s.content, s.created_at
         FROM summaries s
         LEFT JOIN summary_lineage sl ON sl.summary_id = s.summary_id
         ${whereClause}
         ORDER BY s.created_at DESC`,
      )
      .all(...args) as unknown as Array<{
      summary_id: string;
      conversation_id: number;
      kind: SummaryKind;
      freshness_state: SummaryFreshnessState | null;
      content: string;
      created_at: string;
    }>;

    const results: SummarySearchResult[] = [];
    for (const row of rows) {
      if (results.length >= limit) {
        break;
      }
      const match = re.exec(row.content);
      if (match) {
        results.push({
          summaryId: row.summary_id,
          conversationId: row.conversation_id,
          kind: row.kind,
          freshnessState: normalizeSummaryFreshnessState(row.freshness_state),
          snippet: match[0],
          createdAt: new Date(row.created_at),
          rank: 0,
        });
      }
    }
    return results;
  }

  // ── Large files ───────────────────────────────────────────────────────────

  async insertLargeFile(input: CreateLargeFileInput): Promise<LargeFileRecord> {
    this.db
      .prepare(
        `INSERT INTO large_files (file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        input.fileId,
        input.conversationId,
        input.fileName ?? null,
        input.mimeType ?? null,
        input.byteSize ?? null,
        input.storageUri,
        input.explorationSummary ?? null,
      );

    const row = this.db
      .prepare(
        `SELECT file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary, created_at
       FROM large_files WHERE file_id = ?`,
      )
      .get(input.fileId) as unknown as LargeFileRow;

    return toLargeFileRecord(row);
  }

  async getLargeFile(fileId: string): Promise<LargeFileRecord | null> {
    const row = this.db
      .prepare(
        `SELECT file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary, created_at
       FROM large_files WHERE file_id = ?`,
      )
      .get(fileId) as unknown as LargeFileRow | undefined;
    return row ? toLargeFileRecord(row) : null;
  }

  async getLargeFilesByConversation(conversationId: number): Promise<LargeFileRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary, created_at
       FROM large_files
       WHERE conversation_id = ?
       ORDER BY created_at`,
      )
      .all(conversationId) as unknown as LargeFileRow[];
    return rows.map(toLargeFileRecord);
  }
}
