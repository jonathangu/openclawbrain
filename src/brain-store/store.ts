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
  BrainEvidence,
  BrainEvidenceKind,
  BrainEvidenceResolution,
  ResolvedLabel,
  SeedWeight,
  LearningJournalEventType,
  LearningJournalRecord,
  MutationProposedJournalPayload,
  BundleEvaluationStartedJournalPayload,
  BundleEvaluationCompletedJournalPayload,
  PromotionJournalPayload,
} from "../brain-core/types.js";

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
      DELETE FROM brain_seed_weights;
      DELETE FROM brain_edges;
      DELETE FROM brain_nodes;
    `);
  }

  deleteNode(id: string): void {
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
      SELECT * FROM brain_episodes WHERE reward IS NOT NULL AND updated = 0 ORDER BY created_at ASC LIMIT ?
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
      metadata: JSON.parse((row.metadata as string) || "{}"),
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
      INSERT INTO brain_traces (id, episode_id, pack_version, query_text, seed_scores, trajectory, fired_nodes, vetoed_nodes, context_chars, footer, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      trace.id, trace.episodeId, trace.packVersion, trace.queryText,
      JSON.stringify(trace.seedScores), JSON.stringify(trace.trajectory),
      JSON.stringify(trace.firedNodes), JSON.stringify(trace.vetoedNodes),
      trace.contextChars, trace.footer, trace.createdAt,
    );
  }

  getRecentTraces(limit: number): DecisionTrace[] {
    const rows = this.db.prepare(`SELECT * FROM brain_traces ORDER BY created_at DESC LIMIT ?`).all(limit) as Record<string, unknown>[];
    return rows.map((r) => this.toTrace(r));
  }

  getTrace(id: string): DecisionTrace | null {
    const row = this.db.prepare(`SELECT * FROM brain_traces WHERE id = ?`).get(id) as Record<string, unknown> | undefined;
    return row ? this.toTrace(row) : null;
  }

  private toTrace(row: Record<string, unknown>): DecisionTrace {
    return {
      id: row.id as string,
      episodeId: (row.episode_id as string) ?? null,
      packVersion: (row.pack_version as number) ?? null,
      queryText: (row.query_text as string) || "",
      seedScores: JSON.parse((row.seed_scores as string) || "[]"),
      trajectory: JSON.parse((row.trajectory as string) || "[]"),
      firedNodes: JSON.parse((row.fired_nodes as string) || "[]"),
      vetoedNodes: JSON.parse((row.vetoed_nodes as string) || "[]"),
      contextChars: (row.context_chars as number) || 0,
      footer: (row.footer as string) || "",
      createdAt: row.created_at as number,
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

    return packDir;
  }

  readPackSnapshot(version: number): {
    nodes: BrainNode[];
    edges: BrainEdge[];
    seedWeights: SeedWeight[];
    metadata: Record<string, unknown>;
  } | null {
    if (!this.options.brainRoot) {
      return null;
    }

    const packDir = this.getPackDir(version);
    const nodesFile = join(packDir, "nodes.jsonl");
    const edgesFile = join(packDir, "edges.jsonl");
    const seedWeightsFile = join(packDir, "seed-weights.jsonl");
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
    const metadata = existsSync(metadataFile)
      ? JSON.parse(readFileSync(metadataFile, "utf8")) as Record<string, unknown>
      : {};

    return { nodes, edges, seedWeights, metadata };
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
