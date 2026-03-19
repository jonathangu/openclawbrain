import type { BrainStore } from "../brain-store/store.js";
import type { MutationProposal, MutationStatus, Pack } from "../brain-core/types.js";

export interface PromotionStoryPackSummary {
  version: number;
  createdAt: number;
  promotedAt: number | null;
  rolledBack: boolean;
  nodeCount: number;
  edgeCount: number;
  reason: string | null;
  metadata: Record<string, unknown>;
  health: Record<string, unknown> | null;
}

export interface PromotionStoryCandidateSummary {
  id: string;
  kind: MutationProposal["kind"];
  status: MutationStatus;
  createdAt: number;
  resolvedAt: number | null;
  expectedGain: number | null;
  summary: string;
  proposal: unknown;
  evidence: unknown | null;
}

export interface PromotionStoryLatestActivity {
  type: "pack_promoted" | "candidate_promoted" | "candidate_rejected" | "candidate_pending" | "idle";
  at: number | null;
  summary: string | null;
  packVersion: number | null;
  candidateId: string | null;
}

export interface PromotionStory {
  summary: {
    currentPackVersion: number | null;
    mutationBacklog: Record<MutationStatus, number>;
    lastPromotionReason: string | null;
    lastReplayFailureReason: string | null;
  };
  currentPack: PromotionStoryPackSummary | null;
  recentPromotions: PromotionStoryPackSummary[];
  candidates: {
    pending: PromotionStoryCandidateSummary[];
    promoted: PromotionStoryCandidateSummary[];
    rejected: PromotionStoryCandidateSummary[];
  };
  latestActivity: PromotionStoryLatestActivity;
  integrations: {
    structuredVerdict: null;
    learningJournal: null;
  };
}

function parseJsonObject(value: string | null | undefined): Record<string, unknown> | null {
  if (!value) {
    return null;
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function summarizeMutation(proposal: MutationProposal): string {
  const details = proposal.proposal as Record<string, unknown>;
  switch (proposal.kind) {
    case "connect":
      return `connect ${String(details.nodeA ?? "?")} -> ${String(details.nodeB ?? "?")} (${String(details.coFireCount ?? 0)} co-fires)`;
    case "prune":
      return `prune ${String(details.edgeKind ?? "edge")} ${String(details.source ?? "?")} -> ${String(details.target ?? "?")}`;
    case "inject": {
      const content = String(details.content ?? "").trim();
      const preview = content.length > 60 ? `${content.slice(0, 57)}...` : content;
      const firedCount = Array.isArray(details.firedNodes) ? details.firedNodes.length : 0;
      return `inject ${String(details.nodeKind ?? "node")} "${preview}" (${firedCount} fired nodes)`;
    }
    case "split":
    case "merge":
      return `${proposal.kind} candidate ${proposal.id}`;
    default:
      return proposal.id;
  }
}

function toCandidateSummary(proposal: MutationProposal): PromotionStoryCandidateSummary {
  return {
    id: proposal.id,
    kind: proposal.kind,
    status: proposal.status,
    createdAt: proposal.createdAt,
    resolvedAt: proposal.resolvedAt,
    expectedGain: proposal.expectedGain,
    summary: summarizeMutation(proposal),
    proposal: proposal.proposal,
    evidence: proposal.evidence,
  };
}

function toPackSummary(store: BrainStore, pack: Pack): PromotionStoryPackSummary {
  const snapshot = store.readPackSnapshot(pack.version);
  const metadata = snapshot?.metadata ?? {};
  const reason = typeof metadata.reason === "string" ? metadata.reason : null;
  return {
    version: pack.version,
    createdAt: pack.createdAt,
    promotedAt: pack.promotedAt,
    rolledBack: pack.rolledBack,
    nodeCount: pack.nodeCount,
    edgeCount: pack.edgeCount,
    reason,
    metadata,
    health: parseJsonObject(pack.healthJson),
  };
}

function buildLatestActivity(params: {
  recentPromotions: PromotionStoryPackSummary[];
  pending: PromotionStoryCandidateSummary[];
  promoted: PromotionStoryCandidateSummary[];
  rejected: PromotionStoryCandidateSummary[];
}): PromotionStoryLatestActivity {
  const activities: PromotionStoryLatestActivity[] = [];

  if (params.recentPromotions[0]?.promotedAt) {
    activities.push({
      type: "pack_promoted",
      at: params.recentPromotions[0].promotedAt,
      summary: params.recentPromotions[0].reason ?? `pack v${params.recentPromotions[0].version} promoted`,
      packVersion: params.recentPromotions[0].version,
      candidateId: null,
    });
  }
  if (params.pending[0]) {
    activities.push({
      type: "candidate_pending",
      at: params.pending[0].createdAt,
      summary: params.pending[0].summary,
      packVersion: null,
      candidateId: params.pending[0].id,
    });
  }
  if (params.promoted[0]?.resolvedAt) {
    activities.push({
      type: "candidate_promoted",
      at: params.promoted[0].resolvedAt,
      summary: params.promoted[0].summary,
      packVersion: null,
      candidateId: params.promoted[0].id,
    });
  }
  if (params.rejected[0]?.resolvedAt) {
    activities.push({
      type: "candidate_rejected",
      at: params.rejected[0].resolvedAt,
      summary: params.rejected[0].summary,
      packVersion: null,
      candidateId: params.rejected[0].id,
    });
  }

  if (activities.length === 0) {
    return {
      type: "idle",
      at: null,
      summary: null,
      packVersion: null,
      candidateId: null,
    };
  }

  activities.sort((a, b) => (b.at ?? 0) - (a.at ?? 0));
  return activities[0];
}

export function buildWorkerPromotionSnapshotMetadata(
  store: BrainStore,
  extraMetadata: Record<string, unknown> = {},
): Record<string, unknown> {
  const lastPromotionReason = store.getTrainingState("last_promotion_reason");
  const lastReplayFailureReason = store.getTrainingState("last_replay_failure_reason");

  return {
    ...extraMetadata,
    ...(lastPromotionReason ? { lastPromotionReason } : {}),
    ...(lastReplayFailureReason ? { lastReplayFailureReason } : {}),
    mutationBacklog: store.countMutationsByStatus(),
  };
}

export function buildPromotionStory(
  store: BrainStore,
  options: { recentPromotionLimit?: number; recentCandidateLimit?: number } = {},
): PromotionStory {
  const recentPromotionLimit = options.recentPromotionLimit ?? 5;
  const recentCandidateLimit = options.recentCandidateLimit ?? 5;
  const recentPromotions = store.getRecentPromotedPacks(recentPromotionLimit).map((pack) => toPackSummary(store, pack));
  const currentPackRecord = store.getCurrentPack();
  const currentPackVersion = store.getCurrentPackVersion() ?? currentPackRecord?.version ?? null;
  const currentPack =
    recentPromotions.find((pack) => pack.version === currentPackVersion)
    ?? (currentPackRecord && currentPackRecord.version === currentPackVersion ? toPackSummary(store, currentPackRecord) : null);
  const pending = store.getRecentMutationsByStatus("pending", recentCandidateLimit).map(toCandidateSummary);
  const promoted = store.getRecentMutationsByStatus("promoted", recentCandidateLimit).map(toCandidateSummary);
  const rejected = store.getRecentMutationsByStatus("rejected", recentCandidateLimit).map(toCandidateSummary);

  return {
    summary: {
      currentPackVersion,
      mutationBacklog: store.countMutationsByStatus(),
      lastPromotionReason: store.getTrainingState("last_promotion_reason"),
      lastReplayFailureReason: store.getTrainingState("last_replay_failure_reason"),
    },
    currentPack,
    recentPromotions,
    candidates: {
      pending,
      promoted,
      rejected,
    },
    latestActivity: buildLatestActivity({
      recentPromotions,
      pending,
      promoted,
      rejected,
    }),
    integrations: {
      structuredVerdict: null,
      learningJournal: null,
    },
  };
}
