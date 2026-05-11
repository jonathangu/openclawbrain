import type {
  GraphMaintenanceProposal,
  GraphMaintenanceRun,
  MemoryAuthorityEvent,
  MemoryEdge,
  MemoryNode,
  MemoryValidity,
} from './memory-types.js';
import { MemoryStore, now } from './memory-store.js';
import { isAgentAllowed } from './config.js';
import { defaultValidityForMemory } from './memory-authority.js';
import { hashText, redactJsonValue, redactText, safeString, shortHash } from './redact.js';

export interface GraphMaintenanceEngineOptions {
  store: MemoryStore;
  config?: any;
}

export interface GraphMaintenanceHealth {
  ok: boolean;
  agentId: string;
  generatedAt: string;
  counts: {
    nodes: number;
    activeNodes: number;
    supersededNodes: number;
    deletedNodes: number;
    tombstonedNodes: number;
    edges: number;
    badEdges: number;
    exactDuplicateClusters: number;
    staleHighAuthorityNodes: number;
    tombstoneRecaptureCandidates: number;
    scopedExceptionCandidates: number;
  };
  invariantSummary: {
    authorityBoundary: string;
    connectivityBoundary: string;
    tombstoneBoundary: string;
    feedbackBoundary: string;
  };
  topIssues: Array<{ kind: string; count: number; severity: 'low' | 'medium' | 'high'; nextAction: string }>;
}

export interface GraphMaintenanceDryRunReport {
  ok: boolean;
  agentId: string;
  run: GraphMaintenanceRun;
  health: GraphMaintenanceHealth;
  proposals: GraphMaintenanceProposal[];
}

type LoadedGraph = {
  nodes: MemoryNode[];
  activeNodes: MemoryNode[];
  validity: Map<string, MemoryValidity>;
  edges: MemoryEdge[];
  byId: Map<string, MemoryNode>;
};

export class GraphMaintenanceEngine {
  private store: MemoryStore;
  private config: any;

  constructor(options: GraphMaintenanceEngineOptions) {
    this.store = options.store;
    this.config = options.config || {};
  }

  health(agentId: string, limit = 1000): GraphMaintenanceHealth {
    const graph = this.loadGraph(agentId, limit);
    const duplicateClusters = exactDuplicateClusters(graph).filter((cluster) => cluster.length > 1);
    const badEdges = badEdgesForGraph(graph);
    const stale = staleHighAuthorityNodes(graph);
    const tombstoneCandidates = tombstoneRecaptureCandidates(graph);
    const scoped = scopedExceptionCandidates(this.store.listMemoryAuthorityEvents(agentId, 500));
    const tombstonedNodes = graph.nodes.filter((node) => graph.validity.get(node.id)?.retentionState === 'tombstoned').length;
    const deletedNodes = graph.nodes.filter((node) => node.deletedAt || ['soft_deleted', 'hard_deleted'].includes(graph.validity.get(node.id)?.retentionState || '')).length;
    const topIssues = [
      issue('bad_edges', badEdges.length, 'low', 'Run /brain graph dry-run and apply low-risk edge retirement proposals.'),
      issue('exact_duplicate_clusters', duplicateClusters.length, 'low', 'Run /brain graph dry-run and apply exact duplicate merge proposals after review.'),
      issue('stale_high_authority_nodes', stale.length, 'medium', 'Review stale authority proposals before changing validity features.'),
      issue('tombstone_recapture_candidates', tombstoneCandidates.length, 'high', 'Review immediately; do not expose raw tombstoned content.'),
      issue('scoped_exception_candidates', scoped.length, 'medium', 'Review scoped exception proposals instead of broad preference rewrites.'),
    ].filter((item) => item.count > 0);

    return {
      ok: true,
      agentId,
      generatedAt: now(),
      counts: {
        nodes: graph.nodes.length,
        activeNodes: graph.activeNodes.length,
        supersededNodes: graph.nodes.filter((node) => node.supersededBy).length,
        deletedNodes,
        tombstonedNodes,
        edges: graph.edges.length,
        badEdges: badEdges.length,
        exactDuplicateClusters: duplicateClusters.length,
        staleHighAuthorityNodes: stale.length,
        tombstoneRecaptureCandidates: tombstoneCandidates.length,
        scopedExceptionCandidates: scoped.length,
      },
      invariantSummary: {
        authorityBoundary: 'Graph Maintenance can provide features; Memory Authority recomputes turn-level use.',
        connectivityBoundary: 'Connectivity is never evidence of authority.',
        tombstoneBoundary: 'Tombstoned or hard-deleted content cannot be revived or leaked into proposals/proof.',
        feedbackBoundary: 'Implicit route success can update behavioral hints only, not truth confidence.',
      },
      topIssues,
    };
  }

  dryRun(agentId: string, options: { limit?: number } = {}): GraphMaintenanceDryRunReport {
    const limit = options.limit ?? 1000;
    const graph = this.loadGraph(agentId, limit);
    const health = this.health(agentId, limit);
    const run = this.store.insertGraphMaintenanceRun({
      agentId,
      mode: 'dry_run',
      status: 'running',
      nodesScanned: graph.nodes.length,
      edgesScanned: graph.edges.length,
      proposalsCreated: 0,
      proposalsApplied: 0,
      proposalsRejected: 0,
      riskSummary: {},
      metrics: health.counts,
    });
    const proposals: GraphMaintenanceProposal[] = [];
    for (const proposal of [
      ...this.duplicateMergeProposals(run.id, graph),
      ...this.badEdgeProposals(run.id, graph),
      ...this.staleAuthorityProposals(run.id, graph),
      ...this.tombstoneBlockProposals(run.id, graph),
      ...this.scopedExceptionProposals(run.id, agentId),
      ...this.feedbackObservationProposals(run.id, agentId),
    ]) {
      proposals.push(this.store.insertGraphMaintenanceProposal(proposal));
    }
    const riskSummary = riskSummaryFor(proposals);
    const finished = this.store.finishGraphMaintenanceRun(run.id, {
      status: 'completed',
      nodesScanned: graph.nodes.length,
      edgesScanned: graph.edges.length,
      proposalsCreated: proposals.length,
      riskSummary,
      metrics: health.counts,
    }) || run;
    this.store.insertProofEvent({
      agentId,
      kind: 'graph_maintenance_dry_run',
      rawTranscriptStored: false,
      payload: redactJsonValue({
        runId: run.id,
        proposals: proposals.length,
        riskSummary,
        counts: health.counts,
      }),
    });
    return { ok: true, agentId, run: finished, health, proposals };
  }

  applyProposal(agentId: string, proposalId: string): { ok: boolean; proposal?: GraphMaintenanceProposal | null; reason?: string } {
    const proposal = this.store.getGraphMaintenanceProposal(proposalId);
    if (!proposal || proposal.agentId !== agentId) return { ok: false, reason: 'proposal_not_found' };
    if (proposal.status === 'applied') return { ok: true, proposal, reason: 'already_applied' };
    if (proposal.status === 'rejected') return { ok: false, proposal, reason: 'proposal_rejected' };
    if (!isSafeApplyProposal(proposal)) {
      return { ok: false, proposal, reason: 'review_required_or_not_safe_auto' };
    }
    try {
      const applied = this.store.transaction(() => this.applyProposalTransaction(agentId, proposal));
      return { ok: true, proposal: applied };
    } catch (error: any) {
      const failed = this.store.updateGraphMaintenanceProposal(proposal.id, {
        status: 'failed_apply',
        reason: safeString(error?.message || error || 'apply_failed'),
      });
      return { ok: false, proposal: failed, reason: safeString(error?.message || 'apply_failed') };
    }
  }

  rejectProposal(agentId: string, proposalId: string, reason = 'operator_rejected'): { ok: boolean; proposal?: GraphMaintenanceProposal | null; reason?: string } {
    const proposal = this.store.getGraphMaintenanceProposal(proposalId);
    if (!proposal || proposal.agentId !== agentId) return { ok: false, reason: 'proposal_not_found' };
    if (proposal.status === 'applied') return { ok: false, proposal, reason: 'cannot_reject_applied_proposal' };
    const rejected = this.store.updateGraphMaintenanceProposal(proposal.id, {
      status: 'rejected',
      rejectedAt: now(),
      reviewedAt: now(),
      reviewedBy: 'operator',
      reason,
    });
    this.store.insertProofEvent({
      agentId,
      kind: 'graph_maintenance_rejected',
      rawTranscriptStored: false,
      payload: redactJsonValue({ proposalId, proposalType: proposal.proposalType, reason }),
    });
    return { ok: true, proposal: rejected };
  }

  explainProposal(agentId: string, proposalId: string): { ok: boolean; proposal?: GraphMaintenanceProposal | null; explanation?: Record<string, unknown>; reason?: string } {
    const proposal = this.store.getGraphMaintenanceProposal(proposalId);
    if (!proposal || proposal.agentId !== agentId) return { ok: false, reason: 'proposal_not_found' };
    return {
      ok: true,
      proposal,
      explanation: {
        boundary: 'This proposal may curate graph structure or features, but MemoryAuthorityResolver still decides turn-level use.',
        safeToAutoApply: isSafeApplyProposal(proposal),
        risk: proposal.risk,
        status: proposal.status,
        preconditions: proposal.preconditions,
        evidence: proposal.evidence,
        reviewRequiredReason: proposal.reviewRequiredReason || null,
      },
    };
  }

  private loadGraph(agentId: string, limit: number): LoadedGraph {
    const nodes = this.store.listAllMemoriesForMaintenance(agentId, limit);
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const validity = new Map<string, MemoryValidity>();
    for (const node of nodes) validity.set(node.id, this.store.getMemoryValidity(node.id) ?? defaultValidityForMemory(node));
    const edges = this.store.listEdgesForAgent(agentId, limit);
    const activeNodes = nodes.filter((node) => !node.deletedAt && !node.supersededBy && !isBlockedValidity(validity.get(node.id)));
    return { nodes, activeNodes, validity, edges, byId };
  }

  private duplicateMergeProposals(runId: string, graph: LoadedGraph): Array<Omit<GraphMaintenanceProposal, 'id' | 'createdAt'>> {
    return exactDuplicateClusters(graph)
      .filter((cluster) => cluster.length > 1)
      .map((cluster) => {
        const sorted = [...cluster].sort((a, b) =>
          (b.importance + b.confidence + b.captureCount * 0.02 + b.useCount * 0.01)
          - (a.importance + a.confidence + a.captureCount * 0.02 + a.useCount * 0.01)
        );
        const canonical = sorted[0];
        const duplicates = sorted.slice(1);
        return proposal({
          runId,
          agentId: canonical.agentId,
          proposalType: 'merge_exact_duplicate_nodes',
          targetKind: 'node',
          targetIds: sorted.map((node) => node.id),
          confidence: 0.99,
          risk: 'low',
          status: 'approved',
          reason: 'exact_duplicate_same_content_scope_type_privacy',
          proposedPatch: {
            canonicalId: canonical.id,
            duplicateIds: duplicates.map((node) => node.id),
            relationFamily: 'lineage',
            operation: 'supersede_duplicates_to_canonical',
          },
          evidence: {
            contentHash: hashText(canonical.content),
            redactedPreview: redactText(canonical.content, 180),
            normalizedKeys: sorted.map((node) => node.normalizedKey),
          },
          preconditions: {
            canonical: nodePrecondition(canonical),
            duplicates: duplicates.map(nodePrecondition),
            privacyClasses: sorted.map((node) => graph.validity.get(node.id)?.privacyClass || 'normal'),
          },
          riskFactors: {
            semanticRisk: 'low',
            privacyRisk: 'low',
            authorityImpact: 'low',
            retrievalImpact: 'low',
            deletionImpact: 'none',
            crossScopeImpact: false,
            llmInvolved: false,
            requiresUserReview: false,
          },
        });
      });
  }

  private badEdgeProposals(runId: string, graph: LoadedGraph): Array<Omit<GraphMaintenanceProposal, 'id' | 'createdAt'>> {
    return badEdgesForGraph(graph).map(({ edge, reason }) => proposal({
      runId,
      agentId: edge.agentId,
      proposalType: 'retire_bad_edge',
      targetKind: 'edge',
      targetIds: [edge.id],
      confidence: 0.98,
      risk: 'low',
      status: 'approved',
      reason,
      proposedPatch: {
        edgeId: edge.id,
        operation: 'delete_edge_and_record_retirement_observation',
      },
      evidence: {
        fromId: edge.fromId,
        toId: edge.toId,
        relation: edge.relation,
        reason,
      },
      preconditions: { edge: edgePrecondition(edge) },
      riskFactors: {
        semanticRisk: 'low',
        privacyRisk: 'low',
        authorityImpact: 'none',
        retrievalImpact: 'low',
        deletionImpact: 'soft',
        crossScopeImpact: false,
        llmInvolved: false,
        requiresUserReview: false,
      },
    }));
  }

  private staleAuthorityProposals(runId: string, graph: LoadedGraph): Array<Omit<GraphMaintenanceProposal, 'id' | 'createdAt'>> {
    return staleHighAuthorityNodes(graph).map(({ node, validity, reason }) => proposal({
      runId,
      agentId: node.agentId,
      proposalType: 'mark_stale_high_authority',
      targetKind: 'validity',
      targetIds: [node.id],
      confidence: 0.78,
      risk: 'medium',
      status: 'pending_review',
      reason,
      reviewRequiredReason: 'authority_feature_change_requires_review',
      proposedPatch: {
        memoryId: node.id,
        temporalValidity: 'stale',
        behavioralAvailability: validity.validationStrategy === 'environment_check' ? 'confirm_before_use' : validity.behavioralAvailability,
        stateReason: 'graph_maintenance_stale_high_authority',
      },
      evidence: {
        memoryId: node.id,
        normalizedKey: node.normalizedKey,
        type: node.type,
        freshness: node.freshness,
        behavioralAuthorityScore: validity.behavioralAuthorityScore,
        lastSeenAt: node.lastSeenAt,
        redactedPreview: redactText(node.content, 180),
      },
      preconditions: { node: nodePrecondition(node), validityUpdatedAt: validity.updatedAt },
      riskFactors: {
        semanticRisk: 'medium',
        privacyRisk: validity.privacyClass === 'normal' ? 'low' : 'high',
        authorityImpact: 'medium',
        retrievalImpact: 'low',
        deletionImpact: 'none',
        crossScopeImpact: false,
        llmInvolved: false,
        requiresUserReview: true,
      },
    }));
  }

  private tombstoneBlockProposals(runId: string, graph: LoadedGraph): Array<Omit<GraphMaintenanceProposal, 'id' | 'createdAt'>> {
    return tombstoneRecaptureCandidates(graph).map(({ tombstone, candidate, reason }) => proposal({
      runId,
      agentId: candidate.agentId,
      proposalType: 'block_tombstone_recapture',
      targetKind: 'tombstone',
      targetIds: [candidate.id, tombstone.id],
      confidence: 0.86,
      risk: 'high',
      status: 'pending_review',
      reason,
      reviewRequiredReason: 'privacy_tombstone_guard_requires_review',
      proposedPatch: {
        memoryId: candidate.id,
        operation: 'suppress_candidate_that_matches_tombstone_blocking_key',
        retentionState: 'soft_deleted',
        behavioralAvailability: 'never_use',
      },
      evidence: {
        candidateId: candidate.id,
        tombstoneId: tombstone.id,
        candidateKeyHash: hashText(candidate.normalizedKey),
        tombstoneKeyHash: hashText(tombstone.normalizedKey),
        safeCategory: 'memory_tombstone',
      },
      preconditions: { candidate: nodePrecondition(candidate), tombstone: nodePrecondition(tombstone) },
      riskFactors: {
        semanticRisk: 'medium',
        privacyRisk: 'high',
        authorityImpact: 'high',
        retrievalImpact: 'medium',
        deletionImpact: 'soft',
        crossScopeImpact: false,
        llmInvolved: false,
        requiresUserReview: true,
      },
    }));
  }

  private scopedExceptionProposals(runId: string, agentId: string): Array<Omit<GraphMaintenanceProposal, 'id' | 'createdAt'>> {
    return scopedExceptionCandidates(this.store.listMemoryAuthorityEvents(agentId, 500)).map(({ memoryId, events }) => proposal({
      runId,
      agentId,
      proposalType: 'propose_scoped_exception',
      targetKind: 'node',
      targetIds: [memoryId],
      confidence: Math.min(0.85, 0.55 + events.length * 0.08),
      risk: 'medium',
      status: 'pending_review',
      reason: 'repeated_current_instruction_override_for_memory',
      reviewRequiredReason: 'scoped_exception_creation_requires_review',
      proposedPatch: {
        sourceMemoryId: memoryId,
        operation: 'create_scoped_exception_candidate',
        relationFamily: 'scope',
      },
      evidence: {
        eventIds: events.map((event) => event.id),
        reasons: [...new Set(events.map((event) => event.reason).filter(Boolean))],
      },
      preconditions: { memoryId, minimumOverrideEvents: 2 },
      riskFactors: {
        semanticRisk: 'medium',
        privacyRisk: 'low',
        authorityImpact: 'medium',
        retrievalImpact: 'medium',
        deletionImpact: 'none',
        crossScopeImpact: true,
        llmInvolved: false,
        requiresUserReview: true,
      },
    }));
  }

  private feedbackObservationProposals(runId: string, agentId: string): Array<Omit<GraphMaintenanceProposal, 'id' | 'createdAt'>> {
    return this.store.listRouteTeacherRuns(agentId, 100)
      .filter((run) => run.validated && run.teacherMemoryIds.length >= 2 && run.verdict !== 'unknown')
      .slice(0, 20)
      .map((run) => proposal({
        runId,
        agentId,
        proposalType: 'record_feedback_edge_observation',
        targetKind: 'edge',
        targetIds: run.teacherMemoryIds.slice(0, 6),
        confidence: Math.max(0.2, Math.min(0.9, run.confidence)),
        risk: 'low',
        status: 'approved',
        reason: `route_teacher_${run.verdict}`,
        proposedPatch: {
          operation: 'record_edge_observation_only',
          relationFamily: 'behavioral',
          authorityImpact: 'none',
        },
        evidence: {
          routeTeacherRunId: run.id,
          routeDecisionId: run.routeDecisionId,
          verdict: run.verdict,
          confidence: run.confidence,
          memoryIds: run.teacherMemoryIds.slice(0, 6),
        },
        preconditions: {
          routeTeacherRunId: run.id,
          validated: true,
          memoryIds: run.teacherMemoryIds.slice(0, 6),
        },
        riskFactors: {
          semanticRisk: 'low',
          privacyRisk: 'low',
          authorityImpact: 'none',
          retrievalImpact: 'low',
          deletionImpact: 'none',
          crossScopeImpact: false,
          llmInvolved: true,
          requiresUserReview: false,
          antiSelfReinforcement: 'observation_only_no_authority_change',
        },
      }));
  }

  private applyProposalTransaction(agentId: string, proposal: GraphMaintenanceProposal): GraphMaintenanceProposal | null {
    if (proposal.proposalType === 'merge_exact_duplicate_nodes') {
      this.assertDuplicatePreconditions(proposal);
      const canonicalId = safeString((proposal.proposedPatch as any).canonicalId);
      const duplicateIds = Array.isArray((proposal.proposedPatch as any).duplicateIds) ? (proposal.proposedPatch as any).duplicateIds.map(safeString).filter(Boolean) : [];
      for (const duplicateId of duplicateIds) {
        const duplicate = this.store.getMemory(duplicateId);
        if (!duplicate || duplicate.supersededBy === canonicalId) continue;
        this.store.supersedeMemory(duplicateId, canonicalId);
        this.store.upsertEdge(agentId, duplicateId, canonicalId, 'supersedes');
        this.store.insertMemoryNodeLineage({
          agentId,
          childMemoryId: duplicateId,
          parentMemoryId: canonicalId,
          relation: 'merged_into',
          proposalId: proposal.id,
          evidence: { proposalId: proposal.id, relationFamily: 'lineage' },
        });
        this.store.insertMemoryEdgeObservation({
          agentId,
          fromId: duplicateId,
          toId: canonicalId,
          relation: 'merged_into',
          edgeFamily: 'lineage',
          edgeState: 'structural',
          observationType: 'canonical_merge_applied',
          delta: 0,
          sourceType: 'maintenance',
          sourceIndependence: 'independent',
          signalStrength: 'explicit',
          polarity: 'supersedes',
          causalAttribution: 'unknown',
          reason: 'exact_duplicate_merge',
        });
      }
      this.writeAppliedProof(agentId, proposal, { canonicalId, duplicateIds });
      return this.store.updateGraphMaintenanceProposal(proposal.id, {
        status: 'applied',
        appliedAt: now(),
        appliedDiff: { canonicalId, duplicateIds, lineage: 'merged_into' },
      });
    }

    if (proposal.proposalType === 'retire_bad_edge') {
      const edgeId = safeString((proposal.proposedPatch as any).edgeId || proposal.targetIds[0]);
      const edge = this.store.listEdgesForAgent(agentId, 5000).find((candidate) => candidate.id === edgeId);
      if (edge) {
        this.assertBadEdgeStillBad(edge);
        this.store.deleteEdge(edge.id, agentId);
        this.store.insertMemoryEdgeObservation({
          agentId,
          edgeId: edge.id,
          fromId: edge.fromId,
          toId: edge.toId,
          relation: edge.relation,
          edgeFamily: edgeFamilyForRelation(edge.relation),
          edgeState: 'retired',
          observationType: 'bad_edge_retired',
          delta: -1,
          sourceType: 'maintenance',
          sourceIndependence: 'independent',
          signalStrength: 'explicit',
          polarity: 'irrelevant',
          causalAttribution: 'unknown',
          reason: proposal.reason,
        });
      }
      this.writeAppliedProof(agentId, proposal, { edgeId, retired: Boolean(edge) });
      return this.store.updateGraphMaintenanceProposal(proposal.id, {
        status: 'applied',
        appliedAt: now(),
        appliedDiff: { edgeId, retired: Boolean(edge) },
      });
    }

    if (proposal.proposalType === 'record_feedback_edge_observation') {
      this.recordFeedbackObservation(agentId, proposal);
      this.writeAppliedProof(agentId, proposal, { recorded: true });
      return this.store.updateGraphMaintenanceProposal(proposal.id, {
        status: 'applied',
        appliedAt: now(),
        appliedDiff: { observationOnly: true, authorityImpact: 'none' },
      });
    }

    throw new Error('proposal_not_safe_for_apply');
  }

  private assertDuplicatePreconditions(proposal: GraphMaintenanceProposal) {
    const canonicalId = safeString((proposal.proposedPatch as any).canonicalId);
    const duplicateIds = Array.isArray((proposal.proposedPatch as any).duplicateIds) ? (proposal.proposedPatch as any).duplicateIds.map(safeString).filter(Boolean) : [];
    const ids = [canonicalId, ...duplicateIds].filter(Boolean);
    if (ids.length < 2) throw new Error('duplicate_precondition_missing_targets');
    const memories = ids.map((id) => this.store.getMemory(id));
    if (memories.some((memory) => !memory)) throw new Error('duplicate_precondition_missing_memory');
    const active = memories.filter(Boolean) as MemoryNode[];
    if (active.some((memory) => memory.deletedAt)) throw new Error('duplicate_precondition_deleted_memory');
    const contentHashes = new Set(active.map((memory) => hashText(normalizeText(memory.content))));
    const scopeKeys = new Set(active.map((memory) => `${memory.agentId}:${memory.type}:${memory.scopeKind}:${memory.scopeKey || ''}`));
    const privacy = new Set(active.map((memory) => this.store.getMemoryValidity(memory.id)?.privacyClass || 'normal'));
    if (contentHashes.size !== 1) throw new Error('duplicate_precondition_content_changed');
    if (scopeKeys.size !== 1) throw new Error('duplicate_precondition_scope_changed');
    if (privacy.size !== 1) throw new Error('duplicate_precondition_privacy_changed');
  }

  private assertBadEdgeStillBad(edge: MemoryEdge) {
    const from = this.store.getMemory(edge.fromId);
    const to = this.store.getMemory(edge.toId);
    if (!from || !to) return;
    const fromValidity = this.store.getMemoryValidity(edge.fromId) ?? defaultValidityForMemory(from);
    const toValidity = this.store.getMemoryValidity(edge.toId) ?? defaultValidityForMemory(to);
    if (from.deletedAt || to.deletedAt || isBlockedValidity(fromValidity) || isBlockedValidity(toValidity)) return;
    throw new Error('edge_precondition_no_longer_bad');
  }

  private recordFeedbackObservation(agentId: string, proposal: GraphMaintenanceProposal) {
    const memoryIds = Array.isArray((proposal.evidence as any).memoryIds) ? (proposal.evidence as any).memoryIds.map(safeString).filter(Boolean).slice(0, 6) : [];
    const verdict = safeString((proposal.evidence as any).verdict);
    const polarity = verdict === 'correct_route' ? 'supports' : 'irrelevant';
    const delta = verdict === 'correct_route' ? 0.05 : -0.03;
    for (let i = 0; i < memoryIds.length; i += 1) {
      for (let j = i + 1; j < memoryIds.length; j += 1) {
        this.store.insertMemoryEdgeObservation({
          agentId,
          fromId: memoryIds[i],
          toId: memoryIds[j],
          relation: 'co_injected_outcome_observed',
          edgeFamily: 'behavioral',
          edgeState: 'candidate',
          observationType: 'route_teacher_behavioral_observation',
          delta,
          sourceType: 'route_teacher',
          sourceIndependence: 'derived',
          signalStrength: 'weak',
          polarity,
          causalAttribution: 'unknown',
          routeId: safeString((proposal.evidence as any).routeDecisionId),
          reason: `observation_only:${verdict || 'unknown'}`,
        });
      }
    }
  }

  private writeAppliedProof(agentId: string, proposal: GraphMaintenanceProposal, appliedDiff: Record<string, unknown>) {
    this.store.insertProofEvent({
      agentId,
      kind: 'graph_maintenance_applied',
      rawTranscriptStored: false,
      payload: redactJsonValue({
        proposalId: proposal.id,
        proposalType: proposal.proposalType,
        risk: proposal.risk,
        reason: proposal.reason,
        appliedDiff,
        invariant: 'maintenance_features_only_authority_recomputed_at_turn_time',
      }),
    });
  }
}

export function graphMaintenancePayload(config: any, req: any = {}, action = 'health') {
  const agentId = agentIdFromGraphRequest(req, config);
  if (!isAgentAllowed(config, agentId)) return { ok: false, agentId, reason: 'agent_not_allowed' };
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const engine = new GraphMaintenanceEngine({ store, config });
  try {
    const limit = numberParam(req, 'limit', 1000);
    if (action === 'health') return engine.health(agentId, limit);
    if (action === 'dry-run') return engine.dryRun(agentId, { limit });
    if (action === 'proposals') {
      return {
        ok: true,
        agentId,
        proposals: store.listGraphMaintenanceProposals(agentId, {
          status: stringParam(req, 'status') || undefined,
          limit: numberParam(req, 'limit', 50),
          runId: stringParam(req, 'runId') || undefined,
        }),
      };
    }
    if (action === 'apply') return engine.applyProposal(agentId, requiredParam(req, 'proposalId'));
    if (action === 'reject') return engine.rejectProposal(agentId, requiredParam(req, 'proposalId'), stringParam(req, 'reason') || 'operator_rejected');
    if (action === 'stale') {
      const graph = engine.health(agentId, limit);
      const staleProposals = store.listGraphMaintenanceProposals(agentId, { limit: 200 })
        .filter((proposal) => proposal.proposalType === 'mark_stale_high_authority');
      return { ok: true, agentId, count: graph.counts.staleHighAuthorityNodes, proposals: staleProposals };
    }
    if (action === 'clusters') {
      const nodes = store.listAllMemoriesForMaintenance(agentId, limit);
      const graph = {
        nodes,
        activeNodes: nodes.filter((node) => !node.deletedAt && !node.supersededBy),
        validity: new Map(nodes.map((node) => [node.id, store.getMemoryValidity(node.id) ?? defaultValidityForMemory(node)])),
        edges: store.listEdgesForAgent(agentId, limit),
        byId: new Map(nodes.map((node) => [node.id, node])),
      };
      return { ok: true, agentId, clusters: exactDuplicateClusters(graph).filter((cluster) => cluster.length > 1).map(renderCluster) };
    }
    if (action === 'tombstones') {
      const nodes = store.listAllMemoriesForMaintenance(agentId, limit);
      const rows = nodes
        .map((node) => ({ node, validity: store.getMemoryValidity(node.id) ?? defaultValidityForMemory(node) }))
        .filter((row) => row.validity.retentionState === 'tombstoned')
        .map((row) => ({
          id: row.node.id,
          type: row.node.type,
          normalizedKeyHash: hashText(row.node.normalizedKey),
          scopeKind: row.node.scopeKind,
          scopeKeyHash: hashText(row.node.scopeKey || ''),
          stateReason: row.validity.stateReason || null,
          updatedAt: row.validity.updatedAt,
        }));
      return { ok: true, agentId, tombstones: rows };
    }
    if (action === 'explain') return engine.explainProposal(agentId, requiredParam(req, 'proposalId'));
    return { ok: false, agentId, reason: `unknown_graph_maintenance_action:${action}` };
  } finally {
    store.close();
  }
}

export async function handleGraphBrainCommand(ctx: any, args: string[], config: any): Promise<{ text: string; continueAgent?: boolean }> {
  const agentId = safeString(ctx.agentId ?? config.scopes?.agents?.[0] ?? 'main') || 'main';
  if (!isAgentAllowed(config, agentId)) return { text: `OpenClawBrain is not enabled for agent ${agentId}.` };
  const [subcommand = 'health', ...rest] = args;
  const req = requestFromArgs(rest);
  const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
  const engine = new GraphMaintenanceEngine({ store, config });
  try {
    if (subcommand === 'health') return { text: formatGraphHealth(engine.health(agentId)) };
    if (subcommand === 'dry-run' || subcommand === 'dryrun') return { text: formatDryRun(engine.dryRun(agentId)) };
    if (subcommand === 'proposals') return { text: formatProposals(store.listGraphMaintenanceProposals(agentId, { status: req.status, limit: Number(req.limit || 10) || 10 })) };
    if (subcommand === 'apply') return { text: formatApplyResult(engine.applyProposal(agentId, safeString(rest[0] || req.proposalId))) };
    if (subcommand === 'reject') return { text: formatApplyResult(engine.rejectProposal(agentId, safeString(rest[0] || req.proposalId), safeString(req.reason || 'operator_rejected'))) };
    if (subcommand === 'stale') {
      const report = engine.dryRun(agentId);
      return { text: formatProposals(report.proposals.filter((proposal) => proposal.proposalType === 'mark_stale_high_authority')) };
    }
    if (subcommand === 'clusters') {
      const payload = graphMaintenancePayload(config, { query: { agentId, limit: req.limit || 1000 } }, 'clusters') as any;
      return { text: formatClusters(payload.clusters || []) };
    }
    if (subcommand === 'tombstones') {
      const payload = graphMaintenancePayload(config, { query: { agentId, limit: req.limit || 1000 } }, 'tombstones') as any;
      return { text: formatTombstones(payload.tombstones || []) };
    }
    if (subcommand === 'explain') {
      const result = engine.explainProposal(agentId, safeString(rest[0] || req.proposalId));
      return { text: result.ok ? formatProposalExplanation(result as any) : `No graph proposal found: ${result.reason}` };
    }
    return { text: graphHelpText() };
  } finally {
    store.close();
  }
}

export function graphHelpText() {
  return [
    'OpenClawBrain graph commands:',
    '/brain graph health',
    '/brain graph dry-run',
    '/brain graph proposals',
    '/brain graph apply <proposalId>',
    '/brain graph reject <proposalId>',
    '/brain graph stale',
    '/brain graph clusters',
    '/brain graph tombstones',
    '/brain graph explain <proposalId>',
  ].join('\n');
}

function proposal(input: Omit<GraphMaintenanceProposal, 'id' | 'createdAt' | 'proposalHash' | 'appliedDiff' | 'graphVersion'> & { appliedDiff?: Record<string, unknown>; graphVersion?: number }): Omit<GraphMaintenanceProposal, 'id' | 'createdAt'> {
  const basis = {
    agentId: input.agentId,
    proposalType: input.proposalType,
    targetKind: input.targetKind,
    targetIds: input.targetIds,
    proposedPatch: input.proposedPatch,
    preconditions: input.preconditions,
  };
  return {
    ...input,
    appliedDiff: input.appliedDiff ?? {},
    proposalHash: shortHash(JSON.stringify(basis)),
    graphVersion: input.graphVersion ?? 1,
  };
}

function exactDuplicateClusters(graph: LoadedGraph): MemoryNode[][] {
  const groups = new Map<string, MemoryNode[]>();
  for (const node of graph.activeNodes) {
    const validity = graph.validity.get(node.id);
    if (validity && validity.privacyClass !== 'normal') continue;
    const key = [
      node.agentId,
      node.type,
      node.scopeKind,
      node.scopeKey || '',
      validity?.privacyClass || 'normal',
      hashText(normalizeText(node.content)),
    ].join('|');
    const group = groups.get(key) || [];
    group.push(node);
    groups.set(key, group);
  }
  return [...groups.values()];
}

function badEdgesForGraph(graph: LoadedGraph): Array<{ edge: MemoryEdge; reason: string }> {
  return graph.edges.flatMap((edge) => {
    const from = graph.byId.get(edge.fromId);
    const to = graph.byId.get(edge.toId);
    if (!from || !to) return [{ edge, reason: 'edge_points_to_missing_node' }];
    const fromValidity = graph.validity.get(from.id);
    const toValidity = graph.validity.get(to.id);
    if (from.deletedAt || to.deletedAt) return [{ edge, reason: 'edge_points_to_deleted_node' }];
    if (isBlockedValidity(fromValidity) || isBlockedValidity(toValidity)) return [{ edge, reason: 'edge_points_to_blocked_or_tombstoned_node' }];
    return [];
  });
}

function staleHighAuthorityNodes(graph: LoadedGraph): Array<{ node: MemoryNode; validity: MemoryValidity; reason: string }> {
  return graph.activeNodes.flatMap((node) => {
    const validity = graph.validity.get(node.id) ?? defaultValidityForMemory(node);
    const stale = validity.temporalValidity === 'stale'
      || validity.temporalValidity === 'expired'
      || node.freshness < 0.35
      || (validity.revalidateAfter ? Date.parse(validity.revalidateAfter) <= Date.now() : false);
    const highAuthority = validity.behavioralAuthorityScore >= 0.7 || node.importance >= 0.75;
    if (!stale || !highAuthority) return [];
    return [{ node, validity, reason: validity.temporalValidity === 'expired' ? 'expired_high_authority_memory' : 'stale_high_authority_memory' }];
  });
}

function tombstoneRecaptureCandidates(graph: LoadedGraph): Array<{ tombstone: MemoryNode; candidate: MemoryNode; reason: string }> {
  const tombstones = graph.nodes.filter((node) => graph.validity.get(node.id)?.retentionState === 'tombstoned');
  const active = graph.activeNodes;
  const results: Array<{ tombstone: MemoryNode; candidate: MemoryNode; reason: string }> = [];
  for (const tombstone of tombstones) {
    for (const candidate of active) {
      if (candidate.id === tombstone.id) continue;
      if (candidate.agentId !== tombstone.agentId) continue;
      if (candidate.scopeKind !== tombstone.scopeKind || (candidate.scopeKey || '') !== (tombstone.scopeKey || '')) continue;
      if (candidate.normalizedKey === tombstone.normalizedKey || candidate.normalizedKey.startsWith(`${tombstone.normalizedKey}:`)) {
        results.push({ tombstone, candidate, reason: 'candidate_matches_tombstone_normalized_key' });
      }
    }
  }
  return results;
}

function scopedExceptionCandidates(events: MemoryAuthorityEvent[]): Array<{ memoryId: string; events: MemoryAuthorityEvent[] }> {
  const groups = new Map<string, MemoryAuthorityEvent[]>();
  for (const event of events) {
    if (event.eventType !== 'overridden_by_current_instruction' && !String(event.reason || '').startsWith('current_instruction:')) continue;
    const list = groups.get(event.memoryId) || [];
    list.push(event);
    groups.set(event.memoryId, list);
  }
  return [...groups.entries()].filter(([, list]) => list.length >= 2).map(([memoryId, list]) => ({ memoryId, events: list }));
}

function isSafeApplyProposal(proposal: GraphMaintenanceProposal) {
  if (proposal.risk !== 'low') return false;
  return ['merge_exact_duplicate_nodes', 'retire_bad_edge', 'record_feedback_edge_observation'].includes(proposal.proposalType);
}

function isBlockedValidity(validity?: MemoryValidity | null) {
  return Boolean(validity && ['soft_deleted', 'tombstoned', 'hard_deleted', 'redacted'].includes(validity.retentionState));
}

function nodePrecondition(node: MemoryNode) {
  return {
    id: node.id,
    updatedAt: node.updatedAt,
    deletedAt: node.deletedAt || null,
    supersededBy: node.supersededBy || null,
    type: node.type,
    scopeKind: node.scopeKind,
    scopeKey: node.scopeKey || '',
    contentHash: hashText(normalizeText(node.content)),
  };
}

function edgePrecondition(edge: MemoryEdge) {
  return {
    id: edge.id,
    fromId: edge.fromId,
    toId: edge.toId,
    relation: edge.relation,
    updatedAt: edge.updatedAt,
  };
}

function edgeFamilyForRelation(relation: string): 'epistemic' | 'temporal' | 'scope' | 'behavioral' | 'lineage' | 'retention' {
  if (relation === 'contradicts') return 'epistemic';
  if (relation === 'supersedes') return 'temporal';
  if (relation === 'used_with') return 'behavioral';
  if (relation === 'supports_workflow') return 'behavioral';
  return 'behavioral';
}

function normalizeText(value: string) {
  return value.toLowerCase().replace(/\s+/g, ' ').trim();
}

function issue(kind: string, count: number, severity: 'low' | 'medium' | 'high', nextAction: string) {
  return { kind, count, severity, nextAction };
}

function riskSummaryFor(proposals: GraphMaintenanceProposal[]) {
  return proposals.reduce((acc: Record<string, number>, proposal) => {
    acc[proposal.risk] = (acc[proposal.risk] || 0) + 1;
    return acc;
  }, {});
}

function renderCluster(cluster: MemoryNode[]) {
  return {
    size: cluster.length,
    canonicalCandidateId: cluster[0]?.id,
    ids: cluster.map((node) => node.id),
    type: cluster[0]?.type,
    scope: `${cluster[0]?.scopeKind}:${cluster[0]?.scopeKey || ''}`,
    contentHash: cluster[0] ? hashText(normalizeText(cluster[0].content)) : null,
    preview: cluster[0] ? redactText(cluster[0].content, 160) : '',
  };
}

function formatGraphHealth(health: GraphMaintenanceHealth) {
  const c = health.counts;
  const issues = health.topIssues.length
    ? health.topIssues.map((item) => `- ${item.kind}: ${item.count} (${item.severity})`).join('\n')
    : '- no immediate graph maintenance issues';
  return [
    `OpenClawBrain graph health for ${health.agentId}`,
    `Nodes: ${c.activeNodes}/${c.nodes} active. Edges: ${c.edges}.`,
    `Duplicate clusters: ${c.exactDuplicateClusters}. Bad edges: ${c.badEdges}. Stale high-authority memories: ${c.staleHighAuthorityNodes}.`,
    `Tombstone recapture candidates: ${c.tombstoneRecaptureCandidates}. Scoped exception candidates: ${c.scopedExceptionCandidates}.`,
    '',
    'Top issues:',
    issues,
    '',
    'Invariant: graph maintenance proposes/features only; Memory Authority still decides turn-level use.',
  ].join('\n');
}

function formatDryRun(report: GraphMaintenanceDryRunReport) {
  return [
    `Graph maintenance dry-run complete for ${report.agentId}.`,
    `Run: ${report.run.id}`,
    `Proposals: ${report.proposals.length} (${Object.entries(riskSummaryFor(report.proposals)).map(([risk, count]) => `${risk}:${count}`).join(', ') || 'none'})`,
    'Use /brain graph proposals to inspect, /brain graph apply <proposalId> for low-risk deterministic proposals, or /brain graph reject <proposalId>.',
  ].join('\n');
}

function formatProposals(proposals: GraphMaintenanceProposal[]) {
  if (!proposals.length) return 'No graph maintenance proposals found.';
  return [
    `Graph maintenance proposals (${proposals.length}):`,
    ...proposals.slice(0, 20).map((proposal) => [
      `- ${proposal.id}`,
      `${proposal.proposalType}`,
      `risk=${proposal.risk}`,
      `status=${proposal.status}`,
      `reason=${proposal.reason}`,
    ].join(' | ')),
  ].join('\n');
}

function formatApplyResult(result: { ok: boolean; proposal?: GraphMaintenanceProposal | null; reason?: string }) {
  if (!result.ok) return `Graph proposal not applied: ${result.reason || 'unknown'}${result.proposal ? ` (${result.proposal.id})` : ''}`;
  return `Graph proposal handled: ${result.proposal?.id || 'unknown'} status=${result.proposal?.status || 'unknown'}`;
}

function formatClusters(clusters: any[]) {
  if (!clusters.length) return 'No exact duplicate graph clusters found.';
  return ['Exact duplicate graph clusters:', ...clusters.slice(0, 20).map((cluster) => `- size=${cluster.size} canonical=${cluster.canonicalCandidateId} ${cluster.preview}`)].join('\n');
}

function formatTombstones(tombstones: any[]) {
  if (!tombstones.length) return 'No tombstoned memories found.';
  return ['Tombstones:', ...tombstones.slice(0, 20).map((row) => `- ${row.id} ${row.type} key=${row.normalizedKeyHash}`)].join('\n');
}

function formatProposalExplanation(result: { proposal: GraphMaintenanceProposal; explanation: Record<string, unknown> }) {
  const proposal = result.proposal;
  return [
    `Proposal ${proposal.id}`,
    `Type: ${proposal.proposalType}`,
    `Status: ${proposal.status}`,
    `Risk: ${proposal.risk}`,
    `Safe auto apply: ${(result.explanation as any).safeToAutoApply ? 'yes' : 'no'}`,
    `Reason: ${proposal.reason}`,
    `Boundary: ${(result.explanation as any).boundary}`,
  ].join('\n');
}

function requestFromArgs(args: string[]) {
  const result: Record<string, string> = {};
  for (const arg of args) {
    const match = /^--?([^=]+)=(.*)$/.exec(arg);
    if (match) result[match[1]] = match[2];
  }
  return result;
}

function agentIdFromGraphRequest(req: any = {}, config: any = {}) {
  return stringParam(req, 'agentId') || stringParam(req, 'agent') || safeString(config.scopes?.agents?.[0] || 'main') || 'main';
}

function numberParam(req: any = {}, key: string, fallback: number) {
  const value = stringParam(req, key);
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function requiredParam(req: any = {}, key: string) {
  const value = stringParam(req, key);
  if (!value) throw new Error(`missing_required_param:${key}`);
  return value;
}

function stringParam(req: any = {}, key: string) {
  if (req.query?.[key] != null) return safeString(req.query[key]);
  if (req.body?.[key] != null) return safeString(req.body[key]);
  try {
    const url = new URL(req.url || 'http://local/', 'http://local');
    return safeString(url.searchParams.get(key) || '');
  } catch {
    return '';
  }
}
