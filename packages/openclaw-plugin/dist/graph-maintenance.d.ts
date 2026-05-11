import type { GraphMaintenanceProposal, GraphMaintenanceRun } from './memory-types.js';
import { MemoryStore } from './memory-store.js';
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
    topIssues: Array<{
        kind: string;
        count: number;
        severity: 'low' | 'medium' | 'high';
        nextAction: string;
    }>;
}
export interface GraphMaintenanceDryRunReport {
    ok: boolean;
    agentId: string;
    run: GraphMaintenanceRun;
    health: GraphMaintenanceHealth;
    proposals: GraphMaintenanceProposal[];
}
export interface GraphMaintenanceAutomaticReport extends GraphMaintenanceDryRunReport {
    safeAutoApply: boolean;
    applied: Array<{
        proposalId: string;
        proposalType: string;
        ok: boolean;
        reason?: string;
    }>;
}
export declare class GraphMaintenanceEngine {
    private store;
    private config;
    constructor(options: GraphMaintenanceEngineOptions);
    health(agentId: string, limit?: number): GraphMaintenanceHealth;
    dryRun(agentId: string, options?: {
        limit?: number;
        mode?: GraphMaintenanceRun['mode'];
    }): GraphMaintenanceDryRunReport;
    runAutomatic(agentId: string, options?: {
        limit?: number;
        safeAutoApply?: boolean;
        maxSafeAutoApply?: number;
    }): GraphMaintenanceAutomaticReport;
    applyProposal(agentId: string, proposalId: string): {
        ok: boolean;
        proposal?: GraphMaintenanceProposal | null;
        reason?: string;
    };
    rejectProposal(agentId: string, proposalId: string, reason?: string): {
        ok: boolean;
        proposal?: GraphMaintenanceProposal | null;
        reason?: string;
    };
    explainProposal(agentId: string, proposalId: string): {
        ok: boolean;
        proposal?: GraphMaintenanceProposal | null;
        explanation?: Record<string, unknown>;
        reason?: string;
    };
    private loadGraph;
    private duplicateMergeProposals;
    private badEdgeProposals;
    private staleAuthorityProposals;
    private tombstoneBlockProposals;
    private scopedExceptionProposals;
    private feedbackObservationProposals;
    private applyProposalTransaction;
    private assertDuplicatePreconditions;
    private assertBadEdgeStillBad;
    private recordFeedbackObservation;
    private writeAppliedProof;
}
export declare function graphMaintenancePayload(config: any, req?: any, action?: string): GraphMaintenanceHealth | GraphMaintenanceDryRunReport | {
    ok: boolean;
    proposal?: GraphMaintenanceProposal | null;
    reason?: string;
} | {
    ok: boolean;
    agentId: string;
    reason: string;
    proposals?: undefined;
    count?: undefined;
    clusters?: undefined;
    tombstones?: undefined;
} | {
    ok: boolean;
    agentId: string;
    proposals: GraphMaintenanceProposal[];
    reason?: undefined;
    count?: undefined;
    clusters?: undefined;
    tombstones?: undefined;
} | {
    ok: boolean;
    agentId: string;
    count: number;
    proposals: GraphMaintenanceProposal[];
    reason?: undefined;
    clusters?: undefined;
    tombstones?: undefined;
} | {
    ok: boolean;
    agentId: string;
    clusters: {
        size: number;
        canonicalCandidateId: string;
        ids: string[];
        type: import("./memory-types.js").MemoryType;
        scope: string;
        contentHash: string | null;
        preview: string;
    }[];
    reason?: undefined;
    proposals?: undefined;
    count?: undefined;
    tombstones?: undefined;
} | {
    ok: boolean;
    agentId: string;
    tombstones: {
        id: string;
        type: import("./memory-types.js").MemoryType;
        normalizedKeyHash: string;
        scopeKind: "global_user" | "agent" | "repo" | "project" | "app" | "person" | "channel" | "session" | "task" | "tool";
        scopeKeyHash: string;
        stateReason: string | null;
        updatedAt: string;
    }[];
    reason?: undefined;
    proposals?: undefined;
    count?: undefined;
    clusters?: undefined;
};
export declare function handleGraphBrainCommand(ctx: any, args: string[], config: any): Promise<{
    text: string;
    continueAgent?: boolean;
}>;
export declare function graphHelpText(): string;
