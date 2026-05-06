import type { RoutePolicyRuleV3 } from './memory-types.js';
export interface RoutePolicyV3CompactnessSummary {
    beforeMerge: number;
    afterMerge: number;
    afterPrune: number;
    duplicateGroups: number;
    mergedAway: number;
    dominatedPruned: number;
    avgSignalsPerRule: number;
    avgQueriesPerRule: number;
    maxRulesPerRoute: number;
}
export declare function normalizeQueryTemplateFamilyV3(values: any[], limit?: number): string[];
export declare function normalizeSignalFamilyV3(values: any[], limit?: number): string[];
export declare function canonicalActionKeyV3(rule: Pick<RoutePolicyRuleV3, 'route' | 'memoryTypes' | 'graphDepth' | 'syncPlanner' | 'queries' | 'match'>): string;
export declare function mergeRuleCandidatesV3(rules: RoutePolicyRuleV3[], config?: any): {
    rules: RoutePolicyRuleV3[];
    duplicateGroups: number;
    mergedAway: number;
};
export declare function pruneDominatedRulesV3(rules: RoutePolicyRuleV3[]): {
    rules: RoutePolicyRuleV3[];
    prunedRuleIds: string[];
};
export declare function compactnessSummaryV3(beforeMergeRules: RoutePolicyRuleV3[], afterMergeRules: RoutePolicyRuleV3[], finalRules: RoutePolicyRuleV3[], duplicateGroups: number, dominatedPruned: number): RoutePolicyV3CompactnessSummary;
