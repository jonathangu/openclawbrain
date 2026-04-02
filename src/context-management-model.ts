import type { OpenClawBrainRuntimeConfig, LcmConfig } from "./db/config.js";
import type { BrainPrefetchState } from "./brain-core/types.js";
import type { SummaryFreshnessState } from "./store/summary-store.js";

const SUMMARY_FRESHNESS_STATES: SummaryFreshnessState[] = [
  "fresh",
  "stale_source",
  "stale_branch",
  "stale_pack",
  "superseded",
  "tombstoned",
];

const NON_FRESH_SUMMARY_STATES: Exclude<SummaryFreshnessState, "fresh">[] = [
  "stale_source",
  "stale_branch",
  "stale_pack",
  "superseded",
  "tombstoned",
];

const PREFETCH_LIFECYCLE_STATES: BrainPrefetchState[] = [
  "scheduled",
  "materialized",
  "hit",
  "miss",
  "stale",
  "invalidated",
  "dropped",
];

const SUMMARY_ROUTING_MODES = [
  "ignore",
  "summary_suffices",
  "expand_to_source",
  "prefer_typed_memory",
] as const;

export interface ContextManagementModel {
  model: "summary_spine_with_protected_fresh_tail";
  operatorSummary: string;
  hotContext: {
    layout: "summary_spine_then_protected_fresh_tail";
    sourceOfTruth: "context_items";
    freshTailCount: number;
    detail: string;
  };
  freshness: {
    states: SummaryFreshnessState[];
    nonFreshStates: Exclude<SummaryFreshnessState, "fresh">[];
    detail: string;
  };
  expansion: {
    summaryRoutingModes: Array<(typeof SUMMARY_ROUTING_MODES)[number]>;
    maxExpandTokens: number;
    detail: string;
  };
  prefetch: {
    lifecycleStates: BrainPrefetchState[];
    keyDimensions: string[];
    detail: string;
  };
  budget: {
    detail: string;
    controls: {
      freshTailCount: {
        env: "LCM_FRESH_TAIL_COUNT";
        value: number;
      };
      maxExpandTokens: {
        env: "LCM_MAX_EXPAND_TOKENS";
        value: number;
      };
      learnedQueryBudgetFraction: {
        env: "OPENCLAWBRAIN_BUDGET_FRACTION";
        value: number;
      };
    };
    perTurnStatusFields: string[];
  };
}

export function buildContextManagementModel(params: {
  lcmConfig: Pick<LcmConfig, "freshTailCount" | "maxExpandTokens">;
  brainConfig: Pick<OpenClawBrainRuntimeConfig, "budgetFraction">;
}): ContextManagementModel {
  const freshTailCount = Math.max(0, Math.floor(params.lcmConfig.freshTailCount));
  return {
    model: "summary_spine_with_protected_fresh_tail",
    operatorSummary:
      "Hot context is the ordered summary spine plus the protected fresh tail of raw messages. Prefetch only stages traversal work ahead of a turn, summary freshness decides when recap can be trusted versus expanded back to source, and budget controls keep both recall and injected learned context bounded.",
    hotContext: {
      layout: "summary_spine_then_protected_fresh_tail",
      sourceOfTruth: "context_items",
      freshTailCount,
      detail:
        freshTailCount > 0
          ? `The assembler resolves ordered context_items into a summary spine plus the newest ${freshTailCount} raw message(s). That fresh tail is protected from truncation and compaction, even when older items are clipped to fit budget.`
          : "The assembler resolves ordered context_items into the current summary spine. No raw-message tail is protected because the configured fresh tail count is 0.",
    },
    freshness: {
      states: [...SUMMARY_FRESHNESS_STATES],
      nonFreshStates: [...NON_FRESH_SUMMARY_STATES],
      detail:
        "Summary lineages carry freshness state. Anything outside fresh is a locator map, not proof: use it to find the region, then expand back to source before exact commands, quotes, paths, timestamps, or current-truth claims.",
    },
    expansion: {
      summaryRoutingModes: [...SUMMARY_ROUTING_MODES],
      maxExpandTokens: params.lcmConfig.maxExpandTokens,
      detail:
        "Expansion is the honest escape hatch when recap is too compressed. Summary routing can ignore summaries, allow summary-level recap, prefer typed memory, or force expand-to-source when the turn is precision-sensitive or branch-heavy.",
    },
    prefetch: {
      lifecycleStates: [...PREFETCH_LIFECYCLE_STATES],
      keyDimensions: [
        "queryDigest",
        "activePackVersion",
        "budgetClass",
        "summaryRoutingMode",
        "kind",
      ],
      detail:
        "Prefetch is an opportunistic traversal cache, not a second memory tier. It can be reused on a hit, but pack changes, budget-class changes, or summary-routing changes can make prefetched work stale or invalidated before serve time.",
    },
    budget: {
      detail:
        "LCM protects the fresh tail first. OpenClawBrain separately allocates learned-query budget from the turn budget, and turn traces can additionally cap injected learned context without hiding the retrieval accounting.",
      controls: {
        freshTailCount: {
          env: "LCM_FRESH_TAIL_COUNT",
          value: freshTailCount,
        },
        maxExpandTokens: {
          env: "LCM_MAX_EXPAND_TOKENS",
          value: params.lcmConfig.maxExpandTokens,
        },
        learnedQueryBudgetFraction: {
          env: "OPENCLAWBRAIN_BUDGET_FRACTION",
          value: params.brainConfig.budgetFraction,
        },
      },
      perTurnStatusFields: [
        "lastAssemblyDecision.queryBudgetChars",
        "lastAssemblyDecision.maxContextChars",
        "lastAssemblyDecision.injectedChars",
        "lastAssemblyDecision.droppedChars",
        "lastCompileReportSummary",
        "lastPrefetchDecision",
        "recentPrefetchSummary",
      ],
    },
  };
}
