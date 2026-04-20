export type MemoryBackend =
  | "none"
  | "correction-only"
  | "correction-plus-heuristics"
  | "full-ocb";

export type TurnSlice =
  | "direct-answer"
  | "continuation"
  | "correction-follow-up"
  | "retrieval-heavy"
  | "tool-heavy"
  | "stale-memory-conflict"
  | "unclassified";

export interface RetrievedItem {
  source_id: string;
  content: string;
  score: number;
  age_seconds: number;
}

export interface Decision {
  decision_id: string;
  run_id: string;
  case_id: string;
  turn_index: number;
  backend: MemoryBackend;
  slice: TurnSlice;
  gate_score: number | null;
  gate_threshold: number | null;
  fired: boolean;
  retrieved: RetrievedItem[];
  injected_tokens: number;
  query_text: string;
  timestamp_ms: number;
}

export interface Outcome {
  decision_id: string;
  run_id: string;
  task_passed: boolean;
  used_retrieved_content: boolean;
  response_text: string;
  response_tokens: number;
  latency_ms: number;
  counterfactual_backend: MemoryBackend | null;
  counterfactual_passed: boolean | null;
  timestamp_ms: number;
}

export interface TaskCase {
  case_id: string;
  slice: TurnSlice;
  description: string;
  setup_turns: Array<{ role: "user" | "assistant"; content: string }>;
  filler_turns: number;
  query: string;
  success_criteria: SuccessCriteria;
}

export type SuccessCriteria =
  | {
      type: "contains_all";
      positive_signals: string[];
      negative_signals?: string[];
      case_sensitive?: boolean;
    }
  | {
      type: "contains_any";
      positive_signals: string[];
      negative_signals?: string[];
      case_sensitive?: boolean;
    }
  | {
      type: "regex";
      pattern: string;
      must_not_match?: string;
    };

export interface AblationResult {
  backend: MemoryBackend;
  slice: TurnSlice | "all";
  total_cases: number;
  total_fires: number;
  pass_rate: number;
  fire_conditional_pass_rate: number | null;
  nofire_conditional_pass_rate: number | null;
  abstention_regret_count: number;
  false_fire_harm_count: number;
  mean_injected_tokens: number;
  mean_response_tokens: number;
  tokens_per_pass: number;
}
