import { mkdirSync, writeFileSync } from "node:fs";

import { generateResults } from "../src/results/generate.js";
import type { Ledger } from "../src/ledger/ledger.js";
import type { AblationResult, MemoryBackend, TurnSlice } from "../src/types.js";

const synthetic: Record<string, AblationResult> = {};

function set(backend: MemoryBackend, slice: TurnSlice | "all", partial: Partial<AblationResult>): void {
  const full: AblationResult = {
    backend,
    slice,
    total_cases: 50,
    total_fires: 0,
    pass_rate: 0,
    fire_conditional_pass_rate: null,
    nofire_conditional_pass_rate: null,
    abstention_regret_count: 0,
    false_fire_harm_count: 0,
    mean_injected_tokens: 0,
    mean_response_tokens: 120,
    tokens_per_pass: 0,
    ...partial,
  };
  full.tokens_per_pass =
    full.pass_rate > 0
      ? (full.total_cases * (full.mean_injected_tokens + full.mean_response_tokens)) /
        (full.total_cases * full.pass_rate)
      : Number.POSITIVE_INFINITY;
  synthetic[`${backend}|${slice}`] = full;
}

set("none", "all", { pass_rate: 0.42 });
set("none", "direct-answer", { pass_rate: 0.96 });
set("none", "correction-follow-up", { pass_rate: 0.12 });
set("none", "continuation", { pass_rate: 0.48 });
set("none", "stale-memory-conflict", { pass_rate: 0.30 });

set("correction-only", "all", { pass_rate: 0.74, total_fires: 28, mean_injected_tokens: 42, abstention_regret_count: 3, false_fire_harm_count: 1 });
set("correction-only", "direct-answer", { pass_rate: 0.96, total_fires: 0, mean_injected_tokens: 0 });
set("correction-only", "correction-follow-up", { pass_rate: 0.84, total_fires: 20, mean_injected_tokens: 55, abstention_regret_count: 1 });
set("correction-only", "continuation", { pass_rate: 0.68, total_fires: 6, mean_injected_tokens: 34 });
set("correction-only", "stale-memory-conflict", { pass_rate: 0.42, total_fires: 2, mean_injected_tokens: 38, false_fire_harm_count: 1 });

set("correction-plus-heuristics", "all", { pass_rate: 0.78, total_fires: 35, mean_injected_tokens: 88, abstention_regret_count: 2, false_fire_harm_count: 3 });
set("correction-plus-heuristics", "direct-answer", { pass_rate: 0.92, total_fires: 3, mean_injected_tokens: 42, false_fire_harm_count: 2 });
set("correction-plus-heuristics", "correction-follow-up", { pass_rate: 0.88, total_fires: 22, mean_injected_tokens: 110 });
set("correction-plus-heuristics", "continuation", { pass_rate: 0.70, total_fires: 8, mean_injected_tokens: 70 });
set("correction-plus-heuristics", "stale-memory-conflict", { pass_rate: 0.46, total_fires: 2, mean_injected_tokens: 60, false_fire_harm_count: 1 });

set("full-ocb", "all", { pass_rate: 0.80, total_fires: 31, mean_injected_tokens: 140, abstention_regret_count: 2, false_fire_harm_count: 2 });
set("full-ocb", "direct-answer", { pass_rate: 0.94, total_fires: 1, mean_injected_tokens: 180 });
set("full-ocb", "correction-follow-up", { pass_rate: 0.92, total_fires: 21, mean_injected_tokens: 150 });
set("full-ocb", "continuation", { pass_rate: 0.72, total_fires: 7, mean_injected_tokens: 120 });
set("full-ocb", "stale-memory-conflict", { pass_rate: 0.52, total_fires: 2, mean_injected_tokens: 110, false_fire_harm_count: 1 });

const fakeLedger = {
  aggregate({ backend, slice }: { run_id: string; backend: MemoryBackend; slice: TurnSlice | "all" }): AblationResult {
    return synthetic[`${backend}|${slice}`]!;
  },
} as unknown as Ledger;

const { html, json } = generateResults(fakeLedger, {
  run_id: "synthetic-demo",
  backends: ["none", "correction-only", "correction-plus-heuristics", "full-ocb"],
  slices: ["all", "direct-answer", "correction-follow-up", "continuation", "stale-memory-conflict"],
  title: "OpenClawBrain — Correction-persistence ablation results (synthetic demo)",
  notes: "This page was generated from synthetic numbers to preview the dashboard layout before a real ablation run.",
});

mkdirSync("results", { recursive: true });
writeFileSync("results/index.html", html);
writeFileSync("results/results.json", JSON.stringify(json, null, 2));
console.log("wrote results/index.html and results/results.json");
console.log("\nJSON snapshot preview:");
console.log(JSON.stringify(json, null, 2).split("\n").slice(0, 30).join("\n"));
