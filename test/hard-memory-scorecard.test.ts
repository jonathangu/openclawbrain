import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  HARD_MEMORY_SCORECARD_CONTRACT,
  HARD_MEMORY_SCORECARD_JSON_FILE,
  HARD_MEMORY_SCORECARD_MARKDOWN_FILE,
  buildHardMemoryScorecard,
  buildHardMemoryScorecardMarkdown,
  isStrictHardMemoryLabelRecord,
  writeHardMemoryScorecardOutputs,
  main,
} from "../scripts/hard-memory-scorecard.mjs";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function tempDir(prefix: string) {
  const dir = mkdtempSync(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function labelRecord(traceId: string, overrides: Record<string, unknown> = {}) {
  const overrideLabels = (overrides.labels as Record<string, unknown> | undefined) ?? {};
  const { labels: _ignored, ...rest } = overrides;
  return {
    schema_version: "learned-route-labels.v1",
    trace_id: traceId,
    annotator: "tester",
    labeled_at: "2026-04-15T20:00:00.000Z",
    labels: {
      memory_needed: "yes",
      wrapper_noise: "no",
      continuation_only: "no",
      operational_recovery: "no",
      human_semantic_task: "yes",
      oracle_best_mode: "learned_route",
      cost_sensitive: "medium",
      ...overrideLabels,
    },
    notes: {},
    ...rest,
  };
}

describe("hard-memory scorecard", () => {
  it("scores strict hard-memory labels and computes a pass gate when utility coverage is complete", () => {
    const seedManifest = {
      manifestId: "hard-memory-seed-5",
      traces: [
        { traceId: "trace-a" },
        { traceId: "trace-b" },
        { traceId: "trace-c" },
        { traceId: "trace-d" },
        { traceId: "trace-e" },
      ],
    };
    const labels = [
      labelRecord("trace-a", { labels: { oracle_best_mode: "learned_route" } }),
      labelRecord("trace-b", { labels: { oracle_best_mode: "learned_route" } }),
      labelRecord("trace-c", { labels: { oracle_best_mode: "tie" } }),
      labelRecord("trace-d", { labels: { oracle_best_mode: "graph_prior_only" } }),
      labelRecord("trace-e", { labels: { wrapper_noise: "yes", oracle_best_mode: "learned_route" } }),
    ];
    const replayLaneSummaryTables = {
      scorecard: {
        requiredContextRecall: {
          delta: 0.125,
        },
      },
      traces: [
        { traceId: "trace-a", candidateRelationVsBaseline: "tied" },
        { traceId: "trace-b", candidateRelationVsBaseline: "tied" },
        { traceId: "trace-c", candidateRelationVsBaseline: "tied" },
        { traceId: "trace-d", candidateRelationVsBaseline: "tied" },
      ],
    };
    const supplementalMetrics = [
      { traceId: "trace-a", netUtilityDelta: 0.7 },
      { traceId: "trace-b", netUtilityDelta: 0.5 },
      { traceId: "trace-c", netUtilityDelta: 0 },
      { traceId: "trace-d", netUtilityDelta: -0.1 },
    ];

    const scorecard = buildHardMemoryScorecard({
      seedManifest,
      labelRecords: labels,
      replayLaneSummaryTables,
      supplementalMetrics,
    });

    expect(scorecard.contract).toBe(HARD_MEMORY_SCORECARD_CONTRACT);
    expect(scorecard.focus_lane).toBe("hard_memory");
    expect(scorecard.focus_cohort_id).toBe("hard-memory-seed-5");
    expect(scorecard.trace_count).toBe(4);
    expect(scorecard.lr_vs_gpo_better).toBe(2);
    expect(scorecard.lr_vs_gpo_tied).toBe(1);
    expect(scorecard.lr_vs_gpo_worse).toBe(1);
    expect(scorecard.tie_or_better_rate).toBe(0.75);
    expect(scorecard.regression_rate).toBe(0.25);
    expect(scorecard.required_context_recall_delta).toBe(0.125);
    expect(scorecard.net_utility_delta).toBe(0.275);
    expect(scorecard.gate_status).toBe("pass");
    expect(scorecard.blockers).toEqual([]);
    expect(scorecard.coverage.strict_eligible_trace_count).toBe(4);
    expect(scorecard.coverage.numeric_utility_trace_count).toBe(4);
    expect(scorecard.traces.find((row: any) => row.trace_id === "trace-e")?.strict_hard_memory_eligible).toBe(false);

    const markdown = buildHardMemoryScorecardMarkdown(scorecard);
    expect(markdown).toContain("| 1 | `focus_lane` | Focus lane | `hard_memory` |");
    expect(markdown).toContain("| 10 | `net_utility_delta` | Net utility delta | 0.275 |");
    expect(markdown).toContain("trace-e");

    const outDir = tempDir("ocb-hard-memory-scorecard-");
    const outputs = writeHardMemoryScorecardOutputs(outDir, scorecard);
    const writtenJson = JSON.parse(readFileSync(outputs.jsonPath, "utf8"));
    const writtenMarkdown = readFileSync(outputs.markdownPath, "utf8");
    expect(path.basename(outputs.jsonPath)).toBe(HARD_MEMORY_SCORECARD_JSON_FILE);
    expect(path.basename(outputs.markdownPath)).toBe(HARD_MEMORY_SCORECARD_MARKDOWN_FILE);
    expect(writtenJson.gate_status).toBe("pass");
    expect(writtenMarkdown).toContain("## Lead block");
  });

  it("loads real seed-manifest entries plus JSONL labels and supplemental utility rows", () => {
    const root = tempDir("ocb-hard-memory-cli-");
    const labelsDir = path.join(root, "labels");
    const outDir = path.join(root, "out");
    const seedManifestPath = path.join(root, "hard-memory-seed.json");
    const replaySummaryPath = path.join(root, "summary-tables.json");
    const supplementalMetricsPath = path.join(root, "utility-deltas.jsonl");
    const labelsPath = path.join(labelsDir, "reviewed.jsonl");

    mkdirSync(labelsDir, { recursive: true });

    writeFileSync(seedManifestPath, `${JSON.stringify({
      setId: "hard-memory-reviewed-2",
      entries: [
        { traceId: "trace-a" },
        { traceId: "trace-b" },
      ],
    }, null, 2)}\n`, "utf8");
    writeFileSync(replaySummaryPath, `${JSON.stringify({
      scorecard: {
        requiredContextRecall: {
          delta: 0.2,
        },
      },
    }, null, 2)}\n`, "utf8");
    writeFileSync(labelsPath, [
      JSON.stringify(labelRecord("trace-a", { labels: { oracle_best_mode: "learned_route" } })),
      JSON.stringify(labelRecord("trace-b", { labels: { oracle_best_mode: "tie" } })),
    ].join("\n") + "\n", "utf8");
    writeFileSync(supplementalMetricsPath, [
      JSON.stringify({ traceId: "trace-a", utility_delta_vs_graph_prior_only: 0.6 }),
      JSON.stringify({ trace_id: "trace-b", netUtilityDelta: 0.2 }),
    ].join("\n") + "\n", "utf8");

    main([
      "--seed-manifest", seedManifestPath,
      "--labels-dir", labelsDir,
      "--replay-summary-tables", replaySummaryPath,
      "--supplemental-metrics", supplementalMetricsPath,
      "--out-dir", outDir,
    ]);

    const writtenJson = JSON.parse(readFileSync(path.join(outDir, HARD_MEMORY_SCORECARD_JSON_FILE), "utf8"));
    expect(writtenJson.focus_cohort_id).toBe("hard-memory-reviewed-2");
    expect(writtenJson.trace_count).toBe(2);
    expect(writtenJson.lr_vs_gpo_better).toBe(1);
    expect(writtenJson.lr_vs_gpo_tied).toBe(1);
    expect(writtenJson.required_context_recall_delta).toBe(0.2);
    expect(writtenJson.net_utility_delta).toBe(0.4);
    expect(writtenJson.gate_status).toBe("pass");
  });

  it("can source per-trace utility deltas directly from reviewed trace records when present", () => {
    const scorecard = buildHardMemoryScorecard({
      focusCohortId: "hard-memory-reviewed-inline-2",
      seedManifest: {
        entries: [
          { traceId: "trace-a" },
          { traceId: "trace-b" },
        ],
      },
      labelRecords: [
        {
          ...labelRecord("trace-a", { labels: { oracle_best_mode: "learned_route" } }),
          utility_delta_vs_graph_prior_only: 0.9,
        },
        {
          ...labelRecord("trace-b", { labels: { oracle_best_mode: "graph_prior_only" } }),
          route_objective_hint: {
            utility_delta_vs_graph_prior_only: -0.3,
          },
        },
      ],
    });

    expect(scorecard.trace_count).toBe(2);
    expect(scorecard.coverage.numeric_utility_trace_count).toBe(2);
    expect(scorecard.net_utility_delta).toBe(0.3);
    expect(scorecard.gate_status).toBe("watch");
  });

  it("stays scaffold-only when strict labels or utility deltas are missing", () => {
    const labels = [
      labelRecord("trace-a", { labels: { memory_needed: "unclear", oracle_best_mode: "unclear" } }),
      labelRecord("trace-b", { labels: { continuation_only: "yes" } }),
    ];

    expect(isStrictHardMemoryLabelRecord(labels[0])).toBe(false);
    expect(isStrictHardMemoryLabelRecord(labels[1])).toBe(false);

    const scorecard = buildHardMemoryScorecard({
      focusCohortId: "hard-memory-seed-placeholder",
      seedManifest: { traces: [{ traceId: "trace-a" }, { traceId: "trace-b" }, { traceId: "trace-c" }] },
      labelRecords: labels,
    });

    expect(scorecard.trace_count).toBe(0);
    expect(scorecard.net_utility_delta).toBeNull();
    expect(scorecard.gate_status).toBeNull();
    expect(scorecard.blockers).toEqual(expect.arrayContaining([
      "no_strict_hard_memory_labels_scored",
    ]));
    expect(scorecard.notes).toEqual(expect.arrayContaining([
      expect.stringContaining("net utility delta stays unavailable"),
    ]));
  });
});
