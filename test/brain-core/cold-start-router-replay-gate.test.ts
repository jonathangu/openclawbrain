import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, afterEach } from "vitest";
import { compileColdStartDocsQaSourceBundleFromFileV1 } from "../../src/brain-core/cold-start-data-compiler.js";
import {
  replayColdStartRouterArtifactV1,
} from "../../src/brain-core/cold-start-router-replay-gate.js";
import {
  trainColdStartRouterArtifactV1,
} from "../../src/brain-core/cold-start-router-trainer.js";
import type { DataRegistryEntryV1, RouteDecisionRowV1 } from "../../src/brain-core/cold-start-router-contracts.js";
import {
  POLICY_SUPERVISION_ROW_CONTRACT_V1,
  POLICY_SUPERVISION_ROW_VERSION_V1,
} from "../../src/brain-core/policy-supervision-rows.js";
import type { PolicySupervisionRowV1 } from "../../src/brain-core/policy-supervision-rows.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const sampleBundlePath = path.join(repoRoot, "artifacts", "cold-start-router-sample", "docs-qa-sample.raw.json");

const tempRoots: string[] = [];

afterEach(() => {
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
});

function createTempRoot(label: string): string {
  const root = mkdtempSync(path.join(os.tmpdir(), `${label}-`));
  tempRoots.push(root);
  return root;
}

function makeSmokeRegistryEntry(): DataRegistryEntryV1 {
  return {
    dataset_id: "dataset_cold_start_smoke",
    source_family: "qa",
    upstream_url: "https://example.org/cold-start-smoke",
    original_creator: "OpenClaw",
    license: "CC BY 4.0",
    commercial_use_status: "allowed",
    redistribution_status: "allowed",
    pii_risk: "none",
    benchmark_split_status: "train",
    approval_status: "approved_train",
    reviewer: "operator",
    immutable_snapshot_ref: "snapshot:cold-start-smoke@sha256:aaa111",
    exact_files: ["data/train.json"],
    file_hashes: {
      "data/train.json": "sha256:bbb222",
    },
    allowed_uses: ["route supervision", "ranking baselines"],
    disallowed_uses: ["private redistribution"],
    notes: ["fixture"],
    created_at: "2026-04-05T16:00:00Z",
    updated_at: "2026-04-05T16:00:00Z",
  };
}

function makeSmokeRouteRows(): RouteDecisionRowV1[] {
  const datasetId = "dataset_cold_start_smoke";
  return [
    {
      row_id: "row_route_001",
      dataset_id: datasetId,
      query: "Find the shipping correction memory and update the invoice draft",
      cursor_path: ["thread:billing", "mem:user_profile"],
      candidate_set: [
        { candidate_id: "mem:shipping_history", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.92 },
        { candidate_id: "doc:invoice_policy", candidate_type: "doc_chunk", authority: "operator_policy", freshness: "stale", token_cost: 28, score_hint: 0.18 },
        { candidate_id: "tool:gmail_search", candidate_type: "tool", authority: "runtime", freshness: "current", token_cost: 4, score_hint: 0.51 },
      ],
      teacher_action: { kind: "traverse", target_ids: ["mem:shipping_history"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "message_183", start: 21, end: 72 },
        { source_ref: "memory_entry_77", start: 0, end: 44 },
      ],
      hard_negatives: ["doc:invoice_policy"],
      outcome_gain: 0.84,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: "snapshot:cold-start-smoke@sha256:aaa111",
        recorded_by: "operator",
        recorded_at: "2026-04-05T16:01:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-05T16:02:00Z",
    },
    {
      row_id: "row_route_002",
      dataset_id: datasetId,
      query: "Stop once the invoice tool is enough",
      cursor_path: ["thread:billing"],
      candidate_set: [
        { candidate_id: "tool:invoice_draft", candidate_type: "tool", authority: "runtime", freshness: "current", token_cost: 2, score_hint: 0.87 },
        { candidate_id: "mem:shipping_history", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.15 },
      ],
      teacher_action: { kind: "tool", tool_name: "tool:invoice_draft" },
      stop_label: "STOP_LOCAL",
      evidence_spans: [
        { source_ref: "message_184", start: 0, end: 30 },
      ],
      hard_negatives: [],
      outcome_gain: 0.42,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: "snapshot:cold-start-smoke@sha256:aaa111",
        recorded_by: "operator",
        recorded_at: "2026-04-05T16:03:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-05T16:04:00Z",
    },
  ];
}

function makeActivationFirstRegistryEntry(datasetId: string): DataRegistryEntryV1 {
  return {
    dataset_id: datasetId,
    source_family: "qa",
    upstream_url: "https://example.org/activation-first",
    original_creator: "OpenClaw",
    license: "CC BY 4.0",
    commercial_use_status: "allowed",
    redistribution_status: "allowed",
    pii_risk: "none",
    benchmark_split_status: "train",
    approval_status: "approved_train",
    reviewer: "operator",
    immutable_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
    exact_files: ["data/train.json"],
    file_hashes: {
      "data/train.json": "sha256:activation-first-train",
    },
    allowed_uses: ["route supervision", "ranking baselines"],
    disallowed_uses: ["private redistribution"],
    notes: ["activation-first replay fixture"],
    created_at: "2026-04-10T08:00:00Z",
    updated_at: "2026-04-10T08:00:00Z",
  };
}

function makeActivationFirstTrainingRows(datasetId: string): RouteDecisionRowV1[] {
  return [
    {
      row_id: "row_activation_first_continue",
      dataset_id: datasetId,
      query: "Find the current shipping correction memory before drafting the reply",
      cursor_path: ["source:billing-thread"],
      candidate_set: [
        { candidate_id: "mem:shipping_correction", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.88 },
        { candidate_id: "doc:generic_shipping_policy", candidate_type: "doc_chunk", authority: "operator_policy", freshness: "stale", token_cost: 18, score_hint: 0.22 },
      ],
      teacher_action: { kind: "traverse", target_ids: ["mem:shipping_correction"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "message_201", start: 0, end: 32 },
        { source_ref: "memory_201", start: 0, end: 28 },
      ],
      hard_negatives: ["doc:generic_shipping_policy"],
      outcome_gain: 0.8,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
        recorded_by: "operator",
        recorded_at: "2026-04-10T08:01:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-10T08:02:00Z",
    },
    {
      row_id: "row_activation_first_stop_local",
      dataset_id: datasetId,
      query: "Use the invoice tool and stop locally",
      cursor_path: ["source:billing-thread"],
      candidate_set: [
        { candidate_id: "tool:invoice_draft", candidate_type: "tool", authority: "runtime", freshness: "current", token_cost: 2, score_hint: 0.87 },
        { candidate_id: "mem:shipping_correction", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.15 },
      ],
      teacher_action: { kind: "tool", tool_name: "tool:invoice_draft" },
      stop_label: "STOP_LOCAL",
      evidence_spans: [
        { source_ref: "message_202", start: 0, end: 28 },
      ],
      hard_negatives: [],
      outcome_gain: 0.8,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
        recorded_by: "operator",
        recorded_at: "2026-04-10T08:03:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-10T08:04:00Z",
    },
  ];
}

function makeGreedyThresholdRouteRow(datasetId: string): RouteDecisionRowV1 {
  return {
    row_id: "row_activation_first_greedy_lane",
    dataset_id: datasetId,
    query: "Need the exact shipping correction memory, not the generic policy page",
    cursor_path: ["source:billing-thread"],
    candidate_set: [
      { candidate_id: "mem:shipping_correction", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.28 },
      { candidate_id: "doc:generic_shipping_policy", candidate_type: "doc_chunk", authority: "operator_policy", freshness: "current", score_hint: 0.18 },
    ],
    teacher_action: { kind: "traverse", target_ids: ["mem:shipping_correction"] },
    stop_label: "CONTINUE",
    evidence_spans: [
      { source_ref: "message_203", start: 0, end: 30 },
      { source_ref: "memory_203", start: 0, end: 24 },
    ],
    hard_negatives: ["doc:generic_shipping_policy"],
    outcome_gain: 0.72,
    provenance: {
      dataset: datasetId,
      source_license: "CC BY 4.0",
      source_family: "qa",
      source_snapshot_ref: `snapshot:${datasetId}@sha256:activation-first`,
      recorded_by: "operator",
      recorded_at: "2026-04-10T08:05:00Z",
      review_status: "approved_train",
    },
    split_tag: "train",
    created_at: "2026-04-10T08:06:00Z",
  };
}

function makePolicySupervisionRowFixture(params: {
  rowId: string;
  traceId: string;
  routeRowId: string;
  rowType: PolicySupervisionRowV1["row_type"];
  focusLane: string | null;
  rowWeight: number;
  hardNegativeClass?: PolicySupervisionRowV1["hard_negative_class"];
  oracleBestMode?: PolicySupervisionRowV1["oracle_best_mode"];
}): PolicySupervisionRowV1 {
  return {
    schema_version: POLICY_SUPERVISION_ROW_VERSION_V1,
    contract: POLICY_SUPERVISION_ROW_CONTRACT_V1,
    row_id: params.rowId,
    trace_id: params.traceId,
    episode_id: null,
    decision_point_id: null,
    row_type: params.rowType,
    focus_lane: params.focusLane,
    trace_slice: {
      route_row_id: params.routeRowId,
      route_fn_version: null,
      chosen_action_kind: null,
      stop_label: null,
      query_text_hash: null,
    },
    row_weight: params.rowWeight,
    confidence_target: null,
    hard_negative_class: params.hardNegativeClass ?? null,
    net_utility_delta: null,
    net_utility_delta_source: null,
    projection_status: "owner_labeled",
    oracle_best_mode: params.oracleBestMode ?? null,
    notes: ["replay-gate fixture"],
  };
}

function trainActivationFirstReplayFixture(params: {
  outputDir: string;
  datasetId: string;
  policySupervisionRows?: PolicySupervisionRowV1[];
  focusLaneWeights?: Record<string, number>;
  rowTypeWeights?: Partial<Record<PolicySupervisionRowV1["row_type"], number>>;
}) {
  return trainColdStartRouterArtifactV1({
    artifactId: `router-artifact-${params.datasetId}`,
    artifactVersion: "0.0.1",
    packType: "base",
    compatibleRuntimeVersion: "openclawbrain-runtime@0.4.44",
    registryEntries: [makeActivationFirstRegistryEntry(params.datasetId)],
    routeRows: makeActivationFirstTrainingRows(params.datasetId),
    outputDir: params.outputDir,
    routerIdentity: `router:${params.datasetId}`,
    createdAt: "2026-04-10T08:10:00Z",
    trainingDataRefs: [params.datasetId],
    replayGateRefs: [`replay:${params.datasetId}`],
    ...(params.policySupervisionRows ? { policySupervisionRows: params.policySupervisionRows } : {}),
    ...(params.focusLaneWeights ? { focusLaneWeights: params.focusLaneWeights } : {}),
    ...(params.rowTypeWeights ? { rowTypeWeights: params.rowTypeWeights } : {}),
  });
}

describe("cold-start router replay gate", () => {
  it("loads a produced artifact and passes on the curated docs/QA replay rows", () => {
    const bundle = compileColdStartDocsQaSourceBundleFromFileV1({ bundlePath: sampleBundlePath, repoRoot });
    const outputDir = createTempRoot("cold-start-router-replay-gate-good");

    trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-replay-good",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.44",
      registryEntries: [bundle.registryEntry],
      routeRows: bundle.routeRows,
      outputDir,
      routerIdentity: "router:docs-qa:replay",
      createdAt: "2026-04-05T17:35:00Z",
      trainingDataRefs: [bundle.registryEntry.dataset_id],
      replayGateRefs: ["replay:docs-qa-sample:tiny-gate"],
    });

    const verdict = replayColdStartRouterArtifactV1({
      artifactDir: outputDir,
      routeRows: bundle.routeRows,
    });

    expect(verdict).toMatchObject({
      passed: true,
      verdict: "pass",
      evaluatedRowCount: 2,
      passedRowCount: 2,
      failedRowCount: 0,
      skippedRowCount: 0,
    });
    expect(verdict.manifestSummary).toMatchObject({
      artifactId: "router-artifact-replay-good",
      packType: "base",
      trainingDataRefCount: 1,
      replayGateRefCount: 1,
      runtimeVersion: "openclawbrain-runtime@0.4.44",
    });
    expect(verdict.rowResults).toHaveLength(2);
    expect(verdict.rowResults[0]).toMatchObject({
      rowId: "docs-qa-routing-prior-conflict",
      passed: true,
      expectedTopCandidateId: "doc:routing-prior:conflict-resolution",
      actualTopCandidateId: "doc:routing-prior:conflict-resolution",
      expectedStopLabel: "CONTINUE",
      actualStopLabel: "CONTINUE",
    });
    expect(verdict.rowResults[1]).toMatchObject({
      rowId: "docs-qa-ci-first-deterministic-lints",
      passed: true,
      expectedTopCandidateId: "doc:teacher-v3-lints:ci-first",
      actualTopCandidateId: "doc:teacher-v3-lints:ci-first",
      expectedStopLabel: "CONTINUE",
      actualStopLabel: "CONTINUE",
    });
  });

  it("fails replay when the artifact manifest is obviously corrupted", () => {
    const bundle = compileColdStartDocsQaSourceBundleFromFileV1({ bundlePath: sampleBundlePath, repoRoot });
    const outputDir = createTempRoot("cold-start-router-replay-gate-bad");

    trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-replay-bad",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.4.44",
      registryEntries: [makeSmokeRegistryEntry()],
      routeRows: makeSmokeRouteRows(),
      outputDir,
      routerIdentity: "router:smoke:replay",
      createdAt: "2026-04-05T17:36:00Z",
      trainingDataRefs: ["dataset_cold_start_smoke"],
      replayGateRefs: ["replay:docs-qa-sample:tiny-gate"],
    });

    const manifestPath = path.join(outputDir, "manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
    manifest.weights_ref = "weights.json#sha256:deadbeef";
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

    const verdict = replayColdStartRouterArtifactV1({
      artifactDir: outputDir,
      routeRows: bundle.routeRows,
    });

    expect(verdict.passed).toBe(false);
    expect(verdict.verdict).toBe("fail");
    expect(verdict.loadIssues.length).toBeGreaterThan(0);
    expect(verdict.loadIssues.some((issue) => issue.code === "manifest_ref_mismatch" || issue.code === "manifest_checksum_mismatch")).toBe(true);
  });

  it("fails on must-not-fire abstention supervision even when route-row diagnostics still pass", () => {
    const datasetId = "dataset_replay_gate_policy_abstain";
    const outputDir = createTempRoot("cold-start-router-replay-gate-policy-abstain");
    trainActivationFirstReplayFixture({
      outputDir,
      datasetId,
    });

    const replayRow = makeGreedyThresholdRouteRow(datasetId);
    const verdict = replayColdStartRouterArtifactV1({
      artifactDir: outputDir,
      routeRows: [replayRow],
      policySupervisionRows: [
        makePolicySupervisionRowFixture({
          rowId: "ps_replay_gate_must_not_fire",
          traceId: "trace_replay_gate_must_not_fire",
          routeRowId: replayRow.row_id,
          rowType: "abstain",
          focusLane: "must_not_fire_100",
          rowWeight: 2,
          hardNegativeClass: "unnecessary_activation",
          oracleBestMode: "graph_prior_only",
        }),
      ],
    });

    expect(verdict).toMatchObject({
      passed: false,
      verdict: "fail",
      evaluatedRowCount: 1,
      passedRowCount: 0,
      failedRowCount: 1,
      skippedRowCount: 0,
      policyExpectationCount: 1,
      passedPolicyExpectationCount: 0,
      failedPolicyExpectationCount: 1,
    });
    expect(verdict.summary).toContain("policy activation matches 0/1");
    expect(verdict.summary).toContain("abstain matches 0/1");
    expect(verdict.laneSummaries).toEqual([
      expect.objectContaining({
        lane: "must_not_fire_100",
        policyExpectationCount: 1,
        failedPolicyExpectationCount: 1,
        abstainExpectationCount: 1,
        abstainMatchCount: 0,
      }),
    ]);
    expect(verdict.rowResults).toHaveLength(1);
    expect(verdict.rowResults[0]).toMatchObject({
      rowId: replayRow.row_id,
      routeRowDiagnosticPassed: true,
      actualActivated: true,
      actualAbstained: false,
      policyExpectationCount: 1,
      policyExpectationPassCount: 0,
      passed: false,
    });
    expect(verdict.rowResults[0].policyExpectationResults).toEqual([
      expect.objectContaining({
        policyRowId: "ps_replay_gate_must_not_fire",
        focusLane: "must_not_fire_100",
        expectedActivated: false,
        actualActivated: true,
        expectedAbstained: true,
        actualAbstained: false,
        expectedStopLocal: null,
        passed: false,
      }),
    ]);
  });

  it("fails on felt-lane activation supervision when the replayed row still abstains", () => {
    const datasetId = "dataset_replay_gate_policy_activate";
    const outputDir = createTempRoot("cold-start-router-replay-gate-policy-activate");
    trainActivationFirstReplayFixture({
      outputDir,
      datasetId,
      policySupervisionRows: [
        makePolicySupervisionRowFixture({
          rowId: "ps_training_must_not_fire",
          traceId: "trace_training_must_not_fire",
          routeRowId: "row_activation_first_continue",
          rowType: "abstain",
          focusLane: "must_not_fire_100",
          rowWeight: 2,
          hardNegativeClass: "unnecessary_activation",
          oracleBestMode: "graph_prior_only",
        }),
      ],
      focusLaneWeights: { must_not_fire_100: 8 },
      rowTypeWeights: { abstain: 2 },
    });

    const replayRow = makeGreedyThresholdRouteRow(datasetId);
    const verdict = replayColdStartRouterArtifactV1({
      artifactDir: outputDir,
      routeRows: [replayRow],
      policySupervisionRows: [
        makePolicySupervisionRowFixture({
          rowId: "ps_replay_gate_felt_resume",
          traceId: "trace_replay_gate_felt_resume",
          routeRowId: replayRow.row_id,
          rowType: "activate",
          focusLane: "felt_resume_25",
          rowWeight: 1.8,
          oracleBestMode: "learned_route",
        }),
      ],
    });

    expect(verdict).toMatchObject({
      passed: false,
      verdict: "fail",
      evaluatedRowCount: 1,
      passedRowCount: 0,
      failedRowCount: 1,
      skippedRowCount: 0,
      policyExpectationCount: 1,
      passedPolicyExpectationCount: 0,
      failedPolicyExpectationCount: 1,
    });
    expect(verdict.laneSummaries).toEqual([
      expect.objectContaining({
        lane: "felt_resume_25",
        policyExpectationCount: 1,
        failedPolicyExpectationCount: 1,
        activationExpectationCount: 1,
        activationMatchCount: 0,
      }),
    ]);
    expect(verdict.rowResults).toHaveLength(1);
    expect(verdict.rowResults[0]).toMatchObject({
      rowId: replayRow.row_id,
      routeRowDiagnosticPassed: false,
      actualActivated: false,
      actualAbstained: true,
      policyExpectationCount: 1,
      policyExpectationPassCount: 0,
      passed: false,
    });
    expect(verdict.rowResults[0].policyExpectationResults).toEqual([
      expect.objectContaining({
        policyRowId: "ps_replay_gate_felt_resume",
        focusLane: "felt_resume_25",
        expectedActivated: true,
        actualActivated: false,
        expectedAbstained: false,
        actualAbstained: true,
        expectedStopLocal: null,
        passed: false,
      }),
    ]);
  });

  it("reports policy-led summary text even when route-row activation diagnostics disagree", () => {
    const datasetId = "dataset_replay_gate_policy_summary";
    const outputDir = createTempRoot("cold-start-router-replay-gate-policy-summary");
    trainActivationFirstReplayFixture({
      outputDir,
      datasetId,
    });

    const replayRow = {
      ...makeGreedyThresholdRouteRow(datasetId),
      teacher_action: { kind: "tool" as const, tool_name: "__policy_summary_probe__" },
    };
    const verdict = replayColdStartRouterArtifactV1({
      artifactDir: outputDir,
      routeRows: [replayRow],
      policySupervisionRows: [
        makePolicySupervisionRowFixture({
          rowId: "ps_replay_gate_policy_summary",
          traceId: "trace_replay_gate_policy_summary",
          routeRowId: replayRow.row_id,
          rowType: "activate",
          focusLane: "felt_resume_25",
          rowWeight: 1.5,
          oracleBestMode: "learned_route",
        }),
      ],
    });

    expect(verdict).toMatchObject({
      passed: true,
      verdict: "pass",
      policyExpectationCount: 1,
      passedPolicyExpectationCount: 1,
      failedPolicyExpectationCount: 0,
    });
    expect(verdict.rowResults[0]).toMatchObject({
      routeRowDiagnosticPassed: false,
      policyExpectationPassCount: 1,
      passed: true,
      actualActivated: true,
    });
    expect(verdict.summary).toContain("policy activation matches 1/1");
    expect(verdict.summary).toContain("abstain matches 1/1");
    expect(verdict.summary).not.toContain("activation matches 0/1");
  });

  it("runs the smoke script end to end", () => {
    const stdout = execFileSync(
      "node",
      [
        "--experimental-transform-types",
        "scripts/replay-cold-start-router-smoke.ts",
      ],
      {
        cwd: repoRoot,
        encoding: "utf8",
      },
    );

    expect(stdout).toContain("Cold-start router replay smoke: ok");
    expect(stdout).toContain("verdict: pass");
    expect(stdout).toContain("passedRows: 2/2");
    expect(stdout).toContain("manifestChecksum:");
  });
});
