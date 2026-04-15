import path from "node:path";
import os from "node:os";
import { mkdtempSync, rmSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Value } from "@sinclair/typebox/value";
import { buildGraphifyCompiledArtifactPack, writeGraphifyCompiledArtifactPack } from "../../packages/cli/dist/src/graphify-compiled-artifacts.js";
import { exportGraphifyImportSlice } from "../../packages/cli/dist/src/import-export.js";
import {
  GraphifyTrainingReviewBundleSchemaV1,
  materializeGraphifyTrainingReviewBundleV1,
  summarizeGraphifyTrainingReviewBundleV1,
  validateGraphifyTrainingReviewBundleV1,
} from "../../src/brain-core/graphify-training-bridge.js";
import {
  RouteDecisionRowSchemaV1,
  validateRouteDecisionRowV1,
} from "../../src/brain-core/route-rows.js";

function makeRouteRowFixture() {
  return {
    schema_version: 1 as const,
    row_id: "rr_graphify_training_fixture_01",
    trace_id: "trace_graphify_training_fixture_01",
    episode_id: "episode_graphify_training_fixture_01",
    conversation_id: 42,
    route_fn_version: "brain-graph-traverse.v2",
    decision_point_id: "dp_graphify_training_fixture_01",
    decision_point_kind: "local" as const,
    source_node_id: "source_node_graphify_training_fixture_01",
    expansion_index: 0,
    selection_index: 0,
    query_text: "find the best tool and keep the path reviewable",
    cursor_path: ["root", "tools"],
    local_action_set: [
      {
        action_id: "action_traverse_doc_1",
        action_kind: "traverse" as const,
        node_id: "doc_1",
        tool_name: null,
        tool_capability_id: null,
        tool_instance_id: null,
        tool_args_shape: null,
        prior_score: 0.6,
        probability: 0.68,
        retrieval_features: {
          source: "graphify_import_slice",
          subject: "topic:compiled-artifacts",
        },
      },
      {
        action_id: "action_stop_local",
        action_kind: "stop_local" as const,
        node_id: null,
        tool_name: null,
        tool_capability_id: null,
        tool_instance_id: null,
        tool_args_shape: null,
        prior_score: 0.15,
        probability: 0.32,
        retrieval_features: null,
      },
    ],
    chosen_action_id: "action_traverse_doc_1",
    chosen_action_kind: "traverse" as const,
    chosen_node_id: "doc_1",
    chosen_tool_name: null,
    chosen_tool_capability_id: null,
    chosen_tool_instance_id: null,
    chosen_action_probability: 0.68,
    stop_probability: 0.32,
    stop_label: "CONTINUE" as const,
    budget_context: {
      budget_remaining: 84,
      initial_budget: 100,
      reserved_token_cost: 2,
      budget_used: 16,
      budget_used_fraction: 0.16,
      max_hops: 4,
      max_frontier_size: 8,
      frontier_size: 2,
      visited_count: 1,
      fired_count: 1,
      pending_selection_count: 1,
      pressure_level: 0.1,
      frontier_pressure: 0.15,
      budget_pressure: 0.1,
      budget_fraction: 0.16,
      query_budget_chars: 256,
      max_context_chars: 128,
      injected_chars: 80,
      dropped_chars: 0,
      context_clipped: false,
      route_selection_ms: 3,
      total_query_ms: 8,
      compile_deadline_ms: null,
      compile_deadline_hit: null,
    },
    route_context: {
      request_digest: "req_graphify_training_fixture_01",
      active_pack_id: "pack_graphify_training_fixture_01",
      router_identity: "brain-graph-traverse.v2",
      candidate_node_ids: ["doc_1", "tool_1"],
      selected_node_ids: ["doc_1"],
      selected_traversal_node_ids: ["doc_1"],
      selected_path_node_ids: ["doc_1"],
      selected_seed_node_ids: [],
    },
    label_provenance: {
      provenance_id: "lp_graphify_training_fixture_01",
      state: "matched" as const,
      basis: "trace_id" as const,
      confidence: 0.98,
      detail: "trace row is matched to the review fixture",
      source: "teacher" as const,
      kind: "teacher_review" as const,
      binding_mode: "trace_id" as const,
      attribution_quality: "exact" as const,
      feedback_richness: "followup_only" as const,
      trace_id: "trace_graphify_training_fixture_01",
      episode_id: "episode_graphify_training_fixture_01",
      decision_point_id: "dp_graphify_training_fixture_01",
      observation_id: null,
      supervision_id: null,
      update_id: null,
      candidate_ids: ["doc_1", "tool_1"],
      provenance_ref: "prov_lp_graphify_training_fixture_01",
      content_hash: "sha256:graphify-training-fixture-content",
      lineage_hash: "sha256:graphify-training-fixture-lineage",
      created_at: 1_746_000_000_000,
    },
    evidence_spans: [
      {
        source_ref: "req_graphify_training_fixture_01",
        start: 0,
        end: 18,
        excerpt: "find the best tool",
      },
    ],
    hard_negatives: ["tool_1", "doc_2"],
    created_at: "2026-04-07T12:00:00.000Z",
  };
}

describe("graphify training bridge", () => {
  it("attaches EXTRACTED neighborhood context and review-only provenance gaps to route-row training inputs", () => {
    const routeRow = makeRouteRowFixture();
    expect(Value.Check(RouteDecisionRowSchemaV1, routeRow)).toBe(true);
    expect(validateRouteDecisionRowV1(routeRow)).toMatchObject({ valid: true });

    const tempRoot = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-training-"));
    try {
      const fixturePackRoot = path.join(tempRoot, "graphify-pack");
      const packBundle = buildGraphifyCompiledArtifactPack({
        bundleId: "graphify-training-review-fixture",
        bundleStartedAt: "2026-04-06T22:30:00.000Z",
        outputDir: fixturePackRoot,
        graphifyRunId: "graphify-run-training-review-fixture",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyCommand: "graphify compile compiled-artifacts --fixture",
        sourceBundleId: "compiled-artifacts-target-state-scaffold",
        sourceBundleHash: "sha256:source-bundle-hash",
        graphHash: "sha256:graph-hash",
        configHash: "sha256:config-hash",
        labelsHash: "sha256:labels-hash",
      });
      writeGraphifyCompiledArtifactPack(packBundle.outputDir, packBundle);

      const importSliceResult = exportGraphifyImportSlice({
        bundleRoot: packBundle.outputDir,
        outputRoot: path.join(tempRoot, "artifacts", "graphify-imports"),
        runId: "graphify-training-review-fixture",
      });

      expect(importSliceResult.ok).toBe(true);
      expect(importSliceResult.candidatePackInput?.importedPriors?.neighborhoodPriors?.length ?? 0).toBeGreaterThan(0);

      const bundle = materializeGraphifyTrainingReviewBundleV1({
        routeRows: [routeRow],
        graphifyImportSlice: importSliceResult,
        compiledArtifactPackRoot: packBundle.outputDir,
        generatedAt: "2026-04-07T12:00:00.000Z",
        maxRows: 8,
        maxProvenanceGapHints: 4,
        maxHardNegativeSupportsPerRow: 4,
      });

      expect(Value.Check(GraphifyTrainingReviewBundleSchemaV1, bundle)).toBe(true);
      expect(validateGraphifyTrainingReviewBundleV1(bundle)).toMatchObject({ valid: true });
      expect(bundle.review_boundary.live_eligible).toBe(false);
      expect(bundle.review_boundary.review_only_trust_classes).toEqual(["INFERRED", "AMBIGUOUS"]);
      expect(bundle.review_boundary.live_eligible_trust_classes).toEqual(["EXTRACTED"]);
      expect(bundle.route_row_ids).toEqual([routeRow.row_id]);
      expect(bundle.row_count).toBe(1);
      expect(bundle.included_row_count).toBe(1);
      expect(bundle.truncated).toBe(false);
      expect(bundle.neighborhood_context_count).toBeGreaterThan(0);
      expect(bundle.provenance_gap_hint_count).toBeGreaterThan(0);
      expect(bundle.hard_negative_support_count).toBe(2);

      const row = bundle.rows[0];
      expect(row).toBeDefined();
      if (!row) {
        throw new Error("expected a training review row");
      }

      expect(row.route_row_summary.rowId).toBe(routeRow.row_id);
      expect(row.route_row_core.row_id).toBe(routeRow.row_id);
      expect(row.route_objective_hint).toBeNull();
      expect(row.graphify_neighborhood_context.trust_class).toBe("EXTRACTED");
      expect(row.graphify_neighborhood_context.review_boundary.live_eligible).toBe(false);
      expect(row.graphify_neighborhood_context.neighborhoods[0]?.source_artifact_kind).toBe("neighborhood_summary");
      expect(row.graphify_neighborhood_context.neighborhoods[0]?.subject_ids.length ?? 0).toBeGreaterThan(0);
      expect(row.hard_negative_supports).toHaveLength(routeRow.hard_negatives.length);
      expect(row.hard_negative_supports[0]?.summary).toContain("review-only");
      expect(row.hard_negative_supports[0]?.trust_class).toBe("EXTRACTED");
      expect(row.provenance_gap_hint_ids.length).toBeGreaterThan(0);
      expect(row.review_notes[0]).toContain("canonical route row");

      expect(bundle.provenance_gap_hints[0]?.trust_class).toBe("AMBIGUOUS");
      expect(bundle.provenance_gap_hints[0]?.summary).toMatch(/hash verification|manifest generation|replayable proposal wiring/i);
      expect(bundle.provenance_gap_hints[0]?.route_row_ids).toEqual([routeRow.row_id]);

      const summary = summarizeGraphifyTrainingReviewBundleV1(bundle);
      expect(summary).toMatchObject({
        rowCount: 1,
        includedRowCount: 1,
        truncated: false,
        neighborhoodContextCount: bundle.neighborhood_context_count,
        provenanceGapHintCount: bundle.provenance_gap_hint_count,
        hardNegativeSupportCount: 2,
        routeRowIds: [routeRow.row_id],
      });
    }
    finally {
      rmSync(tempRoot, { recursive: true, force: true });
    }
  });

  it("attaches trace-level route-objective hints for stop-aware utility scaffolding without pretending row-level teacher truth", () => {
    const routeRow = makeRouteRowFixture();
    const tempRoot = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-objective-"));
    try {
      const fixturePackRoot = path.join(tempRoot, "graphify-pack");
      const packBundle = buildGraphifyCompiledArtifactPack({
        bundleId: "graphify-training-objective-fixture",
        bundleStartedAt: "2026-04-06T22:30:00.000Z",
        outputDir: fixturePackRoot,
        graphifyRunId: "graphify-run-training-objective-fixture",
        graphifyVersion: "graphify-test@1.2.3",
        graphifyCommand: "graphify compile compiled-artifacts --fixture",
        sourceBundleId: "compiled-artifacts-target-state-scaffold",
        sourceBundleHash: "sha256:source-bundle-hash",
        graphHash: "sha256:graph-hash",
        configHash: "sha256:config-hash",
        labelsHash: "sha256:labels-hash",
      });
      writeGraphifyCompiledArtifactPack(packBundle.outputDir, packBundle);

      const importSliceResult = exportGraphifyImportSlice({
        bundleRoot: packBundle.outputDir,
        outputRoot: path.join(tempRoot, "artifacts", "graphify-imports"),
        runId: "graphify-training-objective-fixture",
      });

      const bundle = materializeGraphifyTrainingReviewBundleV1({
        routeRows: [routeRow],
        graphifyImportSlice: importSliceResult,
        compiledArtifactPackRoot: packBundle.outputDir,
        routeObjectiveLabelsByTraceId: {
          [routeRow.trace_id]: {
            focusLane: "hard_memory",
            strictHardMemoryEligible: true,
            oracleBestMode: "graph_prior_only",
            netUtilityDelta: -0.42,
            costSensitive: "high",
          },
        },
        generatedAt: "2026-04-07T12:00:00.000Z",
      });

      expect(validateGraphifyTrainingReviewBundleV1(bundle)).toMatchObject({ valid: true });
      const row = bundle.rows[0];
      expect(row?.route_objective_hint).not.toBeNull();
      expect(row?.route_objective_hint).toMatchObject({
        focus_lane: "hard_memory",
        strict_hard_memory_eligible: true,
        oracle_best_mode: "graph_prior_only",
        pairwise_preference: "graph_prior_only_over_learned_route",
        activation_preference: "stop_local",
        preference_weight: 1.42,
        utility_delta_vs_graph_prior_only: -0.42,
        cost_sensitivity: "high",
        projection_status: "trace_only",
        hard_negative_mining_mode: "mine_abstention_negatives",
      });
      expect(row?.route_objective_hint?.positive_action_ids).toEqual(["action_stop_local"]);
      expect(row?.route_objective_hint?.negative_action_ids).toEqual(["action_traverse_doc_1", "doc_2", "tool_1"]);
      expect(row?.route_objective_hint?.notes.join(" ")).toMatch(/trace-level route-objective hint only/i);
      expect(row?.review_notes.join(" ")).toMatch(/do not replace decision-point teacher labels/i);
    }
    finally {
      rmSync(tempRoot, { recursive: true, force: true });
    }
  });
});
