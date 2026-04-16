import { describe, expect, it } from "vitest";
import type { RouteDecisionRowV1 } from "../../src/brain-core/route-rows.js";
import {
  buildPolicySupervisionRowV1,
  summarizePolicySupervisionRowsV1,
  validatePolicySupervisionRowV1,
} from "../../src/brain-core/policy-supervision-rows.js";

function makeRouteRowFixture(): RouteDecisionRowV1 {
  return {
    schema_version: 1,
    row_id: "rr_policy_supervision_fixture_01",
    trace_id: "trace_policy_supervision_fixture_01",
    episode_id: "episode_policy_supervision_fixture_01",
    conversation_id: 42,
    route_fn_version: "brain-graph-traverse.v2",
    decision_point_id: "dp_policy_supervision_fixture_01",
    decision_point_kind: "local",
    source_node_id: "source_node_policy_supervision_fixture_01",
    expansion_index: 0,
    selection_index: 0,
    query_text: "find the best support packet and keep the path reviewable",
    cursor_path: ["root", "docs"],
    local_action_set: [],
    chosen_action_id: "action_traverse_doc_1",
    chosen_action_kind: "traverse",
    chosen_node_id: "doc_1",
    chosen_tool_name: null,
    chosen_tool_capability_id: null,
    chosen_tool_instance_id: null,
    chosen_action_probability: 0.82,
    stop_probability: 0.18,
    stop_label: "CONTINUE",
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
      request_digest: "req_policy_supervision_fixture_01",
      active_pack_id: "pack_policy_supervision_fixture_01",
      router_identity: "brain-graph-traverse.v2",
      candidate_node_ids: ["doc_1", "tool_1"],
      selected_node_ids: ["doc_1"],
      selected_traversal_node_ids: ["doc_1"],
      selected_path_node_ids: ["doc_1"],
      selected_seed_node_ids: [],
    },
    label_provenance: {
      provenance_id: "lp_policy_supervision_fixture_01",
      state: "matched",
      basis: "trace_id",
      confidence: 0.98,
      detail: "trace row is matched to the review fixture",
      source: "teacher",
      kind: "teacher_review",
      binding_mode: "trace_id",
      attribution_quality: "exact",
      feedback_richness: "followup_only",
      trace_id: "trace_policy_supervision_fixture_01",
      episode_id: "episode_policy_supervision_fixture_01",
      decision_point_id: "dp_policy_supervision_fixture_01",
      observation_id: null,
      supervision_id: null,
      update_id: null,
      candidate_ids: ["doc_1", "tool_1"],
      provenance_ref: "prov_lp_policy_supervision_fixture_01",
      content_hash: "sha256:policy-supervision-fixture-content",
      lineage_hash: "sha256:policy-supervision-fixture-lineage",
      created_at: 1_746_000_000_000,
    },
    evidence_spans: [],
    hard_negatives: ["tool_1"],
    created_at: "2026-04-07T12:00:00.000Z",
  };
}

describe("policy supervision rows", () => {
  it("projects an activate row from a learned_route trace label", () => {
    const row = buildPolicySupervisionRowV1({
      routeRow: makeRouteRowFixture(),
      traceLabel: {
        traceId: "trace_policy_supervision_fixture_01",
        focusLane: "must_fire_30",
        strictHardMemoryEligible: true,
        oracleBestMode: "learned_route",
        netUtilityDelta: 0.8,
        costSensitive: "low",
      },
    });

    expect(validatePolicySupervisionRowV1(row)).toMatchObject({ valid: true });
    expect(row).toMatchObject({
      row_type: "activate",
      row_weight: 1.8,
      confidence_target: 0.9,
      hard_negative_class: null,
      net_utility_delta: 0.8,
      net_utility_delta_source: "reviewed",
      projection_status: "trace_projected",
      oracle_best_mode: "learned_route",
    });

    const summary = summarizePolicySupervisionRowsV1([row]);
    expect(summary).toMatchObject({
      totalRows: 1,
      traceProjectedCount: 1,
      withNetUtilityDelta: 1,
      withHardNegativeClass: 0,
    });
    expect(summary.byRowType.activate).toBe(1);
  });

  it("projects an abstain row with tie-with-cost hard negative pressure", () => {
    const row = buildPolicySupervisionRowV1({
      routeRow: makeRouteRowFixture(),
      traceLabel: {
        traceId: "trace_policy_supervision_fixture_01",
        focusLane: "must_not_fire_100",
        strictHardMemoryEligible: false,
        oracleBestMode: "tie",
        costSensitive: "high",
      },
    });

    expect(validatePolicySupervisionRowV1(row)).toMatchObject({ valid: true });
    expect(row).toMatchObject({
      row_type: "abstain",
      row_weight: 2,
      confidence_target: 0.7,
      hard_negative_class: "tie_with_cost",
      net_utility_delta: null,
      net_utility_delta_source: null,
      projection_status: "trace_projected",
      oracle_best_mode: "tie",
    });
  });

  it("honors explicit mined hard-negative classes even when oracle mode alone is ambiguous", () => {
    const row = buildPolicySupervisionRowV1({
      routeRow: makeRouteRowFixture(),
      traceLabel: {
        traceId: "trace_policy_supervision_fixture_01",
        focusLane: "must_not_fire_100",
        strictHardMemoryEligible: false,
        oracleBestMode: "tie",
        costSensitive: "low",
        hardNegativeClass: "wrapper_heavy",
      },
    });

    expect(validatePolicySupervisionRowV1(row)).toMatchObject({ valid: true });
    expect(row).toMatchObject({
      row_type: "abstain",
      row_weight: 1.5,
      hard_negative_class: "wrapper_heavy",
      oracle_best_mode: "tie",
    });
  });
});
