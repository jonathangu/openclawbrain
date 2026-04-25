import { describe, expect, it } from "vitest";

import type { RouteDecisionRowV1 } from "../../src/brain-core/cold-start-router-contracts.js";
import {
  loadColdStartRouterArtifactBundleV1,
  selectColdStartRouteCandidateIdsFromArtifactBundleV1,
} from "../../src/brain-core/cold-start-router-runtime.js";

function makeReplayRow(): RouteDecisionRowV1 {
  return {
    row_id: "cold-start-runtime-multiselect-row",
    dataset_id: "cold-start-runtime-multiselect-dataset",
    query: "Need the replay feedback and event context.",
    cursor_path: ["felt_resume_25"],
    candidate_set: [
      {
        candidate_id: "pack:event:a:feedback",
        candidate_type: "graph_node",
        semantic_class: "feedback_context",
        authority: "recorded_session_replay",
        freshness: "replay_eval",
        score_hint: 0.95,
      },
      {
        candidate_id: "pack:event:b:event",
        candidate_type: "graph_node",
        semantic_class: "event_context",
        authority: "recorded_session_replay",
        freshness: "replay_eval",
        score_hint: 0.9,
      },
      {
        candidate_id: "phrase-context:c",
        candidate_type: "graph_node",
        semantic_class: "phrase_context",
        authority: "recorded_session_replay",
        freshness: "replay_eval",
        score_hint: 0.65,
      },
    ],
    teacher_action: { kind: "tool", tool_name: "__recorded_session_replay_candidate_override__" },
    stop_label: "CONTINUE",
    evidence_spans: [{ source_ref: "replay:evidence", start: 0, end: 10, excerpt: "replay" }],
    hard_negatives: [],
    outcome_gain: 1,
    provenance: {
      dataset: "cold-start-runtime-multiselect-dataset",
      source_license: "internal_local_only",
      source_family: "agent_traces",
      source_snapshot_ref: "snapshot:cold-start-runtime-multiselect",
      recorded_by: "test",
      recorded_at: "2026-04-25T14:30:00.000Z",
      review_status: "approved_eval_only",
    },
    split_tag: "eval_only",
    created_at: "2026-04-25T14:30:00.000Z",
  };
}

describe("cold-start router runtime selection", () => {
  it("keeps legacy single-select by default but allows bounded close-score multi-select", () => {
    const artifactBundle = loadColdStartRouterArtifactBundleV1(
      "artifacts/activation-first-gating-retune/T-20260419-269/candidate-artifact",
    );
    const row = makeReplayRow();

    const legacy = selectColdStartRouteCandidateIdsFromArtifactBundleV1({ artifactBundle, row });
    const boundedMulti = selectColdStartRouteCandidateIdsFromArtifactBundleV1({
      artifactBundle,
      row,
      maxCandidateIds: 3,
    });
    const tightWindow = selectColdStartRouteCandidateIdsFromArtifactBundleV1({
      artifactBundle,
      row,
      maxCandidateIds: 3,
      multiSelectScoreWindow: 0.01,
    });

    expect(legacy.stopped).toBe(false);
    expect(legacy.selectedCandidateIds).toEqual(["pack:event:a:feedback"]);
    expect(boundedMulti.selectedCandidateIds).toEqual([
      "pack:event:a:feedback",
      "pack:event:b:event",
    ]);
    expect(tightWindow.selectedCandidateIds).toEqual(["pack:event:a:feedback"]);
    expect(boundedMulti.selectedCandidateIds.length).toBeLessThanOrEqual(3);
  });
});
