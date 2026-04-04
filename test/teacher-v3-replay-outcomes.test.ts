import { describe, expect, it } from "vitest";
import {
  buildTeacherV3ReplayOutcomeSummary,
  captureTeacherV3ReplayOutcomes,
} from "../scripts/teacher-v3-replay-outcomes.mjs";

describe("teacher v3 replay outcomes harness", () => {
  it("normalizes explicit replay outcomes across promotable and shadow-only proposal classes", () => {
    const capture = captureTeacherV3ReplayOutcomes({
      bundleId: "teacher-v3-proof-01",
      proposalId: "prop_teacher_v3_01",
      proposalClass: "compiler",
      replaySuites: ["compiler-shape-smoke", "mutation-shadow-smoke"],
      replayOutcomes: [
        {
          outcomeId: "outcome_01",
          replaySuite: "compiler-shape-smoke",
          proposalClass: "compiler",
          reviewMode: "promotable",
          result: "pass",
          source: "proposal_record",
          summary: "compiler replay passed",
          capturedAt: "2026-04-03T18:30:00Z",
        },
        {
          outcomeId: "outcome_02",
          replaySuite: "mutation-shadow-smoke",
          proposalClass: "mutation",
          reviewMode: "shadow_only",
          result: "warn",
          source: "proposal_record",
          summary: "shadow lane stayed bounded",
          capturedAt: "2026-04-03T18:30:15Z",
        },
      ],
    });

    expect(capture.outcomes).toHaveLength(2);
    expect(capture.outcomes[0]).toMatchObject({
      outcomeId: "outcome_01",
      proposalClass: "compiler",
      reviewMode: "promotable",
      result: "pass",
      source: "proposal_record",
    });
    expect(capture.outcomes[1]).toMatchObject({
      outcomeId: "outcome_02",
      proposalClass: "mutation",
      reviewMode: "shadow_only",
      result: "warn",
      source: "proposal_record",
    });
    expect(capture.summary).toMatchObject({
      replayOutcomeCount: 2,
      replaySuites: ["compiler-shape-smoke", "mutation-shadow-smoke"],
      resultCounts: { pass: 1, warn: 1, fail: 0 },
      reviewModeCounts: { promotable: 1, shadow_only: 1 },
      sourceCounts: { proposal_record: 2, proof_bundle: 0, derived: 0 },
    });
    expect(capture.summary.summary).toContain("promotable=1");
    expect(buildTeacherV3ReplayOutcomeSummary(capture.outcomes)).toEqual(capture.summary);
  });

  it("derives a proof-bundle replay outcome when no explicit record is available", () => {
    const capture = captureTeacherV3ReplayOutcomes({
      bundleId: "teacher-v3-proof-02",
      proposalId: "prop_teacher_v3_02",
      proposalClass: "lint",
      replaySuites: ["lint-review-smoke"],
      proofVerdict: {
        verdict: "reviewable",
        severity: "info",
        why: "proof review stayed bounded",
      },
    });

    expect(capture.outcomes).toHaveLength(1);
    expect(capture.outcomes[0]).toMatchObject({
      outcomeId: "teacher-v3-proof-02:proof-bundle",
      replaySuite: "lint-review-smoke",
      proposalClass: "lint",
      reviewMode: "promotable",
      result: "pass",
      source: "proof_bundle",
    });
    expect(capture.summary).toMatchObject({
      replayOutcomeCount: 1,
      replaySuites: ["lint-review-smoke"],
      resultCounts: { pass: 1, warn: 0, fail: 0 },
      reviewModeCounts: { promotable: 1, shadow_only: 0 },
      sourceCounts: { proposal_record: 0, proof_bundle: 1, derived: 0 },
    });
    expect(capture.summary.summary).toContain("proof_bundle=1");
  });
});
