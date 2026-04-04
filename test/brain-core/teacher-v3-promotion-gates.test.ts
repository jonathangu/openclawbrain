import { describe, expect, it } from "vitest";
import type { ProposalClass, TeacherProposalReplayGateV1 } from "../../src/brain-core/teacher-v3-contracts.js";
import {
  describeTeacherProposalReplayGate,
  describeTeacherProposalReplayGateReviewModeV1,
  isTeacherProposalPromotableClassV1,
  TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1,
  TEACHER_PROPOSAL_REVIEW_MODE_BY_CLASS_V1,
  TEACHER_PROPOSAL_REPLAY_GATES_V1,
  TEACHER_PROPOSAL_SHADOW_ONLY_CLASSES_V1,
} from "../../src/brain-core/teacher-v3-contracts.js";

describe("teacher v3 proposal promotion gates", () => {
  it("splits proposal classes into promotable and shadow-only review modes", () => {
    expect(TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1).toEqual(["compiler", "lint"]);
    expect(TEACHER_PROPOSAL_SHADOW_ONLY_CLASSES_V1).toEqual([
      "mutation",
      "forgetting",
      "correction",
    ]);
    expect(TEACHER_PROPOSAL_REVIEW_MODE_BY_CLASS_V1).toEqual({
      compiler: "promotable",
      lint: "promotable",
      mutation: "shadow_only",
      forgetting: "shadow_only",
      correction: "shadow_only",
    });
  });

  it("keeps the replay gate dimensions identical while review mode follows the class split", () => {
    const proposalClasses: ProposalClass[] = [
      "compiler",
      "lint",
      "mutation",
      "forgetting",
      "correction",
    ];

    expect(Object.keys(TEACHER_PROPOSAL_REPLAY_GATES_V1).sort()).toEqual(
      [...proposalClasses].sort(),
    );

    for (const proposalClass of proposalClasses) {
      const gate: TeacherProposalReplayGateV1 = describeTeacherProposalReplayGate(proposalClass);
      const reviewMode = describeTeacherProposalReplayGateReviewModeV1(proposalClass);

      expect(gate.proposalClass).toBe(proposalClass);
      expect(gate.reviewMode).toBe(reviewMode);
      expect(isTeacherProposalPromotableClassV1(proposalClass)).toBe(reviewMode === "promotable");
      expect(Object.keys(gate.dimensions).sort()).toEqual([
        "attributionFloor",
        "boundedness",
        "reversibility",
        "truthInvariants",
      ]);
      expect(gate.dimensions.truthInvariants.requirements).toEqual(
        expect.arrayContaining([
          "Explicit correction memory still outranks teacher synthesis.",
          "The live path stays read-only to the proposal.",
          "Evidence refs stay attached to any non-trivial claim.",
        ]),
      );
      expect(gate.dimensions.attributionFloor.requirements).toEqual(
        expect.arrayContaining([
          "Every proposal carries durable evidence refs.",
          "Source ids must be stable record ids, not display labels.",
          "Unattributed payload stays out of promotion.",
        ]),
      );
      expect(gate.dimensions.boundedness.requirements).toEqual(
        expect.arrayContaining([
          "Proposal subject sets stay finite and small.",
          "Payloads avoid raw corpus dumps and unbounded excerpts.",
          "Replay fits inside a single review pass.",
        ]),
      );
      expect(gate.dimensions.reversibility.requirements).toEqual(
        expect.arrayContaining([
          "RollbackKey identifies the reversible path.",
          "Prior state remains recoverable for replay.",
          "Rejected or superseded proposals keep lineage.",
        ]),
      );
    }
  });
});
