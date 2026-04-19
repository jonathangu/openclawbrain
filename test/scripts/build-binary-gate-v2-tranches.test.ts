import { describe, expect, it } from "vitest";

import {
  buildBinaryGateV2HardNegativeSpec,
  buildBinaryGateV2Tranches,
  classifyTrapAnchor,
  splitMustFireAnchors,
  type HardNegativeSpec,
  type TrancheManifest,
} from "../../scripts/build-binary-gate-v2-tranches.ts";

function makeManifest(trancheId: string, anchors: TrancheManifest["anchors"]): TrancheManifest {
  return {
    contract: "learned_route_eval_tranche_manifest.v1",
    trancheId,
    taskId: "T-test",
    anchors,
    sourceManifests: ["task-artifacts/test-source.manifest.json"],
  };
}

describe("build-binary-gate-v2-tranches", () => {
  it("splits must-fire anchors into the explicit v2 positive buckets", () => {
    const split = splitMustFireAnchors([
      {
        traceId: "live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003",
        sourcePath: "trace-1.json",
        bucket: "owner constraints",
      },
      {
        traceId: "live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002",
        sourcePath: "trace-2.json",
        bucket: "merge/deploy closeout",
      },
      {
        traceId: "live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002",
        sourcePath: "trace-3.json",
        bucket: "short follow-up",
      },
      {
        traceId: "live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006",
        sourcePath: "trace-4.json",
        bucket: "blocker diagnosis",
      },
    ]);

    expect(split.must_fire_recent_decision.map((anchor) => anchor.traceId)).toEqual([
      "live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003",
    ]);
    expect(split.must_fire_exact_artifact.map((anchor) => anchor.traceId)).toEqual([
      "live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002",
    ]);
    expect(split.must_fire_resume_state.map((anchor) => anchor.traceId)).toEqual([
      "live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002",
    ]);
    expect(split.must_fire_stale_summary_repair.map((anchor) => anchor.traceId)).toEqual([
      "live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006",
    ]);
  });

  it("classifies trap anchors into wrapper, operator, and user-visible buckets", () => {
    expect(classifyTrapAnchor({
      traceId: "trace-wrapper",
      sourcePath: "wrapper.json",
      bucket: "main vector collapse",
      preview: "Conversation info (untrusted metadata): {\"message_id\":\"1\"}",
    })).toBe("trap_wrapper_system");

    expect(classifyTrapAnchor({
      traceId: "trace-operator",
      sourcePath: "operator.json",
      bucket: "main vector collapse",
      preview: "Additional requirement from Jon: the fix must treat explicit nonstandard OpenClaw homes as first-class.",
    })).toBe("trap_operator_artifact");

    expect(classifyTrapAnchor({
      traceId: "trace-user-visible",
      sourcePath: "user-visible.json",
      bucket: "main vector collapse",
      preview: "Please continue from the last approved plan and finish the remaining cleanup.",
    })).toBe("trap_user_visible_resume");
  });

  it("builds merged v2 lanes and extends hard-negative mappings for trap buckets", () => {
    const mustFireManifest = makeManifest("must-fire-30", [
      {
        traceId: "live-pelican-58e7c9e8-bc09-492d-8ce5-6e92f0078397-window-003",
        sourcePath: "trace-1.json",
        bucket: "owner constraints",
        whyIncluded: "Needs active owner direction.",
      },
      {
        traceId: "live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002",
        sourcePath: "trace-2.json",
        bucket: "merge/deploy closeout",
        whyIncluded: "Needs exact proof state.",
      },
      {
        traceId: "live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-002",
        sourcePath: "trace-3.json",
        bucket: "short follow-up",
        whyIncluded: "Needs recent continuity.",
      },
      {
        traceId: "live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006",
        sourcePath: "trace-4.json",
        bucket: "blocker diagnosis",
        whyIncluded: "Needs stale-summary repair.",
      },
    ]);
    const mustNotFireManifest = makeManifest("must-not-fire-100", [
      {
        traceId: "neg-1",
        sourcePath: "neg-1.json",
        bucket: "wrapper_or_continuation",
      },
      {
        traceId: "neg-2",
        sourcePath: "neg-2.json",
        bucket: "graph-prior-preferred operational",
      },
    ]);
    const trapManifest = makeManifest("vector-only-trap-50", [
      {
        traceId: "trap-1",
        sourcePath: "trap-1.json",
        bucket: "main vector collapse",
        preview: "Conversation info (untrusted metadata): {\"message_id\":\"1\"}",
      },
      {
        traceId: "trap-2",
        sourcePath: "trap-2.json",
        bucket: "main vector collapse",
        preview: "Additional requirement from Jon: the fix must treat explicit nonstandard OpenClaw homes as first-class.",
      },
      {
        traceId: "trap-3",
        sourcePath: "trap-3.json",
        bucket: "main vector collapse",
        preview: "Please continue from the last approved plan and finish the remaining cleanup.",
      },
    ]);
    const baseSpec: HardNegativeSpec = {
      missionTestCommand: "npm run test:learned-route-mission",
      weightFloorsByClass: {
        wrapper_heavy: 1.5,
      },
    };

    const built = buildBinaryGateV2Tranches({
      outputTaskId: "T-20260419-267",
      mustFireManifest,
      mustNotFireManifest,
      trapManifest,
      baseHardNegativeSpec: baseSpec,
      generatedAt: "2026-04-19T13:00:00.000Z",
    });

    expect(built.mergedPositive.trancheId).toBe("must_fire_binary_gate_v2");
    expect(built.mergedPositive.anchorTraceCount).toBe(4);
    expect(built.mergedAbstention.trancheId).toBe("must_not_fire_binary_gate_v2");
    expect(built.mergedAbstention.anchorTraceCount).toBe(5);
    expect(built.mergedAbstention.anchors.map((anchor) => anchor.bucket)).toEqual([
      "wrapper_or_continuation",
      "graph_prior_preferred_operational",
      "trap_wrapper_system",
      "trap_operator_artifact",
      "trap_user_visible_resume",
    ]);
    expect(built.hardNegativeSpec.bucketToHardNegativeClass).toMatchObject({
      trap_wrapper_system: "wrapper_heavy",
      trap_operator_artifact: "unnecessary_activation",
      trap_user_visible_resume: "graph_prior_preferred",
    });
  });

  it("builds a hard-negative spec with explicit trap bucket mappings", () => {
    const spec = buildBinaryGateV2HardNegativeSpec({
      missionTestCommand: "npm run test:learned-route-mission",
      weightFloorsByClass: {
        tie_with_cost: 2,
      },
    });

    expect(spec.contract).toBe("learned_route_hard_negative_mining_spec.v1");
    expect(spec.bucketToHardNegativeClass).toMatchObject({
      trap_wrapper_system: "wrapper_heavy",
      trap_operator_artifact: "unnecessary_activation",
      trap_user_visible_resume: "graph_prior_preferred",
    });
    expect(spec.weightFloorsByClass?.tie_with_cost).toBe(2);
  });
});
