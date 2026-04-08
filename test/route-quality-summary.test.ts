import { describe, expect, it } from "vitest";

import { buildRouteQualitySummaryV1 } from "../src/brain-runtime/route-quality-summary.js";

describe("buildRouteQualitySummaryV1", () => {
  it("marks the route posture promotable when replay and compact health are both healthy", () => {
    const summary = buildRouteQualitySummaryV1({
      surface: "status",
      activePackVersion: 7,
      activePackId: "brain-pack-v7",
      routerIdentity: "route_fn.v1",
      replayVerdict: {
        passed: true,
        summary: "replay gate passed",
      },
      stopLocalWeights: [{ sourceNodeId: "episode_anchor", weight: 0.81 }],
      toolActionPriors: [
        { sourceNodeId: "episode_anchor", toolNodeId: "search_tool", weight: 0.93 },
        { sourceNodeId: "workflow_root", toolNodeId: "rerank_tool", weight: 0.72 },
      ],
      disabled: false,
      shadowMode: false,
      rolledBack: false,
      rollbackKey: "rollback:brain:route-quality:7",
      proofBundleId: "proof-bundle-7",
    });

    expect(summary).toMatchObject({
      contract: "openclawbrain_route_quality_summary.v1",
      posture: "promotable",
      activePackVersion: 7,
      activePackId: "brain-pack-v7",
      routerIdentity: "route_fn.v1",
      controlState: {
        summary: "controls: live",
      },
      stopLocalHealth: {
        status: "healthy",
        count: 1,
      },
      toolActionPriorsHealth: {
        status: "healthy",
        count: 2,
        sourceCount: 2,
        toolCount: 2,
      },
      rollbackLinkage: {
        rollbackKey: "rollback:brain:route-quality:7",
        proofBundleId: "proof-bundle-7",
        bound: true,
      },
    });
    expect(summary.summary).toContain("posture promotable");
    expect(summary.explainability).toContain("replay passed");
    expect(summary.explainability).toContain("controls: live");
  });

  it("holds the route posture when replay is blocked or the compact health is thin", () => {
    const summary = buildRouteQualitySummaryV1({
      surface: "proof",
      activePackVersion: 7,
      routerIdentity: "route_fn.v1",
      replayVerdict: {
        passed: false,
        summary: "replay gate blocked",
      },
      stopLocalWeights: [],
      toolActionPriors: [],
      disabled: false,
      shadowMode: false,
      rolledBack: false,
      rollbackKey: null,
      proofBundleId: null,
    });

    expect(summary.posture).toBe("held");
    expect(summary.stopLocalHealth.status).toBe("missing");
    expect(summary.toolActionPriorsHealth.status).toBe("missing");
    expect(summary.summary).toContain("posture held");
    expect(summary.explainability).toContain("replay did not pass");
  });

  it("quarantines the route posture when control flags are disabled or rolled back", () => {
    const summary = buildRouteQualitySummaryV1({
      surface: "status",
      activePackVersion: 7,
      routerIdentity: "route_fn.v1",
      replayVerdict: {
        passed: true,
        summary: "replay gate passed",
      },
      stopLocalWeights: [{ sourceNodeId: "episode_anchor", weight: 0.81 }],
      toolActionPriors: [{ sourceNodeId: "episode_anchor", toolNodeId: "search_tool", weight: 0.93 }],
      disabled: true,
      shadowMode: false,
      rolledBack: true,
      rollbackKey: "rollback:brain:route-quality:quarantined",
      proofBundleId: null,
    });

    expect(summary.posture).toBe("quarantined");
    expect(summary.controlState.summary).toContain("disabled");
    expect(summary.controlState.summary).toContain("rolled-back");
    expect(summary.rollbackLinkage.bound).toBe(true);
    expect(summary.explainability).toContain("posture quarantined");
  });
});
