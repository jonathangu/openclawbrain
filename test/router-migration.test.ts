import { describe, expect, it } from "vitest";

import { summarizeRouterMigrationComparisonV1 } from "../src/brain-core/router-migration.js";

describe("summarizeRouterMigrationComparisonV1", () => {
  it("prefers mixed when it preserves the replay set and keeps explicit corrections intact", () => {
    const comparison = summarizeRouterMigrationComparisonV1({
      migrationId: "migration_mix_01",
      proposalId: "proposal_mix_01",
      rollbackKey: "rollback:router:migration:01",
      proofBundleId: "teacher-v3-proof-migration-01",
      priorLivePackVersion: 7,
      priorLivePackId: "live_pack_07",
      priorRouterChecksum: "sha256:old-live-router",
      proofBundleFiles: [
        "summary.md",
        "status.json",
        "surface-map.json",
        "proposal-report.json",
        "verdict.json",
      ],
      variants: {
        old_live: {
          packVersion: 7,
          packId: "live_pack_07",
          routerChecksum: "sha256:old-live-router",
          graphHash: "graph_old_live",
        },
        base_only: {
          packVersion: 8,
          packId: "base_pack_08",
          routerChecksum: "sha256:base-only-router",
          graphHash: "graph_base_only",
        },
        mixed: {
          packVersion: 9,
          packId: "mixed_pack_09",
          routerChecksum: "sha256:mixed-router",
          graphHash: "graph_mixed",
        },
      },
      replayCases: [
        {
          caseId: "explicit_correction_01",
          summary: "explicit correction must survive the upgrade path",
          highAuthority: true,
          explicitCorrection: true,
          preserved: {
            old_live: true,
            base_only: true,
            mixed: true,
          },
        },
        {
          caseId: "high_authority_02",
          summary: "high-authority historical route should be preserved by the new base prior",
          highAuthority: true,
          explicitCorrection: false,
          preserved: {
            old_live: false,
            base_only: true,
            mixed: true,
          },
        },
        {
          caseId: "routine_03",
          summary: "routine replay case should prefer the mixed policy",
          preserved: {
            old_live: false,
            base_only: false,
            mixed: true,
          },
        },
      ],
    });

    expect(comparison.comparison).toMatchObject({
      decision: "promote",
      winner: "mixed",
      blocked: false,
      allowed: true,
    });
    expect(comparison.variants.old_live.preservedCaseCount).toBe(1);
    expect(comparison.variants.base_only.preservedCaseCount).toBe(2);
    expect(comparison.variants.mixed.preservedCaseCount).toBe(3);
    expect(comparison.explicitCorrectionProtection).toMatchObject({
      protected: true,
      preservedCount: 1,
      regressionCount: 0,
    });
    expect(comparison.rollback.available).toBe(true);
    expect(comparison.proofBundleExpectations).toMatchObject({
      exactArtifactRefs: true,
      checksumsBound: true,
      rollbackBound: true,
      proofBundleBound: true,
    });
    expect(comparison.summary).toContain("old_live=");
    expect(comparison.summary).toContain("base_only=");
    expect(comparison.summary).toContain("mixed=");
  });

  it("blocks mixed promotion when it regresses an explicit correction", () => {
    const comparison = summarizeRouterMigrationComparisonV1({
      migrationId: "migration_block_01",
      proposalId: "proposal_block_01",
      rollbackKey: "rollback:router:migration:block",
      proofBundleId: "teacher-v3-proof-migration-block",
      priorLivePackVersion: 7,
      priorLivePackId: "live_pack_07",
      priorRouterChecksum: "sha256:old-live-router",
      proofBundleFiles: [
        "summary.md",
        "status.json",
        "surface-map.json",
        "proposal-report.json",
        "verdict.json",
      ],
      variants: {
        old_live: {
          packVersion: 7,
          packId: "live_pack_07",
          routerChecksum: "sha256:old-live-router",
          graphHash: "graph_old_live",
        },
        base_only: {
          packVersion: 8,
          packId: "base_pack_08",
          routerChecksum: "sha256:base-only-router",
          graphHash: "graph_base_only",
        },
        mixed: {
          packVersion: 9,
          packId: "mixed_pack_09",
          routerChecksum: "sha256:mixed-router",
          graphHash: "graph_mixed",
        },
      },
      replayCases: [
        {
          caseId: "explicit_correction_01",
          summary: "explicit correction must survive the upgrade path",
          highAuthority: true,
          explicitCorrection: true,
          preserved: {
            old_live: true,
            base_only: true,
            mixed: false,
          },
        },
      ],
    });

    expect(comparison.comparison).toMatchObject({
      decision: "hold",
      winner: "old_live",
      blocked: true,
      allowed: false,
    });
    expect(comparison.explicitCorrectionProtection).toMatchObject({
      protected: false,
      preservedCount: 0,
      regressionCount: 1,
    });
    expect(comparison.comparison.blockers).toContain("mixed regressed 1 explicit-correction case");
    expect(comparison.summary).toContain("does not yet clear the migration gate");
  });
});
