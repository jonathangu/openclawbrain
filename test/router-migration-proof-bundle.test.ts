import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { buildTeacherV3ProofBundle } from "../scripts/teacher-v3-proof-bundle.mjs";

describe("teacher-v3 proof bundle router migration surface", () => {
  const tempDirs = new Set<string>();

  afterEach(() => {
    for (const dir of tempDirs) {
      rmSync(dir, { recursive: true, force: true });
    }
    tempDirs.clear();
  });

  it("surfaces old_live vs base_only vs mixed replay and keeps mixed promotable when explicit corrections are preserved", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "openclawbrain-router-migration-proof-"));
    tempDirs.add(tempDir);

    const bundle = buildTeacherV3ProofBundle({
      bundleId: "router-migration-proof-bundle",
      bundleStartedAt: "2026-04-05T23:00:00Z",
      outputDir: join(tempDir, "bundle"),
      runtimeStatusCommand: "openclawbrain status --detailed",
      runtimeStatus: {
        serveState: "serving_active_pack",
        currentPackVersion: 7,
        currentPackPromotedAt: "2026-04-05T22:00:00Z",
        currentPackMetadata: {
          reason: "live pack",
          kind: "promoted_pack",
        },
        teacherConfigured: true,
        teacherProvider: "openai",
        teacherModel: "gpt-4.1",
        operatorHealth: {
          status: "healthy",
          detail: "runtime and proof surface are aligned",
        },
        learningHealth: {
          status: "healthy",
          detail: "learning loop is steady",
        },
        routeTraceCount: 12,
        supervisionCount: 4,
        recentTraceCount: 2,
        pendingLabels: 1,
        pendingObservations: 3,
        lastCompileReportSummary: "compile report is bounded",
        lastAssemblyDecision: {
          summary: "assembly summary",
          verdict: "approved",
        },
        lastPrefetchDecision: {
          summary: "prefetch summary",
          verdict: "ready",
        },
        lastPromotionReason: "current pack promoted",
        lastPromotionVerdict: {
          verdict: "promoted",
          summary: "promotion succeeded",
        },
        lastReplayFailureReason: null,
        lastReplayGateVerdict: {
          verdict: "pass",
          summary: "replay gate passed",
        },
      },
      operatorProofCommand: "openclawbrain proof --openclaw-home ~/.openclaw",
      operatorProof: {
        bundleDir: "artifacts/operator-proof-router-migration",
        command: "openclawbrain proof --openclaw-home ~/.openclaw",
        summary: "operator proof summary",
        verdict: {
          verdict: "success_and_proven",
          severity: "info",
          why: "runtime truth and proof truth were aligned",
          missingProofs: [],
        },
        runtimeLoadProofPath: "~/.openclaw/activation/attachment-truth/runtime-load-proofs.json",
        runtimeLoadProofExists: true,
        stepCount: 5,
        postBundleCount: 2,
      },
      docsTruth: {
        path: "docs/architecture/teacher-v3-proof.md",
        title: "Teacher v3 proposal reporting / proof surfaces",
        summary: "Design-only mapping of shipped truth to target-state proof surfaces.",
      },
      proposalClass: "compiler",
      proposalLane: "compiler",
      proposalStatus: "promotable",
      proposalRecord: {
        proposalId: "prop_router_migration_01",
        proposalClass: "compiler",
        proposalLane: "compiler",
        status: "promotable",
        reviewMode: "promotable",
        lineage: {
          proposalClass: "compiler",
          basePackVersion: 7,
          baseGraphHash: "graph_old_live",
          producerVersion: "router-migration-proof@1.0.0",
          promptHash: "prompt-router-migration-proof",
          templateId: "router-migration/proof-v1",
          scope: "router-migration",
          idempotencyKey: "router-migration::proof::01",
        },
        subjectIds: ["router-migration", "old_live", "base_only", "mixed"],
        evidence: [],
        counterevidence: [],
        payload: {
          summary: "router migration proof",
        },
        confidence: 0.97,
        replaySuites: ["router-migration-replay"],
        rollbackKey: "rollback:router:migration:01",
        proofBundle: {
          bundleId: "router-migration-proof-bundle",
          rollbackKey: "rollback:router:migration:01",
        },
      },
      routerMigrationComparison: {
        migrationId: "migration_router_01",
        proposalId: "prop_router_migration_01",
        rollbackKey: "rollback:router:migration:01",
        proofBundleId: "router-migration-proof-bundle",
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
            summary: "base prior should recover a high-authority route",
            highAuthority: true,
            preserved: {
              old_live: false,
              base_only: true,
              mixed: true,
            },
          },
        ],
      },
    });

    expect(bundle.proposalReport.routerMigrationComparison?.comparison).toMatchObject({
      decision: "promote",
      winner: "mixed",
      blocked: false,
      allowed: true,
    });
    expect(bundle.statusReport.routerMigrationComparison?.explicitCorrectionProtection.protected).toBe(true);
    expect(bundle.summaryMarkdown).toContain("Router migration comparison");
    expect(bundle.summaryMarkdown).toContain("old_live");
    expect(bundle.summaryMarkdown).toContain("base_only");
    expect(bundle.summaryMarkdown).toContain("mixed");
    expect(bundle.summaryMarkdown).toContain("explicit correction protection: yes");
    expect(bundle.verdictReport.verdict).toBe("reviewable");
    expect(bundle.verdictReport.why).toContain("mixed router migration replay passed");
  });
});
