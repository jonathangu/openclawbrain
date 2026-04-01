import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { describe, expect, it } from "vitest";
import {
  buildHealthSnapshot,
  buildNightlyAggregate,
  collectBundleCandidates,
  formatHealthMarkdown,
  formatNightlyMarkdown,
  summarizeScan,
} from "../scripts/proof-cron.mjs";

function tempWorkspace() {
  return mkdtempSync(path.join(os.tmpdir(), "ocb-proof-cron-"));
}

function writeText(filePath: string, content: string) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, content, "utf8");
}

function writeJson(filePath: string, value: unknown) {
  writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

describe("proof cron bundle scanning", () => {
  it("classifies bundle roots and excludes the cron output root", () => {
    const workspaceRoot = tempWorkspace();
    const repoRoot = path.join(workspaceRoot, "openclawbrain");
    const artifactsRoot = path.join(workspaceRoot, "artifacts");
    const docsEvidenceRoot = path.join(repoRoot, "docs", "evidence");
    const excludedRoot = path.join(artifactsRoot, "openclawbrain-proof-cron");

    const operatorRoot = path.join(artifactsRoot, "operator-proof-20260331-120000Z");
    writeText(path.join(operatorRoot, "summary.md"), "# operator proof\n");
    writeJson(path.join(operatorRoot, "steps.json"), {
      steps: [
        { stepId: "01", durationMs: 1000, resultClass: "success" },
        { stepId: "02", durationMs: 500, resultClass: "success" },
      ],
    });
    writeJson(path.join(operatorRoot, "verdict.json"), {
      bundleStartedAt: "2026-03-31T12:00:00.000Z",
      verdict: "success_and_proven",
      severity: "none",
      runtimeLoadProofPath: "/tmp/runtime-load-proofs.json",
      warnings: [],
    });
    writeJson(path.join(operatorRoot, "validation-report.json"), { ok: true });

    const replayRoot = path.join(docsEvidenceRoot, "2026-03-26", "abc123", "recorded-session-replay", "trace-a");
    writeText(path.join(replayRoot, "summary.md"), "# replay proof\n");
    writeJson(path.join(replayRoot, "manifest.json"), { contract: "recorded_session_replay_manifest.v1" });
    writeJson(path.join(replayRoot, "bundle.json"), {
      contract: "recorded_session_replay_bundle.v1",
      traceId: "trace-a",
      generatedAt: "2026-03-26T00:10:00.000Z",
      recordedAt: "2026-03-26T00:00:00.000Z",
      traceHash: "sha256-trace",
      fixtureHash: "sha256-fixture",
      scoreHash: "sha256-score",
      bundleHash: "sha256-bundle",
      summary: { winnerMode: "learned_route" },
      modes: [],
    });
    writeJson(path.join(replayRoot, "summary-tables.json"), {
      winnerMode: "learned_route",
      ranking: [
        { mode: "learned_route", qualityScore: 90 },
        { mode: "vector_only", qualityScore: 80 },
      ],
      modes: [
        { mode: "learned_route", qualityScore: 90 },
        { mode: "vector_only", qualityScore: 80 },
      ],
    });
    writeJson(path.join(replayRoot, "coverage-snapshot.json"), {
      totalTurns: 4,
      compileOkRate: 0.75,
      phraseHitRate: 0.75,
      modes: [
        { mode: "learned_route", learnedRouteTurnRate: 0.5 },
        { mode: "vector_only", learnedRouteTurnRate: 0 },
      ],
    });
    writeJson(path.join(replayRoot, "hardening-snapshot.json"), { warnings: [] });
    writeJson(path.join(replayRoot, "validation-report.json"), { ok: true, verifiedFileCount: 6 });

    const hostRoot = path.join(docsEvidenceRoot, "2026-03-26", "abc123", "host-proof");
    writeText(path.join(hostRoot, "summary.md"), "# host evidence\n");
    writeJson(path.join(hostRoot, "status.json"), {
      runtimeVersion: "2026.3.13",
      workerHealthy: true,
      workerMode: "child",
      currentPackVersion: 17,
      recentDecisionSummary: {
        sampleSize: 2,
        clipRate: { rate: 0.1 },
        failOpenRate: { rate: 0.05 },
      },
      securityAudit: { summary: { critical: 1, warn: 2 } },
      sessions: { count: 1, recent: [{ percentUsed: 12 }] },
      memory: { files: 4, chunks: 8 },
      gateway: { reachable: true },
      promotionStory: { summary: { currentPackVersion: 17 } },
    });
    writeJson(path.join(hostRoot, "doctor.json"), { ok: true });
    writeJson(path.join(hostRoot, "config-snapshot.json"), { ok: true });
    writeJson(path.join(hostRoot, "validation-report.json"), { ok: true });

    const excludedProof = path.join(excludedRoot, "operator-proof-20260331-121500Z");
    writeText(path.join(excludedProof, "summary.md"), "# excluded\n");
    writeJson(path.join(excludedProof, "steps.json"), { steps: [] });
    writeJson(path.join(excludedProof, "verdict.json"), { bundleStartedAt: "2026-03-31T12:15:00.000Z", verdict: "success_and_proven" });

    const candidates = collectBundleCandidates([artifactsRoot, docsEvidenceRoot], [excludedRoot]);
    const bundleKinds = candidates.map((bundle) => bundle.kind).sort();

    expect(bundleKinds).toEqual(["host-evidence", "operator-proof", "recorded-session-replay"]);

    const bundles = summarizeScan(candidates, new Date("2026-03-31T13:00:00.000Z"), workspaceRoot);
    expect(bundles.find((bundle) => bundle.kind === "operator-proof")?.metrics.totalStepDurationMs).toBe(1500);
    expect(bundles.find((bundle) => bundle.kind === "recorded-session-replay")?.metrics.winnerMode).toBe("learned_route");
    expect(bundles.find((bundle) => bundle.kind === "host-evidence")?.metrics.securityCriticalCount).toBe(1);
  });
});

describe("proof cron metric surfaces", () => {
  it("builds a useful health snapshot and nightly aggregate", () => {
    const now = new Date("2026-03-31T13:00:00.000Z");
    const statusProbe = {
      command: "node bin/openclawbrain.js status --json",
      startedAt: "2026-03-31T12:59:00.000Z",
      endedAt: "2026-03-31T12:59:02.000Z",
      durationMs: 2000,
      exitCode: 0,
      signal: null,
      stdout: "{}",
      stderr: "",
      parsed: {
        workerHealthy: true,
        workerMode: "child",
        currentPackVersion: 17,
        recentDecisionSummary: {
          sampleSize: 3,
          clipRate: { rate: 0.25 },
          failOpenRate: { rate: 0.1 },
        },
        securityAudit: { summary: { critical: 1, warn: 2 } },
      },
    };

    const bundles = [
      {
        kind: "operator-proof",
        bundleId: "operator-proof-20260331-120000Z",
        relativePath: "artifacts/operator-proof-20260331-120000Z",
        canonicalAt: "2026-03-31T12:00:00.000Z",
        ageDays: 0,
        fileCount: 5,
        artifactBytes: 2000,
        validationOk: true,
        metrics: {
          totalStepDurationMs: 1500,
          stepCount: 2,
          verdict: "success_and_proven",
        },
      },
      {
        kind: "recorded-session-replay",
        bundleId: "trace-a",
        relativePath: "openclawbrain/docs/evidence/2026-03-26/abc123/recorded-session-replay/trace-a",
        canonicalAt: "2026-03-26T00:10:00.000Z",
        ageDays: 5,
        fileCount: 6,
        artifactBytes: 4000,
        validationOk: true,
        metrics: {
          winnerMode: "learned_route",
          winnerScore: 90,
          compileOkRate: 0.75,
          phraseHitRate: 0.75,
          learnedRouteTurnRate: 0.25,
          totalTurns: 4,
        },
      },
      {
        kind: "host-evidence",
        bundleId: "host-proof",
        relativePath: "openclawbrain/docs/evidence/2026-03-26/abc123/host-proof",
        canonicalAt: "2026-03-26T00:00:00.000Z",
        ageDays: 5,
        fileCount: 4,
        artifactBytes: 1000,
        validationOk: true,
        metrics: {
          securityCriticalCount: 1,
          securityWarnCount: 2,
          gatewayReachable: true,
          workerHealthy: true,
          memoryFiles: 4,
          sessionCount: 1,
        },
      },
    ];

    const config = {
      healthFreshnessDays: 7,
      freshnessThresholdDays: 21,
    };

    const health = buildHealthSnapshot({ config, statusProbe, bundles, now, scanDurationMs: 42 });
    expect(health.proofInventory.bundleCount).toBe(3);
    expect(health.performance.operatorStepMsTotal).toBe(1500);
    expect(health.costProxy.artifactBytes).toBe(7000);
    expect(health.latestBundles.map((bundle) => bundle.kind)).toEqual(["operator-proof", "recorded-session-replay", "host-evidence"]);
    expect(formatHealthMarkdown(health)).toContain("proof minutes proxy");
    expect(formatHealthMarkdown(health)).toContain("clip rate: 0.25");

    const aggregate = buildNightlyAggregate({ config, bundles, now, scanDurationMs: 42 });
    expect(aggregate.bundleTypeCounts.operatorProof).toBe(1);
    expect(aggregate.bundleTypeCounts.recordedSessionReplay).toBe(1);
    expect(aggregate.bundleTypeCounts.hostEvidence).toBe(1);
    expect(aggregate.replayMetrics.winnerModeCounts.learned_route).toBe(1);
    expect(aggregate.operatorMetrics.stepMsTotal).toBe(1500);
    expect(aggregate.costProxy.bundleCount).toBe(3);
    expect(formatNightlyMarkdown(aggregate)).toContain("winner modes");
    expect(formatNightlyMarkdown(aggregate)).toContain("proof minutes");
  });
});
