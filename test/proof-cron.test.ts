import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { describe, expect, it } from "vitest";
// @ts-ignore runtime script module has no checked-in typed surface yet
import { buildHealthSnapshot, buildNightlyAggregate, collectBundleCandidates, formatHealthMarkdown, formatNightlyMarkdown, loadConfig, summarizeScan } from "../scripts/proof-cron.mjs";

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
      summary: { winnerMode: "learned_route", ranking: [{ mode: "learned_route", qualityScore: 90 }] },
      modes: [
        {
          mode: "no_brain",
          turns: [
            {
              turnId: "turn-1",
              selectedContextIds: ["no-brain-ctx-1"],
              selectedContextTexts: ["alpha"],
              observability: { selectionDigestCount: 0 },
            },
          ],
        },
        {
          mode: "vector_only",
          turns: [
            {
              turnId: "turn-2",
              selectedContextIds: ["vector-ctx-1"],
              selectedContextTexts: ["beta beta"],
              observability: { selectionDigestCount: 2 },
            },
          ],
        },
        {
          mode: "graph_prior_only",
          turns: [
            {
              turnId: "turn-1",
              selectedContextIds: ["graph-ctx-1"],
              selectedContextTexts: ["gamma"],
              observability: { selectionDigestCount: 1 },
            },
          ],
        },
        {
          mode: "learned_route",
          turns: [
            {
              turnId: "turn-2",
              selectedContextIds: ["learned-ctx-1"],
              selectedContextTexts: ["delta delta"],
              observability: { selectionDigestCount: 3 },
            },
          ],
        },
      ],
    });
    writeJson(path.join(replayRoot, "trace.json"), {
      contract: "recorded_session_trace.v1",
      traceId: "trace-a",
      turns: [
        { turnId: "turn-1", feedback: [{ kind: "teaching" }] },
        { turnId: "turn-2", feedback: [{ kind: "approval" }] },
      ],
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
    const bundleKinds = candidates.map((bundle: any) => bundle.kind).sort();

    expect(bundleKinds).toEqual(["host-evidence", "operator-proof", "recorded-session-replay"]);

    const bundles = summarizeScan(candidates, new Date("2026-03-31T13:00:00.000Z"), workspaceRoot);
    expect(bundles.find((bundle: any) => bundle.kind === "operator-proof")?.metrics.totalStepDurationMs).toBe(1500);
    const replayBundle = bundles.find((bundle: any) => bundle.kind === "recorded-session-replay");
    expect(replayBundle?.metrics.winnerMode).toBe("learned_route");
    expect(replayBundle?.metrics.selectedContextChars).toBe(30);
    expect(replayBundle?.metrics.selectedContextBlockCount).toBe(4);
    expect(replayBundle?.metrics.estimatedPromptTokens).toBe(10);
    expect(replayBundle?.metrics.retrievalToolHopCount).toBe(6);
    expect(replayBundle?.metrics.retrievalToolHopTurnCount).toBe(3);
    expect(replayBundle?.metrics.feedbackEventCount).toBe(2);
    expect(replayBundle?.metrics.nonApprovalFeedbackEventCount).toBe(1);
    expect(replayBundle?.metrics.turnsWithNonApprovalFeedbackCount).toBe(1);
    expect(replayBundle?.metrics.savingsByMode).toEqual([
      {
        mode: "no_brain",
        turnCount: 1,
        selectedContextBlockCount: 1,
        selectedContextChars: 5,
        estimatedPromptTokens: 2,
        retrievalToolHopCount: 0,
        retrievalToolHopTurnCount: 0,
        selectedContextCharsPerTurnMean: 5,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 2,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 0,
        retrievalToolHopTurnRate: 0,
      },
      {
        mode: "vector_only",
        turnCount: 1,
        selectedContextBlockCount: 1,
        selectedContextChars: 9,
        estimatedPromptTokens: 3,
        retrievalToolHopCount: 2,
        retrievalToolHopTurnCount: 1,
        selectedContextCharsPerTurnMean: 9,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 3,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 2,
        retrievalToolHopTurnRate: 1,
      },
      {
        mode: "graph_prior_only",
        turnCount: 1,
        selectedContextBlockCount: 1,
        selectedContextChars: 5,
        estimatedPromptTokens: 2,
        retrievalToolHopCount: 1,
        retrievalToolHopTurnCount: 1,
        selectedContextCharsPerTurnMean: 5,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 2,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 1,
        retrievalToolHopTurnRate: 1,
      },
      {
        mode: "learned_route",
        turnCount: 1,
        selectedContextBlockCount: 1,
        selectedContextChars: 11,
        estimatedPromptTokens: 3,
        retrievalToolHopCount: 3,
        retrievalToolHopTurnCount: 1,
        selectedContextCharsPerTurnMean: 11,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 3,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 3,
        retrievalToolHopTurnRate: 1,
      },
    ]);
    expect(bundles.find((bundle: any) => bundle.kind === "host-evidence")?.metrics.securityCriticalCount).toBe(1);
  });
});

describe("proof cron config", () => {
  it("migrates the legacy brain-cli status probe to the operator CLI probe", () => {
    const workspaceRoot = tempWorkspace();
    const configPath = path.join(workspaceRoot, "cron-config.json");
    writeJson(configPath, {
      statusCommand: [
        "node",
        "bin/openclawbrain.js",
        "status",
        "--openclaw-home",
        "{{openclawHome}}",
        "--json",
      ],
    });

    const loaded = loadConfig(configPath, { openclawHome: path.join(workspaceRoot, ".openclaw") });
    expect(loaded.statusCommand).toEqual([
      process.execPath,
      path.join(process.cwd(), "packages", "cli", "dist", "src", "cli.js"),
      "status",
      "--openclaw-home",
      "{{openclawHome}}",
      "--json",
    ]);
  });
});

describe("proof cron metric surfaces", () => {
  it("builds a useful health snapshot and nightly aggregate", () => {
    const now = new Date("2026-03-31T13:00:00.000Z");
    const statusProbe = {
      command: "node packages/cli/dist/src/cli.js status --openclaw-home ~/.openclaw --json",
      startedAt: "2026-03-31T12:59:00.000Z",
      endedAt: "2026-03-31T12:59:02.000Z",
      durationMs: 2000,
      exitCode: 0,
      signal: null,
      stdout: "{}",
      stderr: "",
      parsed: {
        brain: {
          activePackId: "pack-9e3c579b",
          routeFreshness: "updated",
        },
        brainStatus: {
          status: "ok",
          serveState: "serving_active_pack",
          usedLearnedRouteFn: true,
        },
        hook: {
          loadProof: "status_probe_ready",
        },
        currentTurnAttribution: {
          usedLearnedRouteFn: true,
        },
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
        selectedContextChars: 30,
        selectedContextBlockCount: 4,
        estimatedPromptTokens: 10,
        retrievalToolHopCount: 6,
        retrievalToolHopTurnCount: 3,
        feedbackEventCount: 2,
        nonApprovalFeedbackEventCount: 1,
        turnsWithFeedbackCount: 2,
        turnsWithNonApprovalFeedbackCount: 1,
        turnsWithNonApprovalFeedbackRate: 0.25,
        turnsWithSelectedContextCount: 4,
        turnsWithSelectedContextRate: 1,
        savingsByMode: [
          {
            mode: "no_brain",
            turnCount: 1,
            selectedContextBlockCount: 1,
            selectedContextChars: 5,
            estimatedPromptTokens: 2,
            retrievalToolHopCount: 0,
            retrievalToolHopTurnCount: 0,
            selectedContextCharsPerTurnMean: 5,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 2,
            turnsWithSelectedContextCount: 1,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 0,
            retrievalToolHopTurnRate: 0,
          },
          {
            mode: "vector_only",
            turnCount: 1,
            selectedContextBlockCount: 1,
            selectedContextChars: 9,
            estimatedPromptTokens: 3,
            retrievalToolHopCount: 2,
            retrievalToolHopTurnCount: 1,
            selectedContextCharsPerTurnMean: 9,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 3,
            turnsWithSelectedContextCount: 1,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 2,
            retrievalToolHopTurnRate: 1,
          },
          {
            mode: "graph_prior_only",
            turnCount: 1,
            selectedContextBlockCount: 1,
            selectedContextChars: 5,
            estimatedPromptTokens: 2,
            retrievalToolHopCount: 1,
            retrievalToolHopTurnCount: 1,
            selectedContextCharsPerTurnMean: 5,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 2,
            turnsWithSelectedContextCount: 1,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 1,
            retrievalToolHopTurnRate: 1,
          },
          {
            mode: "learned_route",
            turnCount: 1,
            selectedContextBlockCount: 1,
            selectedContextChars: 11,
            estimatedPromptTokens: 3,
            retrievalToolHopCount: 3,
            retrievalToolHopTurnCount: 1,
            selectedContextCharsPerTurnMean: 11,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 3,
            turnsWithSelectedContextCount: 1,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 3,
            retrievalToolHopTurnRate: 1,
          },
        ],
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
    expect(health.latestBundles.map((bundle: any) => bundle.kind)).toEqual(["operator-proof", "recorded-session-replay", "host-evidence"]);
    expect(formatHealthMarkdown(health)).toContain("runtime healthy: true");
    expect(formatHealthMarkdown(health)).toContain("serve state: serving_active_pack");
    expect(formatHealthMarkdown(health)).toContain("clip rate: 0.25");
    expect(formatHealthMarkdown(health)).toContain("replay context chars total");
    expect(formatHealthMarkdown(health)).toContain("learned_route: 11 chars, 1 blocks, 3 estimated prompt tokens");
    expect(formatHealthMarkdown(health)).toContain("proof minutes proxy");

    const aggregate = buildNightlyAggregate({ config, bundles, now, scanDurationMs: 42 });
    expect(aggregate.bundleTypeCounts.operatorProof).toBe(1);
    expect(aggregate.bundleTypeCounts.recordedSessionReplay).toBe(1);
    expect(aggregate.bundleTypeCounts.hostEvidence).toBe(1);
    expect(aggregate.replayMetrics.winnerModeCounts.learned_route).toBe(1);
    expect(aggregate.replayMetrics.selectedContextCharsTotal).toBe(30);
    expect(aggregate.replayMetrics.selectedContextBlocksTotal).toBe(4);
    expect(aggregate.replayMetrics.estimatedPromptTokensTotal).toBe(10);
    expect(aggregate.replayMetrics.retrievalToolHopCountTotal).toBe(6);
    expect(aggregate.replayMetrics.retrievalToolHopTurnCountTotal).toBe(3);
    expect(aggregate.replayMetrics.feedbackEventCountTotal).toBe(2);
    expect(aggregate.replayMetrics.nonApprovalFeedbackEventCountTotal).toBe(1);
    expect(aggregate.replayMetrics.turnsWithNonApprovalFeedbackCountTotal).toBe(1);
    expect(aggregate.replayMetrics.savingsByMode[3].mode).toBe("learned_route");
    expect(aggregate.replayMetrics.savingsByMode[3].estimatedPromptTokens).toBe(3);
    expect(aggregate.replayMetrics.savingsByMode[3].retrievalToolHopCount).toBe(3);
    expect(aggregate.operatorMetrics.stepMsTotal).toBe(1500);
    expect(aggregate.costProxy.bundleCount).toBe(3);
    expect(formatNightlyMarkdown(aggregate)).toContain("winner modes");
    expect(formatNightlyMarkdown(aggregate)).toContain("| learned_route | 11 | 1 | 3 | 3 | 1 | 1 | 1 | 1 |");
    expect(formatNightlyMarkdown(aggregate)).toContain("replay retrieval/tool-hop count total: 6");
    expect(formatNightlyMarkdown(aggregate)).toContain("replay turns with non-approval feedback total: 1");
    expect(formatNightlyMarkdown(aggregate)).toContain("proof minutes");
  });
});
