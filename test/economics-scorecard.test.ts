import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  PROOF_CRON_MANIFEST_LAYOUT,
  buildHealthSnapshot,
  buildNightlyAggregate,
  writeHealthOutputs,
  writeNightlyOutputs,
} from "../scripts/proof-cron.mjs";
import {
  ECONOMICS_SCORECARD_CONTRACT,
  ECONOMICS_SCORECARD_JSON_FILE,
  ECONOMICS_SCORECARD_MARKDOWN_FILE,
  buildEconomicsScorecardFromHealthSnapshot,
  buildEconomicsScorecardFromNightlyAggregate,
  buildEconomicsScorecardMarkdown,
  isEconomicsScorecard,
} from "../scripts/economics-scorecard.mjs";

function tempWorkspace() {
  return mkdtempSync(path.join(os.tmpdir(), "ocb-economics-scorecard-"));
}

function writeText(filePath: string, content: string) {
  writeFileSync(filePath, content, "utf8");
}

function writeJson(filePath: string, value: unknown) {
  writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function readText(filePath: string) {
  return readFileSync(filePath, "utf8");
}

function readJson(filePath: string) {
  return JSON.parse(readText(filePath));
}

function buildBundles() {
  return [
    {
      kind: "operator-proof",
      bundleId: "operator-proof-01",
      relativePath: "artifacts/operator-proof-01",
      canonicalAt: "2026-04-07T18:00:00.000Z",
      ageDays: 1,
      fileCount: 3,
      artifactBytes: 1200,
      validationOk: true,
      metrics: {
        totalStepDurationMs: 900,
        stepCount: 3,
      },
    },
    {
      kind: "recorded-session-replay",
      bundleId: "replay-proof-01",
      relativePath: "docs/evidence/2026-04-07/abc123/recorded-session-replay/replay-proof-01",
      canonicalAt: "2026-04-07T18:00:00.000Z",
      ageDays: 2,
      fileCount: 4,
      artifactBytes: 3400,
      validationOk: true,
      metrics: {
        winnerScore: 0.8,
        compileOkRate: 0.75,
        phraseHitRate: 0.5,
        learnedRouteTurnRate: 1,
        pricingTableVersion: "v1",
        pricingTablePath: "scripts/pricing-table.v1.json",
        replayFileBytes: 3400,
        totalTurns: 2,
        estimatedPromptCostUsd: 0.000005,
        estimatedCompletionCostUsd: 0.000012,
        estimatedTotalCostUsd: 0.000017,
        savingsByMode: [
          {
            mode: "learned_route",
            turnCount: 2,
            pricingTableVersion: "v1",
            pricingTablePath: "scripts/pricing-table.v1.json",
            selectedContextBlockCount: 1,
            selectedContextChars: 18,
            completionChars: 14,
            estimatedPromptTokens: 4,
            estimatedCompletionTokens: 3,
            estimatedPromptCostUsd: 0.000005,
            estimatedCompletionCostUsd: 0.000012,
            estimatedTotalCostUsd: 0.000017,
            retrievalToolHopCount: 3,
            retrievalToolHopTurnCount: 1,
            selectedContextCharsPerTurnMean: 9,
            selectedContextBlocksPerTurnMean: 0.5,
            estimatedPromptTokensPerTurnMean: 2,
            completionCharsObservedTurnCount: 1,
            completionCharsObservedRate: 1,
            completionCharsPerTurnMean: 7,
            estimatedCompletionTokensPerTurnMean: 1.5,
            estimatedPromptCostUsdPerTurnMean: 0.0000025,
            estimatedCompletionCostUsdPerTurnMean: 0.000006,
            estimatedTotalCostUsdPerTurnMean: 0.0000085,
            turnsWithSelectedContextCount: 2,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 1.5,
            retrievalToolHopTurnRate: 1,
          },
        ],
        retrievalToolHopCount: 3,
        retrievalToolHopTurnCount: 1,
        feedbackEventCount: 2,
        nonApprovalFeedbackEventCount: 1,
        turnsWithNonApprovalFeedbackCount: 1,
      },
    },
    {
      kind: "host-evidence",
      bundleId: "host-proof-01",
      relativePath: "artifacts/host-proof-01",
      canonicalAt: "2026-04-07T18:00:00.000Z",
      ageDays: 1,
      fileCount: 4,
      artifactBytes: 900,
      validationOk: true,
      metrics: {
        securityCriticalCount: 0,
        securityWarnCount: 1,
        gatewayReachable: true,
        workerHealthy: true,
        memoryFiles: 2,
        sessionCount: 1,
      },
    },
  ];
}

function buildStatusProbe() {
  return {
    command: "node packages/cli/dist/src/cli.js status --openclaw-home ~/.openclaw --json",
    startedAt: "2026-04-07T17:59:58.000Z",
    endedAt: "2026-04-07T18:00:00.000Z",
    durationMs: 2000,
    exitCode: 0,
    signal: null,
    parsed: {
      brainStatus: {
        status: "ok",
        serveState: "serving_active_pack",
        usedLearnedRouteFn: true,
      },
      brain: {
        activePackId: "pack-abc123",
        routeFreshness: "updated",
      },
      hook: {
        loadProof: "status_probe_ready",
      },
      passiveLearning: {
        watch: {
          state: "healthy",
          lastHeartbeatAt: "2026-04-07T17:59:30.000Z",
          intervalSeconds: 30,
          proofState: "self_proving",
          teacherArtifactCount: 3,
        },
      },
      workerHealthy: true,
      workerMode: "child",
    },
  };
}

describe("economics scorecard contract", () => {
  it("builds bounded health and nightly scorecards with explicit measured/derived/proxy labels", () => {
    const bundles = buildBundles();
    const config = {
      healthFreshnessDays: 7,
      freshnessThresholdDays: 21,
    };
    const now = new Date("2026-04-07T18:00:00.000Z");

    const healthSnapshot = buildHealthSnapshot({
      config,
      now,
      scanDurationMs: 42,
      bundles,
      statusProbe: buildStatusProbe(),
    });
    const healthScorecard = buildEconomicsScorecardFromHealthSnapshot(healthSnapshot);

    expect(isEconomicsScorecard(healthScorecard)).toBe(true);
    expect(healthScorecard.contract).toBe(ECONOMICS_SCORECARD_CONTRACT);
    expect(healthScorecard.scope).toBe("health");
    expect(healthScorecard.boundedness.sectionLimit).toBeGreaterThanOrEqual(healthScorecard.measured.length);
    expect(healthScorecard.measured.every((entry: any) => entry.kind === "measured")).toBe(true);
    expect(healthScorecard.derived.every((entry: any) => entry.kind === "derived")).toBe(true);
    expect(healthScorecard.proxy.every((entry: any) => entry.kind === "proxy")).toBe(true);
    expect(healthScorecard.measured.map((entry: any) => entry.metric)).toEqual(expect.arrayContaining([
      "status_probe_duration_ms",
      "scan_duration_ms",
      "bundle_count",
      "artifact_bytes_scanned",
    ]));
    expect(healthScorecard.derived.map((entry: any) => entry.metric)).toEqual(expect.arrayContaining([
      "operator_step_ms_total",
      "replay_context_chars_total",
      "replay_estimated_prompt_tokens_total",
      "replay_retrieval_tool_hop_count_total",
    ]));
    expect(healthScorecard.proxy.map((entry: any) => entry.metric)).toEqual(expect.arrayContaining([
      "proof_minutes_proxy",
      "replay_prompt_cost_usd_proxy",
      "replay_completion_cost_usd_proxy",
      "replay_total_cost_usd_proxy",
    ]));
    expect(buildEconomicsScorecardMarkdown(healthScorecard)).toContain("labels: measured / derived / proxy");
    expect(buildEconomicsScorecardMarkdown(healthScorecard)).toContain("## Measured");
    expect(buildEconomicsScorecardMarkdown(healthScorecard)).toContain("status_probe_duration_ms");
    expect(buildEconomicsScorecardMarkdown(healthScorecard)).toContain("proof_minutes_proxy");

    const nightlyAggregate = buildNightlyAggregate({
      config,
      now,
      scanDurationMs: 42,
      bundles,
    });
    const nightlyScorecard = buildEconomicsScorecardFromNightlyAggregate(nightlyAggregate);

    expect(isEconomicsScorecard(nightlyScorecard)).toBe(true);
    expect(nightlyScorecard.scope).toBe("nightly");
    expect(nightlyScorecard.measured.map((entry: any) => entry.metric)).toEqual(expect.arrayContaining([
      "bundle_count",
      "operator_proof_count",
      "recorded_session_replay_count",
      "host_evidence_count",
    ]));
    expect(nightlyScorecard.derived.map((entry: any) => entry.metric)).toEqual(expect.arrayContaining([
      "replay_winner_score_mean",
      "replay_context_chars_total",
      "replay_estimated_prompt_tokens_total",
    ]));
    expect(nightlyScorecard.proxy.map((entry: any) => entry.metric)).toEqual(expect.arrayContaining([
      "proof_minutes_proxy",
      "replay_total_cost_usd_proxy",
    ]));
    expect(buildEconomicsScorecardMarkdown(nightlyScorecard)).toContain("## Proxy");
    expect(buildEconomicsScorecardMarkdown(nightlyScorecard)).toContain("recorded_session_replay_count");
  });

  it("writes the bounded economics artifacts into the proof-cron health and nightly surfaces", () => {
    const workspaceRoot = tempWorkspace();
    const outputRoot = path.join(workspaceRoot, "artifacts");
    const healthOutputDir = path.join(outputRoot, "openclawbrain-proof-cron", "health-snapshot");
    const nightlyOutputDir = path.join(outputRoot, "openclawbrain-proof-cron", "nightly-aggregate");
    rmSync(outputRoot, { recursive: true, force: true });

    const bundles = buildBundles();
    const config = {
      healthFreshnessDays: 7,
      freshnessThresholdDays: 21,
    };
    const now = new Date("2026-04-07T18:00:00.000Z");
    const healthSnapshot = buildHealthSnapshot({
      config,
      now,
      scanDurationMs: 42,
      bundles,
      statusProbe: buildStatusProbe(),
    });
    const nightlyAggregate = buildNightlyAggregate({
      config,
      now,
      scanDurationMs: 42,
      bundles,
    });

    writeHealthOutputs(healthOutputDir, healthSnapshot, buildStatusProbe(), bundles, workspaceRoot);
    writeNightlyOutputs(nightlyOutputDir, nightlyAggregate, bundles, workspaceRoot);

    const healthEconomics = readJson(path.join(healthOutputDir, ECONOMICS_SCORECARD_JSON_FILE));
    const nightlyEconomics = readJson(path.join(nightlyOutputDir, ECONOMICS_SCORECARD_JSON_FILE));
    const healthEconomicsMarkdown = readText(path.join(healthOutputDir, ECONOMICS_SCORECARD_MARKDOWN_FILE));
    const nightlyEconomicsMarkdown = readText(path.join(nightlyOutputDir, ECONOMICS_SCORECARD_MARKDOWN_FILE));
    const healthManifest = readJson(path.join(healthOutputDir, PROOF_CRON_MANIFEST_LAYOUT.manifest));
    const nightlyManifest = readJson(path.join(nightlyOutputDir, PROOF_CRON_MANIFEST_LAYOUT.manifest));

    expect(healthEconomics.contract).toBe(ECONOMICS_SCORECARD_CONTRACT);
    expect(healthEconomics.scope).toBe("health");
    expect(healthEconomicsMarkdown).toContain("## Measured");
    expect(healthEconomicsMarkdown).toContain("## Proxy");
    expect(healthEconomicsMarkdown).toContain("labels: measured / derived / proxy");
    expect(healthManifest.output.supporting.map((item: any) => item.path)).toEqual(expect.arrayContaining([
      "summary.md",
      ECONOMICS_SCORECARD_JSON_FILE,
      ECONOMICS_SCORECARD_MARKDOWN_FILE,
    ]));
    expect(healthManifest.output.supporting.find((item: any) => item.path === ECONOMICS_SCORECARD_JSON_FILE)?.contract).toBe(ECONOMICS_SCORECARD_CONTRACT);
    expect(healthManifest.output.supporting.find((item: any) => item.path === ECONOMICS_SCORECARD_MARKDOWN_FILE)?.contract).toBe(ECONOMICS_SCORECARD_CONTRACT);

    expect(nightlyEconomics.contract).toBe(ECONOMICS_SCORECARD_CONTRACT);
    expect(nightlyEconomics.scope).toBe("nightly");
    expect(nightlyEconomicsMarkdown).toContain("## Derived");
    expect(nightlyEconomicsMarkdown).toContain("replay_total_cost_usd_proxy");
    expect(nightlyManifest.output.supporting.map((item: any) => item.path)).toEqual(expect.arrayContaining([
      "summary.md",
      ECONOMICS_SCORECARD_JSON_FILE,
      ECONOMICS_SCORECARD_MARKDOWN_FILE,
    ]));
    expect(nightlyManifest.output.supporting.find((item: any) => item.path === ECONOMICS_SCORECARD_JSON_FILE)?.contract).toBe(ECONOMICS_SCORECARD_CONTRACT);
    expect(nightlyManifest.output.supporting.find((item: any) => item.path === ECONOMICS_SCORECARD_MARKDOWN_FILE)?.contract).toBe(ECONOMICS_SCORECARD_CONTRACT);
  });
});
