import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { describe, expect, it } from "vitest";
import { summarizeOperatorHealth } from "../src/live-runtime-audit.js";
// @ts-ignore runtime script module has no checked-in typed surface yet
import {
  PROOF_CRON_MANIFEST_LAYOUT,
  buildHealthSnapshot,
  buildNightlyAggregate,
  collectBundleCandidates,
  formatHealthMarkdown,
  formatNightlyMarkdown,
  loadConfig,
  summarizeScan,
  writeHealthOutputs,
  writeNightlyOutputs,
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

function readText(rootDir: string, relativePath: string) {
  return readFileSync(path.join(rootDir, relativePath), "utf8");
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
    writeJson(path.join(replayRoot, "manifest.json"), {
      contract: "recorded_session_replay_proof_manifest.v1",
      traceId: "trace-a",
      source: "sanitized_recorded_session",
      recordedAt: "2026-03-26T00:00:00.000Z",
      generatedAt: "2026-03-26T00:10:00.000Z",
      hashAlgorithm: "sha256",
      modeOrder: ["no_brain", "vector_only", "graph_prior_only", "learned_route"],
      contracts: {
        trace: "recorded_session_trace.v1",
        fixture: "recorded_session_replay_fixture.v1",
        bundle: "recorded_session_replay_bundle.v1",
        environment: "recorded_session_replay_environment.v1",
        summaryTables: "recorded_session_replay_summary_tables.v1",
        coverageSnapshot: "recorded_session_replay_coverage_snapshot.v1",
        hardeningSnapshot: "recorded_session_replay_hardening_snapshot.v1",
        hashes: "recorded_session_replay_hashes.v1",
      },
      hashes: {
        traceHash: "sha256-trace",
        fixtureHash: "sha256-fixture",
        scoreHash: "sha256-score",
        bundleHash: "sha256-bundle",
      },
      files: {
        trace: "trace.json",
        fixture: "fixture.json",
        bundle: "bundle.json",
        environment: "environment.json",
        summary: "summary.md",
        summaryTables: "summary-tables.json",
        coverageSnapshot: "coverage-snapshot.json",
        hardeningSnapshot: "hardening-snapshot.json",
        hashes: "hashes.json",
        modes: [
          { mode: "no_brain", path: "modes/no_brain.json" },
          { mode: "vector_only", path: "modes/vector_only.json" },
          { mode: "graph_prior_only", path: "modes/graph_prior_only.json" },
          { mode: "learned_route", path: "modes/learned_route.json" },
        ],
      },
    });
    writeJson(path.join(replayRoot, "fixture.json"), {
      contract: "recorded_session_replay_fixture.v1",
      traceId: "trace-a",
      fixtureHash: "sha256-fixture",
      turns: [],
    });
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
              completionTexts: ["alpha"],
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
              completionTexts: ["beta beta"],
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
              completionTexts: ["gamma"],
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
              completionTexts: ["delta delta"],
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
    writeJson(path.join(replayRoot, "hashes.json"), {
      contract: "recorded_session_replay_hashes.v1",
      algorithm: "sha256",
      semantic: {
        traceHash: "sha256-trace",
        fixtureHash: "sha256-fixture",
        scoreHash: "sha256-score",
        bundleHash: "sha256-bundle",
      },
      files: [
        { path: "trace.json", digest: "sha256-trace-file" },
        { path: "fixture.json", digest: "sha256-fixture-file" },
        { path: "bundle.json", digest: "sha256-bundle-file" },
        { path: "manifest.json", digest: "sha256-manifest-file" },
      ],
    });
    writeJson(path.join(replayRoot, "validation-report.json"), { ok: true, verifiedFileCount: 6 });

    const replayLaneRoot = path.join(docsEvidenceRoot, "2026-03-30", "def456", "recorded-session-replay", "_lane");
    writeText(path.join(replayLaneRoot, "summary.md"), "# replay lane proof\n");
    writeJson(path.join(replayLaneRoot, "closeout.json"), {
      sourceManifest: {
        manifestId: "replay-lane-20260330",
        manifestContract: "openclawbrain_replay_manifest_skeleton_set.v1",
        manifestDigest: "sha256-replay-lane-manifest",
      },
      requestedTraceCount: 20,
      successfulTraceCount: 18,
      failedTraceCount: 2,
      verdict: {
        verdict: "success_and_proven",
        severity: "none",
      },
      files: [],
      traceHashes: [
        { bundleHash: "sha256-bundle-a", scoreHash: "sha256-score-a" },
        { bundleHash: "sha256-bundle-b", scoreHash: "sha256-score-b" },
      ],
    });
    writeJson(path.join(replayLaneRoot, "index.json"), {
      requestedTraceCount: 20,
      successfulTraceCount: 18,
      failedTraceCount: 2,
    });
    writeJson(path.join(replayLaneRoot, "summary-tables.json"), {
      requestedTraceCount: 20,
      successfulTraceCount: 18,
      failedTraceCount: 2,
      traces: [
        { bundleHash: "sha256-bundle-a", scoreHash: "sha256-score-a" },
        { bundleHash: "sha256-bundle-b", scoreHash: "sha256-score-b" },
      ],
    });
    writeJson(path.join(replayLaneRoot, "pairwise-deltas.json"), {});
    writeJson(path.join(replayLaneRoot, "win-rate-matrix.json"), {});
    writeText(path.join(replayLaneRoot, "worked-traces.md"), "# worked traces\n");
    writeJson(path.join(replayLaneRoot, "generation-report.json"), {
      requestedTraceCount: 20,
      successfulTraceCount: 18,
      failedTraceCount: 2,
    });

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
      contextFeedback: {
        verdictCounts: {
          helpful: 3,
          irrelevant: 1,
          harmful: 0,
        },
        coverage: {
          routeTraceCount: 5,
          supervisedTraceCount: 4,
        },
        latest: {
          agentIdentity: {
            agentId: "operator",
            lane: "proof-summary",
          },
        },
      },
      attributionTruth: {
        primaryState: "matched",
        counts: {
          observationCount: 5,
          evaluatedCount: 4,
          completedWithoutEvaluationCount: 1,
          readyCount: 1,
          delayedCount: 0,
          budgetDeferredCount: 0,
        },
      },
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

    expect(bundleKinds).toEqual(["host-evidence", "operator-proof", "recorded-session-replay", "recorded-session-replay-lane"]);

    const bundles = summarizeScan(candidates, new Date("2026-03-31T13:00:00.000Z"), workspaceRoot);
    expect(bundles.find((bundle: any) => bundle.kind === "operator-proof")?.metrics.totalStepDurationMs).toBe(1500);
    const replayBundle = bundles.find((bundle: any) => bundle.kind === "recorded-session-replay");
    expect(replayBundle?.metrics.winnerMode).toBe("learned_route");
    expect(replayBundle?.metrics.selectedContextChars).toBe(30);
    expect(replayBundle?.metrics.selectedContextBlockCount).toBe(4);
    expect(replayBundle?.metrics.estimatedPromptTokens).toBe(10);
    expect(replayBundle?.metrics.completionChars).toBe(30);
    expect(replayBundle?.metrics.estimatedCompletionTokens).toBe(10);
    expect(replayBundle?.metrics.estimatedPromptCostUsd).toBeCloseTo(0.000014, 12);
    expect(replayBundle?.metrics.estimatedCompletionCostUsd).toBeCloseTo(0.00005, 12);
    expect(replayBundle?.metrics.estimatedTotalCostUsd).toBeCloseTo(0.000064, 12);
    expect(replayBundle?.metrics.pricingTableVersion).toBe("v1");
    expect(replayBundle?.metrics.retrievalToolHopCount).toBe(6);
    expect(replayBundle?.metrics.retrievalToolHopTurnCount).toBe(3);
    expect(replayBundle?.metrics.feedbackEventCount).toBe(2);
    expect(replayBundle?.metrics.nonApprovalFeedbackEventCount).toBe(1);
    expect(replayBundle?.metrics.turnsWithNonApprovalFeedbackCount).toBe(1);
    const replayLaneBundle = bundles.find((bundle: any) => bundle.kind === "recorded-session-replay-lane");
    expect(replayLaneBundle?.bundleId).toBe("replay-lane-20260330");
    expect(replayLaneBundle?.metrics.requestedTraceCount).toBe(20);
    expect(replayLaneBundle?.metrics.successfulTraceCount).toBe(18);
    const hostBundle = bundles.find((bundle: any) => bundle.kind === "host-evidence");
    expect(hostBundle?.feedbackTruth).toMatchObject({
      visible: true,
      helpfulCount: 3,
      supervisedTraceCount: 4,
      routeTraceCount: 5,
      latestAgentIdentity: "operator/proof-summary",
    });
    expect(hostBundle?.attributionCoverageTruth).toMatchObject({
      visible: true,
      evaluatedCount: 4,
      observationCount: 5,
      completedWithoutEvaluationCount: 1,
      readyCount: 1,
    });
    expect(replayBundle?.metrics.savingsByMode).toEqual([
      {
        mode: "no_brain",
        turnCount: 1,
        pricingTableVersion: "v1",
        pricingTablePath: "scripts/pricing-table.v1.json",
        selectedContextBlockCount: 1,
        selectedContextChars: 5,
        completionChars: 5,
        estimatedPromptTokens: 2,
        estimatedCompletionTokens: 2,
        estimatedPromptCostUsd: 0.000003,
        estimatedCompletionCostUsd: 0.00001,
        estimatedTotalCostUsd: 0.000013,
        retrievalToolHopCount: 0,
        retrievalToolHopTurnCount: 0,
        selectedContextCharsPerTurnMean: 5,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 2,
        completionCharsObservedTurnCount: 1,
        completionCharsObservedRate: 1,
        completionCharsPerTurnMean: 5,
        estimatedCompletionTokensPerTurnMean: 2,
        estimatedPromptCostUsdPerTurnMean: 0.000003,
        estimatedCompletionCostUsdPerTurnMean: 0.00001,
        estimatedTotalCostUsdPerTurnMean: 0.000013,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 0,
        retrievalToolHopTurnRate: 0,
      },
      {
        mode: "vector_only",
        turnCount: 1,
        pricingTableVersion: "v1",
        pricingTablePath: "scripts/pricing-table.v1.json",
        selectedContextBlockCount: 1,
        selectedContextChars: 9,
        completionChars: 9,
        estimatedPromptTokens: 3,
        estimatedCompletionTokens: 3,
        estimatedPromptCostUsd: 0.000004,
        estimatedCompletionCostUsd: 0.000015,
        estimatedTotalCostUsd: 0.000019,
        retrievalToolHopCount: 2,
        retrievalToolHopTurnCount: 1,
        selectedContextCharsPerTurnMean: 9,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 3,
        completionCharsObservedTurnCount: 1,
        completionCharsObservedRate: 1,
        completionCharsPerTurnMean: 9,
        estimatedCompletionTokensPerTurnMean: 3,
        estimatedPromptCostUsdPerTurnMean: 0.000004,
        estimatedCompletionCostUsdPerTurnMean: 0.000015,
        estimatedTotalCostUsdPerTurnMean: 0.000019,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 2,
        retrievalToolHopTurnRate: 1,
      },
      {
        mode: "graph_prior_only",
        turnCount: 1,
        pricingTableVersion: "v1",
        pricingTablePath: "scripts/pricing-table.v1.json",
        selectedContextBlockCount: 1,
        selectedContextChars: 5,
        completionChars: 5,
        estimatedPromptTokens: 2,
        estimatedCompletionTokens: 2,
        estimatedPromptCostUsd: 0.000003,
        estimatedCompletionCostUsd: 0.00001,
        estimatedTotalCostUsd: 0.000013,
        retrievalToolHopCount: 1,
        retrievalToolHopTurnCount: 1,
        selectedContextCharsPerTurnMean: 5,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 2,
        completionCharsObservedTurnCount: 1,
        completionCharsObservedRate: 1,
        completionCharsPerTurnMean: 5,
        estimatedCompletionTokensPerTurnMean: 2,
        estimatedPromptCostUsdPerTurnMean: 0.000003,
        estimatedCompletionCostUsdPerTurnMean: 0.00001,
        estimatedTotalCostUsdPerTurnMean: 0.000013,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 1,
        retrievalToolHopTurnRate: 1,
      },
      {
        mode: "learned_route",
        turnCount: 1,
        pricingTableVersion: "v1",
        pricingTablePath: "scripts/pricing-table.v1.json",
        selectedContextBlockCount: 1,
        selectedContextChars: 11,
        completionChars: 11,
        estimatedPromptTokens: 3,
        estimatedCompletionTokens: 3,
        estimatedPromptCostUsd: 0.000004,
        estimatedCompletionCostUsd: 0.000015,
        estimatedTotalCostUsd: 0.000019,
        retrievalToolHopCount: 3,
        retrievalToolHopTurnCount: 1,
        selectedContextCharsPerTurnMean: 11,
        selectedContextBlocksPerTurnMean: 1,
        estimatedPromptTokensPerTurnMean: 3,
        completionCharsObservedTurnCount: 1,
        completionCharsObservedRate: 1,
        completionCharsPerTurnMean: 11,
        estimatedCompletionTokensPerTurnMean: 3,
        estimatedPromptCostUsdPerTurnMean: 0.000004,
        estimatedCompletionCostUsdPerTurnMean: 0.000015,
        estimatedTotalCostUsdPerTurnMean: 0.000019,
        turnsWithSelectedContextCount: 1,
        turnsWithSelectedContextRate: 1,
        retrievalToolHopPerTurnMean: 3,
        retrievalToolHopTurnRate: 1,
      },
    ]);
    expect(bundles.find((bundle: any) => bundle.kind === "host-evidence")?.metrics.securityCriticalCount).toBe(1);

    const config = {
      healthFreshnessDays: 7,
      freshnessThresholdDays: 21,
    };
    const now = new Date("2026-03-31T13:00:00.000Z");
    const nightlyAggregate = buildNightlyAggregate({
      config,
      now,
      scanDurationMs: 42,
      bundles,
    });
    const nightlyOutputDir = path.join(workspaceRoot, "artifacts", "openclawbrain-proof-cron", "nightly-aggregate");
    writeNightlyOutputs(nightlyOutputDir, nightlyAggregate, bundles, workspaceRoot);

    const nightlyManifest = JSON.parse(readText(nightlyOutputDir, PROOF_CRON_MANIFEST_LAYOUT.manifest));
    const nightlyReplayManifests = JSON.parse(readText(nightlyOutputDir, PROOF_CRON_MANIFEST_LAYOUT.replayManifests));
    const nightlySmoke = JSON.parse(readText(nightlyOutputDir, PROOF_CRON_MANIFEST_LAYOUT.smoke));
    const nightlySummary = readText(nightlyOutputDir, "summary.md");

    expect(nightlyManifest.contract).toBe("openclawbrain_proof_manifest_skeleton.v1");
    expect(nightlyManifest.runKind).toBe("nightly");
    expect(nightlyManifest.replayInputs.count).toBe(1);
    expect(nightlyManifest.replayInputs.items).toEqual([
      {
        traceId: "trace-a",
        proofBundleRelativePath: "openclawbrain/docs/evidence/2026-03-26/abc123/recorded-session-replay/trace-a",
        fixtureHash: "sha256-fixture",
        bundleHash: "sha256-bundle",
        scoreHash: "sha256-score",
        proofManifestDigest: "sha256-manifest-file",
      },
    ]);
    expect(nightlyReplayManifests.contract).toBe("openclawbrain_replay_manifest_skeleton_set.v1");
    expect(nightlyReplayManifests.linkageSummary.fixtureToReplayLinkedCount).toBe(1);
    expect(nightlyReplayManifests.linkageSummary.replayToProofManifestLinkedCount).toBe(1);
    expect(nightlyReplayManifests.linkageSummary.manifestToHashLedgerLinkedCount).toBe(1);
    expect(nightlyReplayManifests.items[0].releaseCloseout.state).toBe("unlinked");
    expect(nightlySmoke.contract).toBe("openclawbrain_proof_manifest_smoke.v1");
    expect(nightlySmoke.output.primary.path).toBe("aggregate.json");
    expect(nightlySmoke.replayInputs.allReplayHashesLinked).toBe(true);
    expect(nightlyAggregate.feedbackTruth).toMatchObject({
      visible: true,
      helpfulCount: 3,
      supervisedTraceCount: 4,
      routeTraceCount: 5,
    });
    expect(nightlyAggregate.attributionCoverageTruth).toMatchObject({
      visible: true,
      evaluatedCount: 4,
      observationCount: 5,
      completedWithoutEvaluationCount: 1,
      readyCount: 1,
    });
    expect(nightlyAggregate.replayFreshnessTruth).toMatchObject({
      visible: true,
      bundleId: "replay-lane-20260330",
      requestedTraceCount: 20,
      successfulTraceCount: 18,
      freshness: "fresh",
    });
    expect(nightlySummary).toContain("## Thin truth");
    expect(nightlySummary).toContain("feedback: helpful=3 irrelevant=1 harmful=0 coverage=4/5 latest=operator/proof-summary (source=latest_host_evidence)");
    expect(nightlySummary).toContain("attribution coverage: evaluated=4/5 completedWithoutEval=1 ready=1 delayed=0 budgetDeferred=0 (source=latest_host_evidence)");
    expect(nightlySummary).toContain("replay freshness: latest=replay-lane-20260330 age=1.54d band=fresh traces=18/20");

    const statusProbe = {
      command: "node packages/cli/dist/src/cli.js status --openclaw-home ~/.openclaw --json",
      startedAt: "2026-03-31T12:59:00.000Z",
      endedAt: "2026-03-31T12:59:02.000Z",
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
          activePackId: "pack-9e3c579b",
          routeFreshness: "updated",
        },
        hook: {
          loadProof: "status_probe_ready",
        },
        passiveLearning: {
          watch: {
            state: "healthy",
            lastHeartbeatAt: "2026-03-31T12:58:45.000Z",
            intervalSeconds: 30,
            proofState: "self_proving",
            teacherArtifactCount: 3,
          },
        },
        workerHealthy: true,
        workerMode: "child",
      },
    };
    const healthSnapshot = buildHealthSnapshot({
      config,
      now,
      scanDurationMs: 42,
      bundles,
      statusProbe,
    });
    const healthOutputDir = path.join(workspaceRoot, "artifacts", "openclawbrain-proof-cron", "health-snapshot");
    writeHealthOutputs(healthOutputDir, healthSnapshot, statusProbe, bundles, workspaceRoot);

    const healthManifest = JSON.parse(readText(healthOutputDir, PROOF_CRON_MANIFEST_LAYOUT.manifest));
    const healthSmoke = JSON.parse(readText(healthOutputDir, PROOF_CRON_MANIFEST_LAYOUT.smoke));
    const healthSummary = readText(healthOutputDir, "summary.md");
    expect(healthManifest.runKind).toBe("health");
    expect(healthManifest.output.primary.path).toBe("snapshot.json");
    expect(healthManifest.replayInputs.linkageSummary.fixtureToReplayLinkedCount).toBe(1);
    expect(healthSmoke.output.primary.path).toBe("snapshot.json");
    expect(healthSmoke.replayInputs.allReplayHashesLinked).toBe(true);
    expect(healthSnapshot.feedbackTruth).toMatchObject({
      visible: true,
      helpfulCount: 3,
      supervisedTraceCount: 4,
      routeTraceCount: 5,
    });
    expect(healthSnapshot.attributionCoverageTruth).toMatchObject({
      visible: true,
      evaluatedCount: 4,
      observationCount: 5,
      completedWithoutEvaluationCount: 1,
      readyCount: 1,
    });
    expect(healthSnapshot.replayFreshnessTruth).toMatchObject({
      visible: true,
      bundleId: "replay-lane-20260330",
      requestedTraceCount: 20,
      successfulTraceCount: 18,
      freshness: "fresh",
    });
    expect(healthSummary).toContain("## Thin truth");
    expect(healthSummary).toContain("feedback: helpful=3 irrelevant=1 harmful=0 coverage=4/5 latest=operator/proof-summary (source=latest_host_evidence)");
    expect(healthSummary).toContain("attribution coverage: evaluated=4/5 completedWithoutEval=1 ready=1 delayed=0 budgetDeferred=0 (source=latest_host_evidence)");
    expect(healthSummary).toContain("replay freshness: latest=replay-lane-20260330 age=1.54d band=fresh traces=18/20");
  });

  it("marks unknown host metrics as unknown instead of collapsing them to zero", () => {
    const now = new Date("2026-03-31T13:00:00.000Z");
    const config = {
      healthFreshnessDays: 7,
      freshnessThresholdDays: 21,
    };
    const aggregate = buildNightlyAggregate({
      config,
      now,
      scanDurationMs: 42,
      bundles: [
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
            securityCriticalCount: null,
            securityWarnCount: null,
            gatewayReachable: null,
            workerHealthy: null,
            memoryFiles: null,
            sessionCount: null,
          },
        },
      ],
    });

    expect(aggregate.hostMetrics.gatewayReachableCount).toBe(0);
    expect(aggregate.hostMetrics.gatewayReachableKnownCount).toBe(0);
    expect(aggregate.hostMetrics.gatewayReachableUnknownCount).toBe(1);
    expect(aggregate.hostMetrics.workerHealthyKnownCount).toBe(0);
    expect(aggregate.hostMetrics.memoryFilesTotal).toBeNull();
    expect(aggregate.hostMetrics.sessionCountTotal).toBeNull();

    const markdown = formatNightlyMarkdown(aggregate);
    expect(markdown).toContain("gateway reachable bundles: unknown (1/1 bundles missing metric)");
    expect(markdown).toContain("worker healthy bundles: unknown (1/1 bundles missing metric)");
    expect(markdown).toContain("memory files total: unknown (1/1 bundles missing metric)");
    expect(markdown).toContain("session count total: unknown (1/1 bundles missing metric)");
  });

  it("surfaces stale watcher state explicitly in the health markdown", () => {
    const health = buildHealthSnapshot({
      config: {
        healthFreshnessDays: 7,
        freshnessThresholdDays: 21,
      },
      now: new Date("2026-04-02T11:30:00.000Z"),
      scanDurationMs: 42,
      bundles: [],
      statusProbe: {
        command: "openclawbrain status --json",
        startedAt: "2026-04-02T11:29:59.000Z",
        endedAt: "2026-04-02T11:30:00.000Z",
        durationMs: 1000,
        exitCode: 0,
        signal: null,
        parsed: {
          brainStatus: {
            status: "ok",
            serveState: "serving_active_pack",
            usedLearnedRouteFn: true,
          },
          brain: {
            activePackId: "pack-abc",
          },
          hook: {
            loadProof: "status_probe_ready",
          },
          passiveLearning: {
            watchState: "stale_snapshot",
            lastWatchHeartbeatAt: "2026-04-02T11:26:18.163Z",
            watchIntervalSeconds: 30,
            proofState: "self_proving",
            teacherArtifactCount: 735,
          },
          workerHealthy: null,
          workerMode: null,
        },
      },
    });

    const markdown = formatHealthMarkdown(health);
    expect(health.effectivenessReadout).toMatchObject({
      helping: "unproven",
    });
    expect(markdown).toContain("helping: unproven");
    expect(markdown).toContain("watch state: stale_snapshot");
    expect(markdown).toContain("watch heartbeat: 2026-04-02T11:26:18.163Z");
    expect(markdown).toContain("teacher artifacts: 735");
    expect(markdown).toContain("operator health: stale");
    expect(markdown).toContain("operator health flags: partial=false, unknown=false, stale=true");
    expect(markdown).toContain("watcher state is stale_snapshot; background-learning truth is stale");
    expect(markdown).toContain("worker health is unknown in the live status surface");
  });

  it("consumes the structured lagging watch object without overstating health", () => {
    const health = buildHealthSnapshot({
      config: {
        healthFreshnessDays: 7,
        freshnessThresholdDays: 21,
      },
      now: new Date("2026-04-02T11:30:00.000Z"),
      scanDurationMs: 42,
      bundles: [],
      statusProbe: {
        command: "openclawbrain status --json",
        startedAt: "2026-04-02T11:29:59.000Z",
        endedAt: "2026-04-02T11:30:00.000Z",
        durationMs: 1000,
        exitCode: 0,
        signal: null,
        parsed: {
          brainStatus: {
            status: "ok",
            serveState: "serving_active_pack",
            usedLearnedRouteFn: true,
          },
          brain: {
            activePackId: "pack-abc",
          },
          hook: {
            loadProof: "status_probe_ready",
          },
          passiveLearning: {
            watch: {
              state: "lagging",
              detail: "watch heartbeat missed the healthy window but has not crossed the stale snapshot threshold",
              lastHeartbeatAt: "2026-04-02T11:28:44.060Z",
              lagSeconds: 75.94,
              intervalSeconds: 30,
              healthyWithinSeconds: 75,
              staleAfterSeconds: 105,
            },
            proofState: "self_proving",
            teacherArtifactCount: 735,
          },
          workerHealthy: null,
          workerMode: null,
        },
      },
    });

    expect(health.watch).toMatchObject({
      state: "lagging",
      lastHeartbeatAt: "2026-04-02T11:28:44.060Z",
      lagSeconds: 75.94,
      intervalSeconds: 30,
      healthyWithinSeconds: 75,
      staleAfterSeconds: 105,
      proofState: "self_proving",
      teacherArtifactCount: 735,
    });

    const markdown = formatHealthMarkdown(health);
    expect(markdown).toContain("watch state: lagging");
    expect(markdown).toContain("watcher state is lagging; do not treat this as a fully healthy background-learning surface");
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
  function buildThinReadoutStatusProbe(params: {
    contextFeedback?: any;
    learningAttribution?: any;
    workerHealthy?: boolean | null;
    workerMode?: string | null;
    workerStatus?: string | null;
    watchState?: string | null;
  } = {}) {
    const {
      contextFeedback = null,
      learningAttribution = {
        quality: "exact_only",
        nonZeroObservationCount: 2,
        exactMatchCount: 2,
        heuristicMatchCount: 0,
        unmatchedCount: 0,
        ambiguousCount: 0,
      },
      workerHealthy = true,
      workerMode = "child",
      workerStatus = "running",
      watchState = "watching",
    } = params;

    return {
      command: "node packages/cli/dist/src/cli.js status --openclaw-home ~/.openclaw --json",
      startedAt: "2026-04-02T11:59:58.000Z",
      endedAt: "2026-04-02T12:00:00.000Z",
      durationMs: 2000,
      exitCode: 0,
      signal: null,
      stdout: "{}",
      stderr: "",
      parsed: {
        brain: {
          activePackId: "pack-live",
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
        passiveLearning: {
          watch: {
            state: watchState,
            lastHeartbeatAt: "2026-04-02T11:59:45.000Z",
            intervalSeconds: 30,
            proofState: "self_proving",
            teacherArtifactCount: 3,
          },
        },
        workerHealthy,
        workerMode,
        workerStatus,
        ...(learningAttribution === null ? {} : { learningAttribution }),
        ...(contextFeedback === null ? {} : { contextFeedback }),
      },
    };
  }

  function buildThinReadoutBundles(params: {
    replayAgeDays?: number;
    replayValidationOk?: boolean;
  } = {}) {
    const {
      replayAgeDays = 1,
      replayValidationOk = true,
    } = params;

    return [
      {
        kind: "operator-proof",
        bundleId: "operator-proof-live",
        relativePath: "artifacts/operator-proof-live",
        canonicalAt: "2026-04-02T11:00:00.000Z",
        ageDays: 0,
        fileCount: 5,
        artifactBytes: 2000,
        validationOk: true,
        metrics: {
          totalStepDurationMs: 1200,
          stepCount: 2,
          verdict: "success_and_proven",
        },
      },
      {
        kind: "recorded-session-replay",
        bundleId: "trace-live",
        relativePath: "docs/evidence/2026-04-01/demo/recorded-session-replay/trace-live",
        canonicalAt: "2026-04-01T12:00:00.000Z",
        ageDays: replayAgeDays,
        fileCount: 6,
        artifactBytes: 3000,
        validationOk: replayValidationOk,
        metrics: {
          winnerMode: "learned_route",
          compileOkRate: 0.9,
          phraseHitRate: 0.8,
        },
      },
      {
        kind: "host-evidence",
        bundleId: "host-proof-live",
        relativePath: "docs/evidence/2026-04-01/demo/host-proof",
        canonicalAt: "2026-04-01T12:00:00.000Z",
        ageDays: 1,
        fileCount: 4,
        artifactBytes: 1000,
        validationOk: true,
        metrics: {
          workerHealthy: true,
          gatewayReachable: true,
        },
      },
    ];
  }

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
        completionChars: 30,
        estimatedCompletionTokens: 10,
        estimatedPromptCostUsd: 0.000013,
        estimatedCompletionCostUsd: 0.00005,
        estimatedTotalCostUsd: 0.000063,
        pricingTableVersion: "v1",
        pricingTablePath: "scripts/pricing-table.v1.json",
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
            pricingTableVersion: "v1",
            pricingTablePath: "scripts/pricing-table.v1.json",
            selectedContextBlockCount: 1,
            selectedContextChars: 5,
            completionChars: 5,
            estimatedPromptTokens: 2,
            estimatedCompletionTokens: 2,
            estimatedPromptCostUsd: 0.000003,
            estimatedCompletionCostUsd: 0.00001,
            estimatedTotalCostUsd: 0.000013,
            retrievalToolHopCount: 0,
            retrievalToolHopTurnCount: 0,
            selectedContextCharsPerTurnMean: 5,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 2,
            completionCharsObservedTurnCount: 1,
            completionCharsObservedRate: 1,
            completionCharsPerTurnMean: 5,
            estimatedCompletionTokensPerTurnMean: 2,
            estimatedPromptCostUsdPerTurnMean: 0.000003,
            estimatedCompletionCostUsdPerTurnMean: 0.00001,
            estimatedTotalCostUsdPerTurnMean: 0.000013,
            turnsWithSelectedContextCount: 1,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 0,
            retrievalToolHopTurnRate: 0,
          },
          {
            mode: "vector_only",
            turnCount: 1,
            pricingTableVersion: "v1",
            pricingTablePath: "scripts/pricing-table.v1.json",
            selectedContextBlockCount: 1,
            selectedContextChars: 9,
            completionChars: 9,
            estimatedPromptTokens: 3,
            estimatedCompletionTokens: 3,
            estimatedPromptCostUsd: 0.000004,
            estimatedCompletionCostUsd: 0.000015,
            estimatedTotalCostUsd: 0.000019,
            retrievalToolHopCount: 2,
            retrievalToolHopTurnCount: 1,
            selectedContextCharsPerTurnMean: 9,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 3,
            completionCharsObservedTurnCount: 1,
            completionCharsObservedRate: 1,
            completionCharsPerTurnMean: 9,
            estimatedCompletionTokensPerTurnMean: 3,
            estimatedPromptCostUsdPerTurnMean: 0.000004,
            estimatedCompletionCostUsdPerTurnMean: 0.000015,
            estimatedTotalCostUsdPerTurnMean: 0.000019,
            turnsWithSelectedContextCount: 1,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 2,
            retrievalToolHopTurnRate: 1,
          },
          {
            mode: "graph_prior_only",
            turnCount: 1,
            pricingTableVersion: "v1",
            pricingTablePath: "scripts/pricing-table.v1.json",
            selectedContextBlockCount: 1,
            selectedContextChars: 5,
            completionChars: 5,
            estimatedPromptTokens: 2,
            estimatedCompletionTokens: 2,
            estimatedPromptCostUsd: 0.000003,
            estimatedCompletionCostUsd: 0.00001,
            estimatedTotalCostUsd: 0.000013,
            retrievalToolHopCount: 1,
            retrievalToolHopTurnCount: 1,
            selectedContextCharsPerTurnMean: 5,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 2,
            completionCharsObservedTurnCount: 1,
            completionCharsObservedRate: 1,
            completionCharsPerTurnMean: 5,
            estimatedCompletionTokensPerTurnMean: 2,
            estimatedPromptCostUsdPerTurnMean: 0.000003,
            estimatedCompletionCostUsdPerTurnMean: 0.00001,
            estimatedTotalCostUsdPerTurnMean: 0.000013,
            turnsWithSelectedContextCount: 1,
            turnsWithSelectedContextRate: 1,
            retrievalToolHopPerTurnMean: 1,
            retrievalToolHopTurnRate: 1,
          },
          {
            mode: "learned_route",
            turnCount: 1,
            pricingTableVersion: "v1",
            pricingTablePath: "scripts/pricing-table.v1.json",
            selectedContextBlockCount: 1,
            selectedContextChars: 11,
            completionChars: 11,
            estimatedPromptTokens: 3,
            estimatedCompletionTokens: 3,
            estimatedPromptCostUsd: 0.000004,
            estimatedCompletionCostUsd: 0.000015,
            estimatedTotalCostUsd: 0.000019,
            retrievalToolHopCount: 3,
            retrievalToolHopTurnCount: 1,
            selectedContextCharsPerTurnMean: 11,
            selectedContextBlocksPerTurnMean: 1,
            estimatedPromptTokensPerTurnMean: 3,
            completionCharsObservedTurnCount: 1,
            completionCharsObservedRate: 1,
            completionCharsPerTurnMean: 11,
            estimatedCompletionTokensPerTurnMean: 3,
            estimatedPromptCostUsdPerTurnMean: 0.000004,
            estimatedCompletionCostUsdPerTurnMean: 0.000015,
            estimatedTotalCostUsdPerTurnMean: 0.000019,
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
    expect(health.effectivenessReadout).toMatchObject({
      helping: "replay_backed_only",
    });
    expect(formatHealthMarkdown(health)).toContain("runtime healthy: true");
    expect(formatHealthMarkdown(health)).toContain("operator health: unknown");
    expect(formatHealthMarkdown(health)).toContain("helping: replay_backed_only");
    expect(formatHealthMarkdown(health)).toContain("latest replay proof is healthy, but live helpful/irrelevant/harmful context feedback is not visible here");
    expect(formatHealthMarkdown(health)).toContain("serve state: serving_active_pack");
    expect(formatHealthMarkdown(health)).toContain("clip rate: 0.25");
    expect(formatHealthMarkdown(health)).toContain("replay context chars total");
    expect(formatHealthMarkdown(health)).toContain("replay completion chars total");
    expect(formatHealthMarkdown(health)).toContain("replay estimated completion tokens total");
    expect(formatHealthMarkdown(health)).toContain("replay diagnostic top score mean");
    expect(formatHealthMarkdown(health)).toContain("pricing table version: v1");
    expect(formatHealthMarkdown(health)).toContain("learned_route: 11 prompt chars, 11 completion chars, 1 blocks, 3 estimated prompt tokens, 3 estimated completion tokens, $0.000004 prompt cost, $0.000015 completion cost, $0.000019 total cost");
    expect(formatHealthMarkdown(health)).toContain("proof minutes proxy");

    const aggregate = buildNightlyAggregate({ config, bundles, now, scanDurationMs: 42 });
    expect(aggregate.bundleTypeCounts.operatorProof).toBe(1);
    expect(aggregate.bundleTypeCounts.recordedSessionReplay).toBe(1);
    expect(aggregate.bundleTypeCounts.hostEvidence).toBe(1);
    expect(aggregate.replayMetrics.winnerModeCounts.learned_route).toBe(1);
    expect(aggregate.replayMetrics.selectedContextCharsTotal).toBe(30);
    expect(aggregate.replayMetrics.selectedContextBlocksTotal).toBe(4);
    expect(aggregate.replayMetrics.estimatedPromptTokensTotal).toBe(10);
    expect(aggregate.replayMetrics.completionCharsTotal).toBe(30);
    expect(aggregate.replayMetrics.estimatedCompletionTokensTotal).toBe(10);
    expect(aggregate.replayMetrics.estimatedPromptCostUsdTotal).toBeCloseTo(0.000014, 12);
    expect(aggregate.replayMetrics.estimatedCompletionCostUsdTotal).toBeCloseTo(0.00005, 12);
    expect(aggregate.replayMetrics.estimatedTotalCostUsdTotal).toBeCloseTo(0.000064, 12);
    expect(aggregate.replayMetrics.pricingTableVersion).toBe("v1");
    expect(aggregate.replayMetrics.retrievalToolHopCountTotal).toBe(6);
    expect(aggregate.replayMetrics.retrievalToolHopTurnCountTotal).toBe(3);
    expect(aggregate.replayMetrics.feedbackEventCountTotal).toBe(2);
    expect(aggregate.replayMetrics.nonApprovalFeedbackEventCountTotal).toBe(1);
    expect(aggregate.replayMetrics.turnsWithNonApprovalFeedbackCountTotal).toBe(1);
    expect(aggregate.replayMetrics.savingsByMode[3].mode).toBe("learned_route");
    expect(aggregate.replayMetrics.savingsByMode[3].estimatedPromptTokens).toBe(3);
    expect(aggregate.replayMetrics.savingsByMode[3].estimatedCompletionTokens).toBe(3);
    expect(aggregate.replayMetrics.savingsByMode[3].estimatedTotalCostUsd).toBe(0.000019);
    expect(aggregate.replayMetrics.savingsByMode[3].retrievalToolHopCount).toBe(3);
    expect(aggregate.operatorMetrics.stepMsTotal).toBe(1500);
    expect(aggregate.costProxy.bundleCount).toBe(3);
    expect(formatNightlyMarkdown(aggregate)).toContain("## Replay proof diagnostics");
    expect(formatNightlyMarkdown(aggregate)).toContain("diagnostic top-rank modes");
    expect(formatNightlyMarkdown(aggregate)).toContain("mean diagnostic top score");
    expect(formatNightlyMarkdown(aggregate)).toContain("replay estimated completion tokens total: 10");
    expect(formatNightlyMarkdown(aggregate)).toContain("pricing table version: v1");
    expect(formatNightlyMarkdown(aggregate)).toContain("| learned_route | 11 | 11 | 1 | 3 | 3 | $0.000004 | $0.000015 | $0.000019 | 3 | 1 | 1 | 1 | 1 |");
    expect(formatNightlyMarkdown(aggregate)).toContain("replay retrieval/tool-hop count total: 6");
    expect(formatNightlyMarkdown(aggregate)).toContain("replay turns with non-approval feedback total: 1");
    expect(formatNightlyMarkdown(aggregate)).toContain("proof minutes");
  });

  it("reports a feedback-backed thin readout when live helpful verdicts are visible", () => {
    const health = buildHealthSnapshot({
      config: {
        healthFreshnessDays: 7,
        freshnessThresholdDays: 21,
      },
      now: new Date("2026-04-02T12:00:00.000Z"),
      scanDurationMs: 42,
      bundles: buildThinReadoutBundles(),
      statusProbe: buildThinReadoutStatusProbe({
        contextFeedback: {
          verdictCounts: {
            helpful: 2,
            irrelevant: 0,
            harmful: 0,
          },
          coverage: {
            supervisedTraceCount: 2,
            routeTraceCount: 3,
          },
          latest: {
            agentIdentity: {
              agentId: "operator-readout",
              lane: "t107",
            },
          },
        },
      }),
    });

    expect(health.effectivenessReadout).toMatchObject({
      helping: "feedback_backed",
      summary: "live traced-route feedback trends helpful, but coverage is still partial",
    });
    expect(health.effectivenessReadout.where).toEqual(expect.arrayContaining([
      "serve-path pack=pack-live",
      "latest feedback lane=operator-readout/t107",
      "replay bundle=trace-live diagnosticTopMode=learned_route",
    ]));
    expect(health.effectivenessReadout.why).toEqual(expect.arrayContaining([
      expect.stringContaining("feedback helpful=2 irrelevant=0 harmful=0 coverage=2/3"),
      expect.stringContaining("attribution quality=exact_only"),
      expect.stringContaining("replay diagnosticTopMode=learned_route"),
      expect.stringContaining("route freshness=updated"),
    ]));
    expect(health.effectivenessReadout.staleOrMissing).toEqual([
      "operator health is partial: background-learning truth is partial in the current status surface",
    ]);

    const markdown = formatHealthMarkdown(health);
    expect(markdown).toContain("helping: feedback_backed");
    expect(markdown).toContain("summary: live traced-route feedback trends helpful, but coverage is still partial");
  });

  it("reports a mixed thin readout when live feedback includes harm", () => {
    const health = buildHealthSnapshot({
      config: {
        healthFreshnessDays: 7,
        freshnessThresholdDays: 21,
      },
      now: new Date("2026-04-02T12:00:00.000Z"),
      scanDurationMs: 42,
      bundles: buildThinReadoutBundles(),
      statusProbe: buildThinReadoutStatusProbe({
        contextFeedback: {
          verdictCounts: {
            helpful: 1,
            irrelevant: 0,
            harmful: 1,
          },
          coverage: {
            supervisedTraceCount: 2,
            routeTraceCount: 2,
          },
          latest: {
            agentIdentity: {
              agentId: "operator-readout",
              lane: "t107",
            },
          },
        },
      }),
    });

    expect(health.effectivenessReadout).toMatchObject({
      helping: "mixed",
      summary: "live traced-route feedback includes harmful verdicts, so OCB is not yet safely helping",
    });

    const markdown = formatHealthMarkdown(health);
    expect(markdown).toContain("helping: mixed");
    expect(markdown).toContain("harmful verdicts");
  });

  it("falls back to traced-learning feedback and attribution surfaces when legacy fields are absent", () => {
    const health = buildHealthSnapshot({
      config: {
        healthFreshnessDays: 7,
        freshnessThresholdDays: 21,
      },
      now: new Date("2026-04-09T07:05:28.638Z"),
      scanDurationMs: 42,
      bundles: buildThinReadoutBundles(),
      statusProbe: {
        command: "node packages/cli/dist/src/cli.js status --openclaw-home ~/.openclaw --json",
        startedAt: "2026-04-09T07:05:17.018Z",
        endedAt: "2026-04-09T07:05:28.638Z",
        durationMs: 11620,
        exitCode: 0,
        signal: null,
        stdout: "{}",
        stderr: "",
        parsed: {
          brain: {
            activePackId: "pack-live",
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
          passiveLearning: {
            watch: {
              state: "watching",
              lastHeartbeatAt: "2026-04-09T07:04:52.461Z",
              intervalSeconds: 30,
              proofState: "self_proving",
              teacherArtifactCount: 282,
            },
          },
          learningAttribution: {
            quality: "unavailable",
            nonZeroObservationCount: 0,
            exactMatchCount: 0,
            heuristicMatchCount: 0,
            unmatchedCount: 0,
            ambiguousCount: 0,
          },
          tracedLearning: {
            feedbackSummary: {
              visible: true,
              helpfulCount: 32,
              irrelevantCount: 0,
              harmfulCount: 0,
              supervisedTraceCount: 32,
              routeTraceCount: 177,
              latestAgentIdentity: null,
            },
            attributionCoverage: {
              visible: true,
              gatingVisible: true,
              completedWithoutEvaluationCount: 0,
              readyCount: 59,
              delayedCount: 0,
              budgetDeferredCount: 27,
            },
          },
        },
      },
    });

    expect(health.feedbackTruth).toMatchObject({
      visible: true,
      helpfulCount: 32,
      routeTraceCount: 177,
    });
    expect(health.attributionCoverageTruth).toMatchObject({
      visible: true,
      readyCount: 59,
      budgetDeferredCount: 27,
    });

    const markdown = formatHealthMarkdown(health);
    expect(markdown).toContain("feedback: helpful=32 irrelevant=0 harmful=0 coverage=32/177");
    expect(markdown).toContain("attribution coverage: evaluated=59/86 completedWithoutEval=0 ready=59 delayed=0 budgetDeferred=27");
  });

  it("uses live status thin-truth surfaces in nightly aggregate when available", () => {
    const aggregate = buildNightlyAggregate({
      config: {
        healthFreshnessDays: 7,
        freshnessThresholdDays: 21,
      },
      now: new Date("2026-04-09T07:05:28.638Z"),
      scanDurationMs: 42,
      bundles: buildThinReadoutBundles(),
      statusProbe: {
        command: "node packages/cli/dist/src/cli.js status --openclaw-home ~/.openclaw --json",
        startedAt: "2026-04-09T07:05:17.018Z",
        endedAt: "2026-04-09T07:05:28.638Z",
        durationMs: 11620,
        exitCode: 0,
        signal: null,
        stdout: "{}",
        stderr: "",
        parsed: {
          brain: {
            activePackId: "pack-live",
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
          passiveLearning: {
            watch: {
              state: "watching",
              lastHeartbeatAt: "2026-04-09T07:04:52.461Z",
              intervalSeconds: 30,
              proofState: "self_proving",
              teacherArtifactCount: 282,
            },
          },
          learningAttribution: {
            quality: "unavailable",
            nonZeroObservationCount: 0,
            exactMatchCount: 0,
            heuristicMatchCount: 0,
            unmatchedCount: 0,
            ambiguousCount: 0,
          },
          tracedLearning: {
            feedbackSummary: {
              visible: true,
              helpfulCount: 32,
              irrelevantCount: 0,
              harmfulCount: 0,
              supervisedTraceCount: 32,
              routeTraceCount: 177,
              latestAgentIdentity: null,
            },
            attributionCoverage: {
              visible: true,
              gatingVisible: true,
              completedWithoutEvaluationCount: 0,
              readyCount: 59,
              delayedCount: 0,
              budgetDeferredCount: 27,
            },
          },
        },
      },
    });

    expect(aggregate.feedbackTruth).toMatchObject({
      visible: true,
      helpfulCount: 32,
      routeTraceCount: 177,
      source: "live_status",
    });
    expect(aggregate.attributionCoverageTruth).toMatchObject({
      visible: true,
      readyCount: 59,
      budgetDeferredCount: 27,
      source: "live_status",
    });

    const markdown = formatNightlyMarkdown(aggregate);
    expect(markdown).toContain("feedback: helpful=32 irrelevant=0 harmful=0 coverage=32/177 (source=live_status)");
    expect(markdown).toContain("attribution coverage: evaluated=59/86 completedWithoutEval=0 ready=59 delayed=0 budgetDeferred=27 (source=live_status)");
  });

  it("carries latest operator-health truth into the nightly markdown", () => {
    const now = new Date("2026-03-31T13:00:00.000Z");
    const config = {
      healthFreshnessDays: 7,
      freshnessThresholdDays: 21,
    };
    const aggregate = buildNightlyAggregate({
      config,
      now,
      scanDurationMs: 42,
      bundles: [
        {
          kind: "host-evidence",
          bundleId: "host-proof",
          relativePath: "openclawbrain/docs/evidence/2026-03-26/abc123/host-proof",
          canonicalAt: "2026-03-26T00:00:00.000Z",
          ageDays: 5,
          fileCount: 4,
          artifactBytes: 1000,
          validationOk: true,
          operatorHealth: summarizeOperatorHealth({
            workerHealthy: true,
            workerMode: "child",
            workerStatus: "running",
            watchState: null,
            proofState: null,
            teacherArtifactCount: null,
          }),
          metrics: {
            securityCriticalCount: null,
            securityWarnCount: null,
            gatewayReachable: null,
            workerHealthy: true,
            memoryFiles: null,
            sessionCount: null,
          },
        },
      ],
    });

    expect(aggregate.latestOperatorHealth).toMatchObject({
      status: "partial",
      partial: true,
      unknown: false,
      stale: false,
    });
    const markdown = formatNightlyMarkdown(aggregate);
    expect(markdown).toContain("## Latest operator health");
    expect(markdown).toContain("operator health: partial");
    expect(markdown).toContain("operator health flags: partial=true, unknown=false, stale=false");
    expect(markdown).toContain("background-learning truth is partial in the current status surface");
  });
});
