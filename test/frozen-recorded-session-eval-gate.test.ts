import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { checksumJsonPayload } from "@openclawbrain/contracts";
import { afterEach, describe, expect, it } from "vitest";
import {
  CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT,
  FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT,
  FROZEN_RECORDED_SESSION_EVAL_REPORT_CONTRACT,
  runFrozenRecordedSessionEvalGate,
  type CanonicalRecordedSessionTraceSetManifestV1,
  type FrozenRecordedSessionEvalManifestV1,
} from "../scripts/run-frozen-recorded-session-eval-gate.js";

const tempRoots: string[] = [];

afterEach(() => {
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
});

function createTempRoot(label: string): string {
  const root = mkdtempSync(path.join(os.tmpdir(), `${label}-`));
  tempRoots.push(root);
  return root;
}

function writeJson(filePath: string, payload: unknown): void {
  writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function createWorkspace(rootDir: string, label: string, readmeText: string) {
  const workspaceRoot = path.join(rootDir, `${label}-workspace`);
  rmSync(workspaceRoot, { recursive: true, force: true });
  mkdirSync(workspaceRoot, { recursive: true });
  writeFileSync(path.join(workspaceRoot, "README.md"), readmeText, { encoding: "utf8", flag: "wx" });
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-03-25T00:00:00.000Z",
    rootDir: workspaceRoot,
    branch: "main",
    revision: `rev-${label}`,
    labels: ["test"],
  };
}

function buildComparativeTrace(rootDir: string, label: string) {
  const workspace = createWorkspace(
    rootDir,
    label,
    "# Recorded session workspace\nThe routing guide lives here.\n",
  );
  return {
    contract: "recorded_session_trace.v1",
    traceId: `trace-comparative-${label}`,
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-25T00:00:00.000Z",
    bundleBuiltAt: "2026-03-25T00:10:00.000Z",
    agentId: "agent",
    sessionId: `session-comparative-${label}`,
    channel: "cli",
    sourceStream: "recorded/session",
    privacy: {
      sanitized: true,
      notes: ["test fixture"],
    },
    workspace,
    seedBuiltAt: "2026-03-25T00:01:00.000Z",
    seedActivatedAt: "2026-03-25T00:02:00.000Z",
    seedCues: [
      {
        cueId: `cue-routing-guide-${label}`,
        createdAt: "2026-03-25T00:00:30.000Z",
        content: "The routing guide lives here.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: `${label}-turn-1`,
        createdAt: "2026-03-25T00:03:00.000Z",
        deliveredAt: "2026-03-25T00:03:30.000Z",
        userMessage: "show the routing guide",
        runtimeHints: ["routing", "guide"],
        feedback: [
          {
            createdAt: "2026-03-25T00:03:45.000Z",
            content: "Keep the routing guide easy to find.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["routing guide"],
      },
      {
        turnId: `${label}-turn-2`,
        createdAt: "2026-03-25T00:05:00.000Z",
        deliveredAt: "2026-03-25T00:05:30.000Z",
        userMessage: "show the routing guide again",
        runtimeHints: ["routing", "guide", "again"],
        feedback: [
          {
            createdAt: "2026-03-25T00:05:45.000Z",
            content: "The routing guide is still the right answer.",
            kind: "approval",
          },
        ],
        expectedContextPhrases: ["routing guide"],
      },
    ],
  };
}

function buildTrainFreezeTrace(rootDir: string, label: string) {
  const workspace = createWorkspace(
    rootDir,
    label,
    "# Train freeze eval workspace\nAlways read README before editing code.\n",
  );
  return {
    contract: "recorded_session_trace.v1",
    traceId: `trace-train-freeze-${label}`,
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-25T10:00:00.000Z",
    bundleBuiltAt: "2026-03-25T10:30:00.000Z",
    sessionId: `session-train-freeze-${label}`,
    channel: "chat",
    sourceStream: "recorded/session",
    privacy: { sanitized: true, notes: ["test"] },
    workspace,
    seedBuiltAt: "2026-03-25T09:56:00.000Z",
    seedActivatedAt: "2026-03-25T09:57:00.000Z",
    seedCues: [
      {
        cueId: `cue-readme-${label}`,
        createdAt: "2026-03-25T09:50:00.000Z",
        content: "Always read README before editing code.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: `${label}-turn-1`,
        createdAt: "2026-03-25T10:01:00.000Z",
        deliveredAt: "2026-03-25T10:01:30.000Z",
        userMessage: "What should I read before editing?",
        feedback: [
          {
            createdAt: "2026-03-25T10:01:45.000Z",
            content: "Correct: read README before editing code.",
            kind: "approval",
          },
        ],
        expectedContextPhrases: ["readme before editing"],
      },
      {
        turnId: `${label}-turn-2`,
        createdAt: "2026-03-25T10:05:00.000Z",
        deliveredAt: "2026-03-25T10:05:30.000Z",
        userMessage: "Before changing files, what is the rule?",
        expectedContextPhrases: ["readme before editing"],
      },
      {
        turnId: `${label}-turn-3`,
        createdAt: "2026-03-25T10:10:00.000Z",
        deliveredAt: "2026-03-25T10:10:30.000Z",
        userMessage: "What must happen before editing code?",
        expectedContextPhrases: ["readme before editing"],
      },
    ],
  };
}

function buildScoreResolutionTrace(rootDir: string, label: string) {
  const workspace = createWorkspace(
    rootDir,
    label,
    "# Score resolution workspace\nReplay scoring should preserve multi-phrase turn weight.\n",
  );
  return {
    contract: "recorded_session_trace.v1",
    traceId: `trace-score-resolution-${label}`,
    source: "sanitized_recorded_session",
    recordedAt: "2026-03-28T17:26:18.111Z",
    bundleBuiltAt: "2026-03-28T20:02:41.227Z",
    agentId: "main",
    sessionId: `sanitized-session-score-resolution-${label}`,
    channel: "telegram",
    sourceStream: "telegram/direct/proof-plan",
    privacy: {
      sanitized: true,
      notes: ["sanitized recorded session"],
    },
    workspace,
    evalTurnCount: 1,
    seedBuiltAt: "2026-03-28T17:26:18.111Z",
    seedActivatedAt: "2026-03-28T17:27:18.111Z",
    seedCues: [
      {
        cueId: `cue-proof-run-${label}`,
        createdAt: "2026-03-28T17:26:18.111Z",
        content: "T-20260328-040 is the real-trace learned-route proof run for OpenClawBrain.",
        kind: "teaching",
      },
    ],
    turns: [
      {
        turnId: `${label}-plan-turn-1`,
        createdAt: "2026-03-28T19:54:42.389Z",
        deliveredAt: "2026-03-28T19:54:57.546Z",
        userMessage: "So what's next for us to work on?",
        runtimeHints: ["proof", "next", "plan"],
        feedback: [
          {
            createdAt: "2026-03-28T19:54:57.546Z",
            content: "Next steps: commit the rollout-evaluator scaffold, export a sanitized non-test real-trace replay corpus, and rerun the rollout evaluator.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["learned-route proof run"],
      },
      {
        turnId: `${label}-plan-turn-2`,
        createdAt: "2026-03-28T19:56:06.301Z",
        deliveredAt: "2026-03-28T19:58:48.383Z",
        userMessage: "Please make an end-to-end master plan for me.",
        runtimeHints: ["master-plan", "proof", "roadmap"],
        feedback: [
          {
            createdAt: "2026-03-28T19:58:48.383Z",
            content: "The master plan ends with a rollout verdict: ready, limited, or blocked.",
            kind: "teaching",
          },
        ],
        expectedContextPhrases: ["real-trace learned-route proof run"],
      },
      {
        turnId: `${label}-plan-turn-3`,
        createdAt: "2026-03-28T20:01:14.399Z",
        deliveredAt: "2026-03-28T20:01:41.227Z",
        userMessage: "Please do this entire plan end to end.",
        runtimeHints: ["execute", "proof", "end-to-end"],
        expectedContextPhrases: [
          "sanitized non-test real-trace replay corpus",
          "ready, limited, or blocked",
        ],
      },
    ],
  };
}

function writeTrace(rootDir: string, trace: Record<string, unknown>) {
  const tracePath = path.join(rootDir, `${trace.traceId as string}.json`);
  writeJson(tracePath, trace);
  return {
    traceId: trace.traceId as string,
    tracePath,
    traceHash: checksumJsonPayload(trace),
  };
}

function writeManifest(
  rootDir: string,
  manifestId: string,
  traces: Array<{ traceId: string; tracePath: string; traceHash: string }>,
): string {
  const manifestPath = path.join(rootDir, `${manifestId}.manifest.json`);
  const manifest: FrozenRecordedSessionEvalManifestV1 = {
    contract: FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT,
    manifestId,
    expectedTraceCount: traces.length,
    traces: traces.map((trace) => ({
      traceId: trace.traceId,
      tracePath: path.relative(rootDir, trace.tracePath),
      traceHash: trace.traceHash,
    })),
  };
  writeJson(manifestPath, manifest);
  return manifestPath;
}

function writeCanonicalManifest(
  rootDir: string,
  setId: string,
  traces: Array<{ tracePath: string }>,
): string {
  const manifestPath = path.join(rootDir, `${setId}.canonical-manifest.json`);
  const manifest: CanonicalRecordedSessionTraceSetManifestV1 = {
    contract: CANONICAL_RECORDED_SESSION_TRACE_SET_MANIFEST_CONTRACT,
    setId,
    traceCount: traces.length,
    entries: traces.map((trace, index) => ({
      slotId: `slot-${index + 1}`,
      path: path.relative(rootDir, trace.tracePath),
    })),
    realTraceCoverage: {
      summary: "No verified first-party real production traces are checked in; this set is equivalent-only.",
    },
    redactionPolicy: {
      summary: "All inputs are already synthetic or sanitized replayable equivalents.",
    },
  };
  writeJson(manifestPath, manifest);
  return manifestPath;
}

describe("runFrozenRecordedSessionEvalGate", () => {
  it("records the quality-adjusted prompt-cost signal when learned_route improves quality", () => {
    const rootDir = createTempRoot("frozen-eval-gate-pass");
    const trace = writeTrace(rootDir, buildScoreResolutionTrace(rootDir, "pass-a"));
    const manifestPath = writeManifest(rootDir, "pass-manifest", [trace]);
    const outputDir = path.join(rootDir, "gate-output");

    const descriptor = runFrozenRecordedSessionEvalGate({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
    });

    expect(descriptor.report.contract).toBe(FROZEN_RECORDED_SESSION_EVAL_REPORT_CONTRACT);
    expect(descriptor.report.status).toBe("pass");
    expect(descriptor.report.checks.every((check: { status: string }) => check.status === "pass")).toBe(true);
    expect(descriptor.report.checks.find((check: { id: string }) => check.id === "quality_adjusted_prompt_savings_reported")?.status).toBe("pass");
    expect(descriptor.report.traceResults).toHaveLength(1);
    expect(descriptor.report.traceResults[0]?.validationOk).toBe(true);
    expect(descriptor.report.qualityAdjustedPromptSavings.qualityAdjustedPromptSavingsUsd).not.toBeNull();
    expect((descriptor.report.qualityAdjustedPromptSavings.qualityAdjustedPromptSavingsUsd ?? 0) > 0).toBe(true);
    expect(readFileSync(descriptor.reportPath, "utf8")).toContain('"status": "pass"');
    expect(readFileSync(descriptor.summaryPath, "utf8")).toContain("quality-adjusted prompt savings usd");
  });

  it("only hard-fails the prompt-cost signal when an explicit threshold is configured", () => {
    const rootDir = createTempRoot("frozen-eval-gate-fail");
    const trace = writeTrace(rootDir, buildTrainFreezeTrace(rootDir, "fail-a"));
    const manifestPath = writeManifest(rootDir, "fail-manifest", [trace]);
    const outputDir = path.join(rootDir, "gate-output");

    const descriptor = runFrozenRecordedSessionEvalGate({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
    });

    expect(descriptor.report.status).toBe("pass");
    expect(descriptor.report.checks.find((check: { id: string }) => check.id === "trace_replay_proofs_valid")?.status).toBe("pass");
    expect(descriptor.report.checks.find((check: { id: string }) => check.id === "learned_route_non_inferior_to_graph_prior_only")?.status).toBe("pass");
    expect(descriptor.report.checks.find((check: { id: string }) => check.id === "graph_prior_only_clears_no_brain_floor")?.status).toBe("pass");
    expect(descriptor.report.checks.find((check: { id: string }) => check.id === "quality_adjusted_prompt_savings_reported")?.status).toBe("pass");

    const computedSignal = descriptor.report.qualityAdjustedPromptSavings.qualityAdjustedPromptSavingsUsd;
    expect(computedSignal).not.toBeNull();

    const thresholdDescriptor = runFrozenRecordedSessionEvalGate({
      manifestPath,
      outputDir: path.join(rootDir, "threshold-output"),
      scratchRootDir: rootDir,
      thresholds: {
        minQualityAdjustedPromptSavingsUsd: Number(((computedSignal ?? 0) + 0.000001).toFixed(6)),
      },
    });

    expect(thresholdDescriptor.report.status).toBe("fail");
    expect(thresholdDescriptor.report.checks.find((check: { id: string }) => check.id === "quality_adjusted_prompt_savings_threshold_met")?.status).toBe("fail");
  });

  it("accepts the canonical frozen trace-set manifest contract", () => {
    const rootDir = createTempRoot("frozen-eval-gate-canonical");
    const trace = writeTrace(rootDir, buildScoreResolutionTrace(rootDir, "canonical-a"));
    const manifestPath = writeCanonicalManifest(rootDir, "canonical-frozen-20", [trace]);
    const outputDir = path.join(rootDir, "gate-output");

    const descriptor = runFrozenRecordedSessionEvalGate({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
    });

    expect(descriptor.report.status).toBe("pass");
    expect(descriptor.report.manifestId).toBe("canonical-frozen-20");
    expect(descriptor.report.expectedTraceCount).toBe(1);
    expect(descriptor.report.notes.some((note: string) => note.includes("truth boundary:"))).toBe(true);
    expect(descriptor.report.notes.some((note: string) => note.includes("redaction policy:"))).toBe(true);
  });

  it("returns a blocked report when the manifest is missing", () => {
    const rootDir = createTempRoot("frozen-eval-gate-blocked");
    const manifestPath = path.join(rootDir, "missing.manifest.json");
    const outputDir = path.join(rootDir, "gate-output");

    const descriptor = runFrozenRecordedSessionEvalGate({
      manifestPath,
      outputDir,
      scratchRootDir: rootDir,
    });

    expect(descriptor.report.status).toBe("blocked");
    expect(descriptor.report.traceResults).toHaveLength(0);
    expect(descriptor.report.issues[0]).toContain("manifest missing");
    expect(readFileSync(descriptor.reportPath, "utf8")).toContain('"status": "blocked"');
  });
});
