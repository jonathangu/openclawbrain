import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { checksumJsonPayload } from "@openclawbrain/contracts";
import { afterEach, describe, expect, it } from "vitest";
import {
  FROZEN_RECORDED_SESSION_EVAL_MANIFEST_CONTRACT,
  runFrozenRecordedSessionEvalGate,
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

function createWorkspace(rootDir: string, label: string) {
  const workspaceRoot = path.join(rootDir, `${label}-workspace`);
  mkdirSync(workspaceRoot, { recursive: true });
  writeFileSync(
    path.join(workspaceRoot, "README.md"),
    "# Score resolution workspace\nReplay scoring should preserve multi-phrase turn weight.\n",
    "utf8",
  );
  return {
    workspaceId: `ws-${label}`,
    snapshotId: `snapshot-${label}`,
    capturedAt: "2026-03-28T17:26:18.111Z",
    rootDir: workspaceRoot,
    branch: "main",
    revision: `rev-${label}`,
    labels: ["test"],
  };
}

function buildScoreResolutionTrace(rootDir: string, label: string) {
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
    workspace: createWorkspace(rootDir, label),
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

describe("runFrozenRecordedSessionEvalGate economics framing", () => {
  it("records the prompt-savings proxy by default and only hard-gates it when a threshold is set", () => {
    const rootDir = createTempRoot("frozen-eval-gate-economics");
    const trace = writeTrace(rootDir, buildScoreResolutionTrace(rootDir, "economics-a"));
    const manifestPath = writeManifest(rootDir, "economics-manifest", [trace]);

    const defaultDescriptor = runFrozenRecordedSessionEvalGate({
      manifestPath,
      outputDir: path.join(rootDir, "default-output"),
      scratchRootDir: rootDir,
    });

    const defaultCheck = defaultDescriptor.report.checks.find((check) => check.id === "quality_adjusted_prompt_savings_reported");
    expect(defaultDescriptor.report.qualityAdjustedPromptSavings.qualityAdjustedPromptSavingsUsd).not.toBeNull();
    expect(defaultCheck?.status).toBe("pass");
    expect(defaultCheck?.detail).toContain("quality-adjusted prompt savings = ");
    expect(defaultCheck?.detail).toContain("does not model long-run task-level economics");
    expect(defaultDescriptor.report.assumptions.some((assumption) => assumption.includes("long-run task-level economics"))).toBe(true);

    const computedSignal = defaultDescriptor.report.qualityAdjustedPromptSavings.qualityAdjustedPromptSavingsUsd ?? 0;
    const thresholdDescriptor = runFrozenRecordedSessionEvalGate({
      manifestPath,
      outputDir: path.join(rootDir, "threshold-output"),
      scratchRootDir: rootDir,
      thresholds: {
        minQualityAdjustedPromptSavingsUsd: Number((computedSignal + 0.000001).toFixed(6)),
      },
    });

    const thresholdCheck = thresholdDescriptor.report.checks.find((check) => check.id === "quality_adjusted_prompt_savings_threshold_met");
    expect(thresholdDescriptor.report.status).toBe("fail");
    expect(thresholdCheck?.status).toBe("fail");
  });
});
