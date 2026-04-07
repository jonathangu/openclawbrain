import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { expect, test } from "vitest";

import { writeContinuousLearningAcceptanceLane } from "../scripts/continuous-learning-acceptance.ts";

function makeTempRoot() {
  return mkdtempSync(path.join(os.tmpdir(), "openclawbrain-continuous-learning-acceptance-"));
}

function makeSourceBundleFixture(root: string) {
  const openclawHome = path.join(root, ".openclaw-smoke");
  const activationRoot = path.join(root, ".openclawbrain", "activation");
  const sessionsDir = path.join(openclawHome, "agents", "main", "sessions");
  mkdirSync(sessionsDir, { recursive: true });
  mkdirSync(path.join(activationRoot, "attachment-truth"), { recursive: true });

  writeFileSync(path.join(openclawHome, "openclaw.json"), JSON.stringify({ profile: "continuous-learning-acceptance-smoke" }, null, 2), "utf8");
  writeFileSync(
    path.join(sessionsDir, "sessions.json"),
    JSON.stringify(
      {
        smoke: {
          sessionId: "session-continuous-learning-acceptance-smoke",
          sessionFile: path.join(sessionsDir, "smoke-session.jsonl"),
          updatedAt: 1,
          chatType: "telegram",
          origin: "test-fixture",
        },
      },
      null,
      2,
    ),
    "utf8",
  );
  writeFileSync(
    path.join(sessionsDir, "smoke-session.jsonl"),
    [
      JSON.stringify({
        type: "session",
        version: 1,
        id: "session-continuous-learning-acceptance-smoke",
        timestamp: "2026-04-07T00:00:00.000Z",
        cwd: "/tmp",
      }),
      JSON.stringify({
        type: "message",
        id: "msg-1",
        parentId: null,
        timestamp: "2026-04-07T00:00:01.000Z",
        message: {
          role: "assistant",
          content: "Continuous learning acceptance fixture.",
          timestamp: 1775520001000,
        },
      }),
      JSON.stringify({
        type: "message",
        id: "msg-2",
        parentId: "msg-1",
        timestamp: "2026-04-07T00:00:02.000Z",
        message: {
          role: "user",
          content: "Keep the stack off the serve path.",
          timestamp: 1775520002000,
        },
      }),
    ].join("\n") + "\n",
    "utf8",
  );

  return { openclawHome, activationRoot };
}

test("continuous learning acceptance lane widens proof across scheduler, replay, and proof smoke", () => {
  const root = makeTempRoot();
  const repoRoot = path.resolve(process.cwd());
  const { openclawHome, activationRoot } = makeSourceBundleFixture(root);
  const outputRoot = path.join(root, "artifacts", "continuous-learning-acceptance");

  try {
    const result = writeContinuousLearningAcceptanceLane({
      repoRoot,
      workspaceRoot: root,
      outputRoot,
      openclawHome,
      activationRoot,
      generatedAt: "2026-04-07T00:40:00.000Z",
      runId: "smoke",
      proofSmokeMaxAgeDays: 21,
    });

    expect(result.ok).toBe(true);
    expect(result.checks.proofSmoke.ok).toBe(true);
    expect(result.checks.graphifyDelta.ok).toBe(true);
    expect(result.checks.graphifyReorg.ok).toBe(true);
    expect(result.checks.finalReplayProof.ok).toBe(true);
    expect(result.checks.finalReplayProof.summary).toMatch(/pass/);

    const summaryText = readFileSync(result.summaryPath, "utf8");
    const status = JSON.parse(readFileSync(result.statusPath, "utf8"));

    expect(summaryText).toContain("Continuous learning acceptance lane");
    expect(summaryText).toContain("graphify delta");
    expect(summaryText).toContain("final replay proof");
    expect(status.contract).toBe("continuous_learning_acceptance_lane.v1");
    expect(status.ok).toBe(true);
    expect(status.checks.graphifyDelta.ok).toBe(true);
    expect(status.checks.graphifyReorg.ok).toBe(true);
    expect(status.checks.finalReplayProof.ok).toBe(true);
    expect(status.checks.proofSmoke.ok).toBe(true);
    expect(status.outputs.graphifySchedulerRoot).toContain("graphify-scheduler");
    expect(status.outputs.finalReplayProofRoot).toContain("final-replay-proof");
  }
  finally {
    rmSync(root, { recursive: true, force: true });
  }
});
