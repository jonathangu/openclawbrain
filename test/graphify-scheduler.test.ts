import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { expect, test } from "vitest";

import { writeGraphifySchedulerRun } from "../scripts/graphify-scheduler.mjs";

function makeTempRoot() {
  return mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-scheduler-"));
}

function makeSourceBundleFixture(root: string) {
  const openclawHome = path.join(root, ".openclaw-smoke");
  const activationRoot = path.join(root, ".openclawbrain", "activation");
  const sessionsDir = path.join(openclawHome, "agents", "main", "sessions");
  mkdirSync(sessionsDir, { recursive: true });
  mkdirSync(path.join(activationRoot, "attachment-truth"), { recursive: true });

  writeFileSync(path.join(openclawHome, "openclaw.json"), JSON.stringify({ profile: "graphify-scheduler-smoke" }, null, 2), "utf8");
  writeFileSync(
    path.join(sessionsDir, "sessions.json"),
    JSON.stringify(
      {
        smoke: {
          sessionId: "session-graphify-scheduler-smoke",
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
        id: "session-graphify-scheduler-smoke",
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
          content: "Graphify scheduler smoke fixture.",
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
          content: "Please keep Graphify off the serve path.",
          timestamp: 1775520002000,
        },
      }),
    ].join("\n") + "\n",
    "utf8",
  );

  return { openclawHome, activationRoot };
}

test("graphify scheduler delta and reorg cadence runs stay off-path and registry-linked", () => {
  const root = makeTempRoot();
  const repoRoot = path.resolve(process.cwd());
  const { openclawHome, activationRoot } = makeSourceBundleFixture(root);

  try {
    const delta = writeGraphifySchedulerRun({
      cadence: "delta",
      repoRoot,
      workspaceRoot: root,
      openclawHome,
      activationRoot,
      generatedAt: "2026-04-07T00:40:00.000Z",
      runId: "delta-smoke",
    });

    const reorg = writeGraphifySchedulerRun({
      cadence: "reorg",
      repoRoot,
      workspaceRoot: root,
      openclawHome,
      activationRoot,
      generatedAt: "2026-04-07T00:45:00.000Z",
      runId: "reorg-smoke",
    });

    expect(delta.status).toBe("completed");
    expect(delta.offPath).toBe(true);
    expect(delta.inspectable).toBe(true);
    expect(delta.replayable).toBe(true);
    expect(delta.truthBoundary).toBe("below correction/raw-authority truth");
    expect(delta.sourceBundle.corpusDigest?.startsWith("sha256:")).toBe(true);
    expect(delta.graphifyRun.graph.nodeCount).toBeGreaterThan(0);
    expect(delta.downstreamArtifacts.map((artifact: { kind: string }) => artifact.kind)).toEqual(
      expect.arrayContaining([
        "source-bundle",
        "graphify-run",
        "compiled-artifact-pack",
        "candidate-pack-input",
        "import-slice",
        "deterministic-lints",
        "maintenance-diff",
        "retention-policy-json",
        "retention-policy-markdown",
      ]),
    );

    expect(reorg.status).toBe("completed");
    expect(reorg.registryPath).toBe(delta.registryPath);
    expect(reorg.retentionPolicyPath).not.toBe(delta.retentionPolicyPath);

    const registry = JSON.parse(readFileSync(delta.registryPath, "utf8"));
    const deltaRun = registry.runs.find((entry: { cadence: string; runId: string }) => entry.cadence === "delta" && entry.runId === "delta-smoke");
    const reorgRun = registry.runs.find((entry: { cadence: string; runId: string }) => entry.cadence === "reorg" && entry.runId === "reorg-smoke");

    expect(registry.contract).toBe("graphify_scheduler_registry.v1");
    expect(registry.runCount).toBe(2);
    expect(deltaRun).toBeTruthy();
    expect(reorgRun).toBeTruthy();
    expect(registry.latestByCadence.delta.runId).toBe("delta-smoke");
    expect(registry.latestByCadence.reorg.runId).toBe("reorg-smoke");
    expect(deltaRun.downstreamArtifacts.some((artifact: { kind: string }) => artifact.kind === "maintenance-diff")).toBe(true);
    expect(reorgRun.downstreamArtifacts.some((artifact: { kind: string }) => artifact.kind === "import-slice")).toBe(true);

    const retentionMarkdown = readFileSync(delta.retentionPolicyMarkdownPath, "utf8");
    const status = JSON.parse(readFileSync(delta.statusPath, "utf8"));

    expect(retentionMarkdown).toMatch(/Graphify scheduler retention policy/);
    expect(retentionMarkdown).toMatch(/source bundles/);
    expect(retentionMarkdown).toMatch(/candidate-pack inputs/);
    expect(retentionMarkdown).toMatch(/vacuum only after registry linkage is removed or replaced/);
    expect(status.offPath).toBe(true);
    expect(status.inspectable).toBe(true);
    expect(status.replayable).toBe(true);
    expect(status.maintenanceDiff.verdict.verdict).toBeTruthy();
  }
  finally {
    rmSync(root, { recursive: true, force: true });
  }
});
