import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { expect, test } from "vitest";

import {
  GRAPHIFY_FINAL_REPLAY_PROOF_MODE_ORDER_V1,
  writeGraphifyFinalReplayProof,
} from "../scripts/graphify-final-replay-proof.mjs";

test("graphify final replay/eval proof lane writes a bounded packet across the landed surfaces", () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-graphify-final-proof-"));
  try {
    const artifactRoot = path.join(root, "task-artifacts", "T-20260406-166");
    const proofRoot = path.join(artifactRoot, "final-replay-proof");
    const reportPath = path.join(artifactRoot, "final-replay-proof-report.md");
    const statusPath = path.join(root, "task-status", "T-20260406-166", "final-replay-proof.json");

    const result = writeGraphifyFinalReplayProof({
      workspaceRoot: root,
      repoRoot: path.resolve(process.cwd()),
      artifactRoot,
      proofRoot,
      reportPath,
      statusPath,
      generatedAt: "2026-04-07T00:40:00.000Z",
    });

    expect(result.verdict.status).toBe("pass");
    expect(result.report.modes.length).toBe(GRAPHIFY_FINAL_REPLAY_PROOF_MODE_ORDER_V1.length);
    expect(result.report.modeRanking.length).toBe(GRAPHIFY_FINAL_REPLAY_PROOF_MODE_ORDER_V1.length);
    expect(result.report.modeRanking[0].mode).toBe("graphify_artifacts_only");
    expect(result.verdict.coldStartDelta).toBeGreaterThan(0);
    expect(result.verdict.diagnosticOnlySurfaces.join(",")).toBe("deterministic-lints,maintenance-diff");

    const reportText = readFileSync(reportPath, "utf8");
    const status = JSON.parse(readFileSync(statusPath, "utf8"));
    const verdict = JSON.parse(readFileSync(path.join(proofRoot, "verdict.json"), "utf8"));
    const surfaceMap = JSON.parse(readFileSync(path.join(proofRoot, "surface-map.json"), "utf8"));
    const modeScorecard = JSON.parse(readFileSync(path.join(proofRoot, "mode-scorecard.json"), "utf8"));

    expect(reportText).toMatch(/Graphify helps cold start/);
    expect(reportText).toMatch(/maintenance diff/);
    expect(reportText).toMatch(/diagnostic-only/);
    expect(status.status).toBe("pass");
    expect(status.verdict.status).toBe("pass");
    expect(status.blockers).toHaveLength(0);
    expect(verdict.status).toBe("pass");
    expect(surfaceMap.surfaces).toHaveLength(7);
    expect(modeScorecard.modes).toHaveLength(GRAPHIFY_FINAL_REPLAY_PROOF_MODE_ORDER_V1.length);
    expect(modeScorecard.modeRanking[0].mode).toBe("graphify_artifacts_only");

    const importLearned = modeScorecard.modes.find((entry: { mode: string; supportScore: number }) => entry.mode === "graphify_import_plus_learned_route");
    const learnedBaseline = modeScorecard.modes.find((entry: { mode: string; supportScore: number }) => entry.mode === "learned_route_no_graphify_import");
    expect(importLearned?.supportScore ?? 0).toBeGreaterThan(learnedBaseline?.supportScore ?? 0);
    expect(result.report.maintenanceDiff.report.counts.currentSurfaceCount).toBeGreaterThan(0);
    expect(result.report.maintenanceDiff.report.counts.ocbSurfaceCount).toBeGreaterThan(0);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
