import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, afterEach } from "vitest";
import { compileColdStartDocsQaSourceBundleFromFileV1 } from "../../src/brain-core/cold-start-data-compiler.js";
import {
  replayColdStartRouterArtifactV1,
} from "../../src/brain-core/cold-start-router-replay-gate.js";
import {
  trainColdStartRouterArtifactV1,
} from "../../src/brain-core/cold-start-router-trainer.js";
import type { DataRegistryEntryV1, RouteDecisionRowV1 } from "../../src/brain-core/cold-start-router-contracts.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const sampleBundlePath = path.join(repoRoot, "artifacts", "cold-start-router-sample", "docs-qa-sample.raw.json");

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

function makeSmokeRegistryEntry(): DataRegistryEntryV1 {
  return {
    dataset_id: "dataset_cold_start_smoke",
    source_family: "qa",
    upstream_url: "https://example.org/cold-start-smoke",
    original_creator: "OpenClaw",
    license: "CC BY 4.0",
    commercial_use_status: "allowed",
    redistribution_status: "allowed",
    pii_risk: "none",
    benchmark_split_status: "train",
    approval_status: "approved_train",
    reviewer: "operator",
    immutable_snapshot_ref: "snapshot:cold-start-smoke@sha256:aaa111",
    exact_files: ["data/train.json"],
    file_hashes: {
      "data/train.json": "sha256:bbb222",
    },
    allowed_uses: ["route supervision", "ranking baselines"],
    disallowed_uses: ["private redistribution"],
    notes: ["fixture"],
    created_at: "2026-04-05T16:00:00Z",
    updated_at: "2026-04-05T16:00:00Z",
  };
}

function makeSmokeRouteRows(): RouteDecisionRowV1[] {
  const datasetId = "dataset_cold_start_smoke";
  return [
    {
      row_id: "row_route_001",
      dataset_id: datasetId,
      query: "Find the shipping correction memory and update the invoice draft",
      cursor_path: ["thread:billing", "mem:user_profile"],
      candidate_set: [
        { candidate_id: "mem:shipping_history", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.92 },
        { candidate_id: "doc:invoice_policy", candidate_type: "doc_chunk", authority: "operator_policy", freshness: "stale", token_cost: 28, score_hint: 0.18 },
        { candidate_id: "tool:gmail_search", candidate_type: "tool", authority: "runtime", freshness: "current", token_cost: 4, score_hint: 0.51 },
      ],
      teacher_action: { kind: "traverse", target_ids: ["mem:shipping_history"] },
      stop_label: "CONTINUE",
      evidence_spans: [
        { source_ref: "message_183", start: 21, end: 72 },
        { source_ref: "memory_entry_77", start: 0, end: 44 },
      ],
      hard_negatives: ["doc:invoice_policy"],
      outcome_gain: 0.84,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: "snapshot:cold-start-smoke@sha256:aaa111",
        recorded_by: "operator",
        recorded_at: "2026-04-05T16:01:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-05T16:02:00Z",
    },
    {
      row_id: "row_route_002",
      dataset_id: datasetId,
      query: "Stop once the invoice tool is enough",
      cursor_path: ["thread:billing"],
      candidate_set: [
        { candidate_id: "tool:invoice_draft", candidate_type: "tool", authority: "runtime", freshness: "current", token_cost: 2, score_hint: 0.87 },
        { candidate_id: "mem:shipping_history", candidate_type: "memory_node", authority: "human", freshness: "current", score_hint: 0.15 },
      ],
      teacher_action: { kind: "tool", tool_name: "tool:invoice_draft" },
      stop_label: "STOP_LOCAL",
      evidence_spans: [
        { source_ref: "message_184", start: 0, end: 30 },
      ],
      hard_negatives: [],
      outcome_gain: 0.42,
      provenance: {
        dataset: datasetId,
        source_license: "CC BY 4.0",
        source_family: "qa",
        source_snapshot_ref: "snapshot:cold-start-smoke@sha256:aaa111",
        recorded_by: "operator",
        recorded_at: "2026-04-05T16:03:00Z",
        review_status: "approved_train",
      },
      split_tag: "train",
      created_at: "2026-04-05T16:04:00Z",
    },
  ];
}

describe("cold-start router replay gate", () => {
  it("loads a produced artifact and passes on the curated docs/QA replay rows", () => {
    const bundle = compileColdStartDocsQaSourceBundleFromFileV1({ bundlePath: sampleBundlePath, repoRoot });
    const outputDir = createTempRoot("cold-start-router-replay-gate-good");

    trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-replay-good",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.3.8",
      registryEntries: [bundle.registryEntry],
      routeRows: bundle.routeRows,
      outputDir,
      routerIdentity: "router:docs-qa:replay",
      createdAt: "2026-04-05T17:35:00Z",
      trainingDataRefs: [bundle.registryEntry.dataset_id],
      replayGateRefs: ["replay:docs-qa-sample:tiny-gate"],
    });

    const verdict = replayColdStartRouterArtifactV1({
      artifactDir: outputDir,
      routeRows: bundle.routeRows,
    });

    expect(verdict).toMatchObject({
      passed: true,
      verdict: "pass",
      evaluatedRowCount: 2,
      passedRowCount: 2,
      failedRowCount: 0,
      skippedRowCount: 0,
    });
    expect(verdict.manifestSummary).toMatchObject({
      artifactId: "router-artifact-replay-good",
      packType: "base",
      trainingDataRefCount: 1,
      replayGateRefCount: 1,
      runtimeVersion: "openclawbrain-runtime@0.3.8",
    });
    expect(verdict.rowResults).toHaveLength(2);
    expect(verdict.rowResults[0]).toMatchObject({
      rowId: "docs-qa-routing-prior-conflict",
      passed: true,
      expectedTopCandidateId: "doc:routing-prior:conflict-resolution",
      actualTopCandidateId: "doc:routing-prior:conflict-resolution",
      expectedStopLabel: "CONTINUE",
      actualStopLabel: "CONTINUE",
    });
    expect(verdict.rowResults[1]).toMatchObject({
      rowId: "docs-qa-ci-first-deterministic-lints",
      passed: true,
      expectedTopCandidateId: "doc:teacher-v3-lints:ci-first",
      actualTopCandidateId: "doc:teacher-v3-lints:ci-first",
      expectedStopLabel: "CONTINUE",
      actualStopLabel: "CONTINUE",
    });
  });

  it("fails replay when the artifact manifest is obviously corrupted", () => {
    const bundle = compileColdStartDocsQaSourceBundleFromFileV1({ bundlePath: sampleBundlePath, repoRoot });
    const outputDir = createTempRoot("cold-start-router-replay-gate-bad");

    trainColdStartRouterArtifactV1({
      artifactId: "router-artifact-replay-bad",
      artifactVersion: "0.0.1",
      packType: "base",
      compatibleRuntimeVersion: "openclawbrain-runtime@0.3.8",
      registryEntries: [makeSmokeRegistryEntry()],
      routeRows: makeSmokeRouteRows(),
      outputDir,
      routerIdentity: "router:smoke:replay",
      createdAt: "2026-04-05T17:36:00Z",
      trainingDataRefs: ["dataset_cold_start_smoke"],
      replayGateRefs: ["replay:docs-qa-sample:tiny-gate"],
    });

    const manifestPath = path.join(outputDir, "manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
    manifest.weights_ref = "weights.json#sha256:deadbeef";
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

    const verdict = replayColdStartRouterArtifactV1({
      artifactDir: outputDir,
      routeRows: bundle.routeRows,
    });

    expect(verdict.passed).toBe(false);
    expect(verdict.verdict).toBe("fail");
    expect(verdict.loadIssues.length).toBeGreaterThan(0);
    expect(verdict.loadIssues.some((issue) => issue.code === "manifest_ref_mismatch" || issue.code === "manifest_checksum_mismatch")).toBe(true);
  });

  it("runs the smoke script end to end", () => {
    const stdout = execFileSync(
      "node",
      [
        "--experimental-transform-types",
        "scripts/replay-cold-start-router-smoke.ts",
      ],
      {
        cwd: repoRoot,
        encoding: "utf8",
      },
    );

    expect(stdout).toContain("Cold-start router replay smoke: ok");
    expect(stdout).toContain("verdict: pass");
    expect(stdout).toContain("passedRows: 2/2");
    expect(stdout).toContain("manifestChecksum:");
  });
});
