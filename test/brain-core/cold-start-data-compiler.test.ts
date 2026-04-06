import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  compileColdStartDocsQaSourceBundleFromFileV1,
  compileColdStartDocsQaSourceBundleV1,
  summarizeColdStartDocsQaCompilationBundleV1,
  type ColdStartDocsQaCompilationBundleV1,
  type RawDocsQaSourceBundleV1,
} from "../../src/brain-core/cold-start-data-compiler.js";
import {
  validateDataRegistryEntryV1,
  validateGraphCompilerContractV1,
  validateRouteDecisionRowV1,
  validateTeacherLabelContractV1,
  validateVerifierContractV1,
} from "../../src/brain-core/cold-start-router-contracts.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const sampleBundlePath = path.join(repoRoot, "artifacts", "cold-start-router-sample", "docs-qa-sample.raw.json");

function sha256Text(text: string): string {
  return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

function loadSampleBundle(): RawDocsQaSourceBundleV1 {
  return JSON.parse(readFileSync(sampleBundlePath, "utf8")) as RawDocsQaSourceBundleV1;
}

describe("cold-start docs/QA compilation", () => {
  it("compiles the checked sample fixture into governed registry, graph, route-decision, teacher-label, and verifier contracts", () => {
    const bundle = compileColdStartDocsQaSourceBundleFromFileV1({ bundlePath: sampleBundlePath, repoRoot });

    expect(validateDataRegistryEntryV1(bundle.registryEntry)).toMatchObject({ valid: true });
    expect(validateGraphCompilerContractV1(bundle.graphCompiler)).toMatchObject({ valid: true });
    expect(bundle.routeRows).toHaveLength(2);
    expect(bundle.teacherLabels).toHaveLength(2);
    expect(bundle.verifiers).toHaveLength(2);
    expect(bundle.report).toMatchObject({
      contract: "cold_start_docs_qa_compilation_report.v1",
      bundleId: "docs-qa-sample-v1",
      datasetId: "dataset_docs_qa_openclawbrain_v1",
      acceptedRowCount: 2,
      rejectedRowCount: 0,
      cleanup: {
        queryTrimmedCount: 2,
        rationaleTrimmedCount: 2,
        cursorPathSegmentTrimCount: 3,
        candidateDedupedCount: 2,
        evidenceSpanDedupedCount: 2,
        hardNegativeDedupedCount: 4,
        confidenceClampCount: 1,
      },
    });

    const [routingPriorRow, lintRow] = bundle.routeRows;
    expect(validateRouteDecisionRowV1(routingPriorRow)).toMatchObject({ valid: true });
    expect(validateRouteDecisionRowV1(lintRow)).toMatchObject({ valid: true });
    expect(validateTeacherLabelContractV1(bundle.teacherLabels[0])).toMatchObject({ valid: true });
    expect(validateTeacherLabelContractV1(bundle.teacherLabels[1])).toMatchObject({ valid: true });
    expect(validateVerifierContractV1(bundle.verifiers[0])).toMatchObject({ valid: true });
    expect(validateVerifierContractV1(bundle.verifiers[1])).toMatchObject({ valid: true });

    expect(routingPriorRow.query).toBe("What should win when summary and correction memory conflict?");
    expect(routingPriorRow.cursor_path).toEqual([
      "docs/architecture/routing-prior.md",
      "typed-memory/corrections",
    ]);
    expect(routingPriorRow.candidate_set.map((candidate) => candidate.candidate_id)).toEqual([
      "doc:routing-prior:conflict-resolution",
      "doc:teacher-v3-lints:ci-first",
      "tool:openclawbrain-proof",
    ]);
    expect(routingPriorRow.evidence_spans).toHaveLength(2);
    expect(routingPriorRow.hard_negatives).toEqual([
      "doc:teacher-v3-lints:ci-first",
      "tool:openclawbrain-proof",
    ]);
    expect(bundle.teacherLabels[0].confidence).toBe(1);
    expect(bundle.teacherLabels[0].best_next_node_ids).toEqual(["doc:routing-prior:conflict-resolution"]);
    expect(lintRow.candidate_set.map((candidate) => candidate.candidate_id)).toEqual([
      "doc:teacher-v3-lints:ci-first",
      "doc:teacher-v3-lints:semantic-audit",
      "tool:openclawbrain-proof",
    ]);
    expect(lintRow.hard_negatives).toEqual([
      "doc:teacher-v3-lints:semantic-audit",
      "tool:openclawbrain-proof",
    ]);

    const routingPriorDoc = readFileSync(path.join(repoRoot, "docs", "architecture", "routing-prior.md"), "utf8");
    const teacherLintsDoc = readFileSync(path.join(repoRoot, "docs", "architecture", "teacher-v3-lints.md"), "utf8");
    expect(bundle.registryEntry.file_hashes["docs/architecture/routing-prior.md"]).toBe(sha256Text(routingPriorDoc));
    expect(bundle.registryEntry.file_hashes["docs/architecture/teacher-v3-lints.md"]).toBe(sha256Text(teacherLintsDoc));

    expect(summarizeColdStartDocsQaCompilationBundleV1(bundle)).toMatchObject({
      registry: {
        datasetId: "dataset_docs_qa_openclawbrain_v1",
        sourceFamily: "docs",
        approvalStatus: "approved_train",
        fileCount: 2,
      },
      graphCompiler: {
        sourceFamily: "docs",
      },
      routeRows: [
        {
          rowId: "docs-qa-routing-prior-conflict",
          candidateCount: 3,
          hardNegativeCount: 2,
          stopLabel: "CONTINUE",
          splitTag: "train",
        },
        {
          rowId: "docs-qa-ci-first-deterministic-lints",
          candidateCount: 3,
          hardNegativeCount: 2,
          stopLabel: "CONTINUE",
          splitTag: "eval",
        },
      ],
    });
  });

  it("preserves tool-driven STOP_LOCAL supervision as first-class row, label, and verifier evidence", () => {
    const raw = loadSampleBundle();
    const toolStopExample: RawDocsQaSourceBundleV1 = {
      ...raw,
      bundle_id: "docs-qa-sample-v1-tool-stop-local",
      examples: [
        {
          ...raw.examples[0],
          row_id: "docs-qa-tool-stop-local",
          query: "When the proof tool is enough, stop locally.",
          teacher_action: {
            kind: "tool",
            tool_name: "tool:openclawbrain-proof",
          },
          stop_label: "STOP_LOCAL",
          hard_negatives: ["doc:teacher-v3-lints:ci-first"],
          split_tag: "train",
          created_at: "2026-04-05T23:03:00Z",
        },
      ],
    };

    const bundle = compileColdStartDocsQaSourceBundleV1({
      repoRoot,
      bundle: toolStopExample,
    });

    expect(bundle.routeRows).toHaveLength(1);
    expect(bundle.teacherLabels).toHaveLength(1);
    expect(bundle.verifiers).toHaveLength(1);
    expect(bundle.routeRows[0]).toMatchObject({
      row_id: "docs-qa-tool-stop-local",
      stop_label: "STOP_LOCAL",
      teacher_action: {
        kind: "tool",
        tool_name: "tool:openclawbrain-proof",
      },
      hard_negatives: ["doc:routing-prior:conflict-resolution", "doc:teacher-v3-lints:ci-first"],
    });
    expect(bundle.teacherLabels[0]).toMatchObject({
      row_id: "docs-qa-tool-stop-local",
      stop_label: "STOP_LOCAL",
      best_next_node_ids: ["tool:openclawbrain-proof"],
      best_next_tool_name: "tool:openclawbrain-proof",
    });
    expect(bundle.verifiers[0]?.checks.some((check) => check.check_id === "tool_alignment")).toBe(true);
    expect(bundle.report).toMatchObject({
      acceptedRowCount: 1,
      rejectedRowCount: 0,
      supervision: {
        stopLabelCounts: {
          STOP_LOCAL: 1,
        },
        teacherActionKindCounts: {
          tool: 1,
        },
      },
    });
  });

  it("rejects rows whose hard negatives overlap positive targets while preserving the accepted rows", () => {
    const raw = loadSampleBundle();
    const broken: RawDocsQaSourceBundleV1 = {
      ...raw,
      bundle_id: "docs-qa-sample-v1-broken",
      examples: raw.examples.map((example, index) =>
        index === 0
          ? {
              ...example,
              hard_negatives: [
                ...(example.hard_negatives ?? []),
                "doc:routing-prior:conflict-resolution",
              ],
            }
          : example,
      ),
    };

    const bundle: ColdStartDocsQaCompilationBundleV1 = compileColdStartDocsQaSourceBundleV1({
      repoRoot,
      bundle: broken,
    });

    expect(bundle.routeRows).toHaveLength(1);
    expect(bundle.teacherLabels).toHaveLength(1);
    expect(bundle.verifiers).toHaveLength(1);
    expect(bundle.report).toMatchObject({
      acceptedRowCount: 1,
      rejectedRowCount: 1,
    });
    expect(bundle.report.rowSummaries[0]).toMatchObject({
      rowId: "docs-qa-routing-prior-conflict",
      accepted: false,
    });
    expect(bundle.report.issues.some((issue) => issue.code === "hard_negative_overlap")).toBe(true);
    expect(bundle.routeRows[0].row_id).toBe("docs-qa-ci-first-deterministic-lints");
  });
});
