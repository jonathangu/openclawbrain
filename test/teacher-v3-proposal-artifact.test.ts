import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { TeacherProposal, TeacherProposalProofBundleV1 } from "../src/brain-core/teacher-v3-contracts.js";
import {
  buildTeacherProposalReportArtifactV1,
  renderTeacherProposalReportArtifactMarkdownV1,
  TEACHER_V3_PROPOSAL_ARTIFACT_CONTRACT,
} from "../src/brain-core/teacher-v3-proposal-artifact.js";
import { teacherProposalArtifactJobResult } from "../src/brain-worker/teacher-job.js";
import {
  buildTeacherV3ProposalArtifact,
  TEACHER_V3_PROPOSAL_ARTIFACT_LAYOUT,
  writeTeacherV3ProposalArtifact,
} from "../scripts/teacher-v3-proposal-artifact.mjs";

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function makeCompilerReplaySummary(proposal: TeacherProposal): NonNullable<TeacherProposal["replaySummary"]> {
  return {
    replayId: "treplay_compiler_artifact_01",
    proposalId: proposal.proposalId,
    proposalClass: "compiler",
    status: "promotable",
    reviewMode: "promotable",
    basePackVersion: 7,
    baseGraphHash: "graph_sha_compiler_base",
    candidatePackVersion: 8,
    candidatePackId: "candidate_pack_08",
    candidateGraphHash: "graph_sha_compiler_candidate",
    beforeScore: 0.62,
    afterScore: 0.79,
    scoreDelta: 0.17,
    before: {
      phase: "before",
      surfaceState: "shipped",
      packVersion: 7,
      packId: null,
      graphHash: "graph_sha_compiler_base",
      nodeCount: 10,
      edgeCount: 11,
      health: {
        firedPerQuery: 1.1,
        dormantPercent: 0.2,
        orphanCount: 1,
      },
      notes: ["base lineage"],
    },
    after: {
      phase: "after",
      surfaceState: "target",
      packVersion: 8,
      packId: "candidate_pack_08",
      graphHash: "graph_sha_compiler_candidate",
      nodeCount: 12,
      edgeCount: 13,
      health: {
        firedPerQuery: 1.8,
        dormantPercent: 0.1,
        orphanCount: 0,
      },
      notes: ["candidate pack replay"],
    },
    classSummary: {
      kind: "compiler",
      reviewMode: "promotable",
      promotionDiscipline: "promotable",
      subjectCount: 2,
      evidenceCount: 1,
      counterevidenceCount: 0,
      replaySuites: proposal.replaySuites,
      candidatePackVersion: 8,
      candidatePackId: "candidate_pack_08",
      candidateGraphHash: "graph_sha_compiler_candidate",
      summary: "Compiler replay is promotable on candidate pack candidate_pack_08; evidence-backed lineage stays intact and the candidate graph is distinct from base state.",
      notes: [
        "basePackVersion=7",
        "baseGraphHash=graph_sha_compiler_base",
      ],
    },
    summary: "compiler replay accepted on candidate_pack_08; before=0.620 after=0.790 delta=0.170",
    createdAt: "2026-04-19T18:00:00Z",
    updatedAt: "2026-04-19T18:00:00Z",
  };
}

function makeCompilerProposal(): TeacherProposal {
  const proposal: TeacherProposal = {
    proposalId: "prop_compiler_artifact_01",
    proposalClass: "compiler",
    proposalKind: "compiled_artifact",
    lane: "compiler",
    status: "promotable",
    lineage: {
      proposalClass: "compiler",
      basePackVersion: 7,
      baseGraphHash: "graph_sha_compiler_base",
      producerVersion: "teacher-v3@0.1.0",
      producerBuildId: "build_compiler_artifact_01",
      promptHash: "prompt_sha_compiler_artifact_01",
      templateId: "teacher-v3/compiler-v1",
      scope: "docs/compiler-artifact-lane",
      profile: "default",
      idempotencyKey: "teacher-v3::compiler::artifact-lane",
      sourceBundleId: "bundle_compiler_artifact_01",
      parentProposalIds: ["seed_compiler_artifact_00"],
    },
    subjectIds: ["doc:teacher-v3-proof", "doc:compiled-artifacts"],
    evidence: [
      {
        evidenceId: "evi_compiler_artifact_01",
        sourceKind: "file",
        sourceId: "docs/architecture/teacher-v3-proof.md#target-state-surfaces",
        authority: "raw_source",
        derivation: "teacher_compilation",
        excerpt: "proposal-report.json should include lineage, status, replay gate dimensions, and evidence refs.",
        sourceHash: "sha256:compiler-artifact-01",
      },
    ],
    counterevidence: [],
    payload: {
      kind: "compiled_artifact",
      summary: "Emit a compact, report-only compiler proposal artifact.",
    },
    expectedEffect: {
      retrieval: "better",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: 0.93,
    replaySuites: ["compiler-shape-smoke", "compiler-proof-linkage-smoke"],
    rollbackKey: "rollback:teacher-v3:compiler:artifact-lane",
    createdAt: "2026-04-19T17:58:00Z",
    artifacts: [
      {
        artifactId: "ca_compiler_report_01",
        kind: "concept_page",
        contentHash: "sha256:compiler-artifact-ref-01",
      },
    ],
  };

  proposal.replaySummary = makeCompilerReplaySummary(proposal);
  proposal.proofBundle = makeCompilerProofBundle(proposal);
  return proposal;
}

function makeCompilerProofBundle(proposal: TeacherProposal): TeacherProposalProofBundleV1 {
  return {
    bundleId: "pb_compiler_artifact_01",
    proposalId: proposal.proposalId,
    proposalClass: "compiler",
    status: "promoted",
    lineage: proposal.lineage,
    rollbackKey: proposal.rollbackKey,
    replaySuites: proposal.replaySuites,
    replayOutcomes: [
      {
        outcomeId: "compiler-proof-linkage-smoke:1",
        replaySuite: "compiler-proof-linkage-smoke",
        proposalClass: "compiler",
        reviewMode: "promotable",
        result: "pass",
        source: "proposal_record",
        summary: "Compiler proof linkage stayed rollback-bound and reviewable.",
        capturedAt: "2026-04-19T18:01:00Z",
      },
    ],
    surfaceMap: [
      {
        id: "runtime-truth",
        state: "shipped",
        phase: "before",
        kind: "runtime_truth",
        source: "openclawbrain status --detailed",
      },
      {
        id: "docs-truth",
        state: "shipped",
        phase: "before",
        kind: "docs_truth",
        source: "docs/architecture/teacher-v3-proof.md",
      },
      {
        id: "proposal-report",
        state: "target",
        phase: "after",
        kind: "proposal_truth",
        source: "artifacts/teacher-v3-proof/proposal-report.json",
      },
    ],
    evidenceLinks: [
      {
        refId: "evi_compiler_artifact_01",
        kind: "teacher_compilation",
        path: "docs/architecture/teacher-v3-proof.md#target-state-surfaces",
      },
    ],
    summary: "Compiler proposal proof bundle stayed compact and rollback-bound.",
    createdAt: "2026-04-19T18:01:00Z",
  };
}

function makeLintProposal(): TeacherProposal {
  return {
    proposalId: "prop_lint_artifact_01",
    proposalClass: "lint",
    proposalKind: "duplicate_report",
    lane: "lint",
    status: "validated",
    lineage: {
      proposalClass: "lint",
      basePackVersion: 7,
      baseGraphHash: "graph_sha_lint_base",
      producerVersion: "teacher-v3@0.1.0",
      producerBuildId: "build_lint_artifact_01",
      promptHash: "prompt_sha_lint_artifact_01",
      templateId: "teacher-v3/lint-v1",
      scope: "release-drift",
      profile: "default",
      idempotencyKey: "teacher-v3::lint::artifact-lane",
      sourceBundleId: "bundle_lint_artifact_01",
      parentProposalIds: ["seed_lint_artifact_00"],
    },
    subjectIds: ["doc:README", "doc:proof-page"],
    evidence: [
      {
        evidenceId: "evi_lint_artifact_01",
        sourceKind: "file",
        sourceId: "docs/architecture/teacher-v3-lints.md#3-release-drift-motivating-case",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "Objective mismatches should be caught before semantic audits run.",
        sourceHash: "sha256:lint-artifact-01",
      },
    ],
    counterevidence: [
      {
        evidenceId: "cevi_lint_artifact_01",
        sourceKind: "file",
        sourceId: "docs/changelog.md#0-4-29",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "Release notes already mention 0.4.29 in one surface.",
        sourceHash: "sha256:lint-artifact-counter-01",
      },
    ],
    payload: {
      kind: "duplicate_report",
      summary: "Keep release drift findings report-only until replay and proof linkage are attached.",
    },
    expectedEffect: {
      retrieval: "same",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: 0.87,
    replaySuites: ["release-docs-drift-smoke"],
    rollbackKey: "rollback:teacher-v3:lint:artifact-lane",
    createdAt: "2026-04-19T17:59:00Z",
  };
}

describe("teacher v3 proposal artifact lane", () => {
  it("builds a compiler report-only artifact with replay and proof linkage", () => {
    const artifact = buildTeacherProposalReportArtifactV1({
      proposal: makeCompilerProposal(),
    });

    expect(artifact).toMatchObject({
      contract: TEACHER_V3_PROPOSAL_ARTIFACT_CONTRACT,
      proposalId: "prop_compiler_artifact_01",
      proposalClass: "compiler",
      reviewMode: "promotable",
      reviewDiscipline: "report_only",
      status: "promotable",
      replayHook: {
        replayReady: true,
        replaySummaryId: "treplay_compiler_artifact_01",
        placeholder: false,
      },
      proofLinkage: {
        proofBundleId: "pb_compiler_artifact_01",
        rollbackBound: true,
        proofLinked: true,
      },
      proofBundleSummary: {
        bundleId: "pb_compiler_artifact_01",
        surfaceCount: 3,
      },
    });
    expect(artifact.summary).toContain("report-only");
    expect(artifact.artifactRef).toMatchObject({
      artifactId: "teacher-v3-proposal-prop_compiler_artifact_01",
      kind: "proposal_review_bundle",
    });
    expect(artifact.gateMatrix.rows[2]?.status).toBe("pass");
    expect(artifact.gateMatrix.rows[4]?.status).toBe("pass");
    expect(artifact.attachedArtifacts).toHaveLength(1);

    const markdown = renderTeacherProposalReportArtifactMarkdownV1(artifact);
    expect(markdown).toContain("Teacher v3 proposal artifact");
    expect(markdown).toContain("Proof linkage");
    expect(markdown).toContain("ca_compiler_report_01");
  });

  it("keeps lint proposal artifacts report-only with replay placeholders and pending proof linkage", () => {
    const artifact = buildTeacherProposalReportArtifactV1({
      proposal: makeLintProposal(),
    });

    expect(artifact).toMatchObject({
      proposalId: "prop_lint_artifact_01",
      proposalClass: "lint",
      reviewMode: "promotable",
      reviewDiscipline: "report_only",
      replayHook: {
        replayReady: false,
        replaySummaryId: null,
        placeholder: true,
        replaySuites: ["release-docs-drift-smoke"],
      },
      proofLinkage: {
        proofBundleId: null,
        rollbackBound: false,
        proofLinked: false,
      },
    });
    expect(artifact.gateMatrix.rows[4]?.status).toBe("warn");
    expect(artifact.recommendations.join(" ")).toContain("Attach a candidate-pack replay summary");
    expect(artifact.recommendations.join(" ")).toContain("Attach a proof bundle");
    expect(artifact.summary).toContain("report-only");

    const markdown = renderTeacherProposalReportArtifactMarkdownV1(artifact);
    expect(markdown).toContain("Counterevidence refs");
    expect(markdown).toContain("release-docs-drift-smoke");
  });

  it("writes artifact.md and artifact.meta.json to disk", () => {
    const root = mkdtempSync(join(tmpdir(), "teacher-v3-proposal-artifact-"));
    tempDirs.push(root);

    const builtArtifact = buildTeacherV3ProposalArtifact({
      proposal: makeCompilerProposal(),
      outputDir: root,
    });
    const writeResult = writeTeacherV3ProposalArtifact(root, builtArtifact);

    expect(writeResult.writtenFiles).toHaveLength(2);

    const markdownPath = join(root, TEACHER_V3_PROPOSAL_ARTIFACT_LAYOUT.markdown);
    const metaPath = join(root, TEACHER_V3_PROPOSAL_ARTIFACT_LAYOUT.meta);
    const meta = JSON.parse(readFileSync(metaPath, "utf8"));

    expect(meta).toMatchObject({
      contract: TEACHER_V3_PROPOSAL_ARTIFACT_CONTRACT,
      proposalId: "prop_compiler_artifact_01",
      proofLinkage: {
        proofLinked: true,
      },
    });
    expect(readFileSync(markdownPath, "utf8")).toContain("Teacher v3 proposal artifact");
    expect(readFileSync(markdownPath, "utf8")).toContain("rollback bound: yes");
  });

  it("surfaces a bounded report-only teacher job result for proposal artifacts", () => {
    const artifact = buildTeacherProposalReportArtifactV1({
      proposal: makeCompilerProposal(),
    });

    const result = teacherProposalArtifactJobResult(artifact, {
      outputDir: "artifacts/teacher-v3-proposal-artifacts/prop_compiler_artifact_01",
    });

    expect(result).toMatchObject({
      job: "teacher",
      changed: false,
      details: {
        mode: "report_only",
        proposalId: "prop_compiler_artifact_01",
        proposalClass: "compiler",
        artifactId: "teacher-v3-proposal-prop_compiler_artifact_01",
        replayReady: true,
        proofLinked: true,
        outputDir: "artifacts/teacher-v3-proposal-artifacts/prop_compiler_artifact_01",
      },
    });
  });
});
