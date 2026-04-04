import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import type { EvidenceRef } from "../../src/brain-core/types.js";
import {
  describeTeacherProposalReplayGate,
  summarizeTeacherProposalV1,
  type TeacherProposal,
  type TeacherProposalProofBundleV1,
} from "../../src/brain-core/teacher-v3-contracts.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";

const tempDirs: string[] = [];

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "brain-store-teacher-proposals-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);
  return new BrainStore(db);
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

const compilerEvidence: EvidenceRef = {
  evidenceId: "evi_compiler_01",
  sourceKind: "user_turn",
  sourceId: "turn_compiler_01",
  span: { start: 0, end: 64 },
  authority: "user_explicit",
  derivation: "summary_navigation",
  excerpt: "Keep the Teacher v3 proposal substrate durable.",
  sourceHash: "sha256:compiler-01",
  capturedAt: "2026-04-03T18:26:00Z",
};

const lintEvidence: EvidenceRef = {
  evidenceId: "evi_lint_01",
  sourceKind: "file",
  sourceId: "docs/README.md#current-version",
  authority: "raw_source",
  derivation: "teacher_lint",
  excerpt: "Current version still points at 0.4.26.",
  sourceHash: "sha256:lint-01",
};

const lintCounterevidence: EvidenceRef = {
  evidenceId: "evi_lint_counter_01",
  sourceKind: "file",
  sourceId: "docs/changelog.md#0-4-27",
  authority: "raw_source",
  derivation: "teacher_lint",
  excerpt: "Changelog already notes 0.4.27.",
  sourceHash: "sha256:lint-counter-01",
};

function makeCompilerProposal(): TeacherProposal {
  return {
    proposalId: "prop_compiler_01",
    proposalClass: "compiler",
    lane: "compiler",
    status: "proposed",
    lineage: {
      proposalClass: "compiler",
      basePackVersion: 7,
      baseGraphHash: "graph_sha_compiler_01",
      producerVersion: "teacher-v3@0.1.0",
      producerBuildId: "build_compiler_01",
      promptHash: "prompt_sha_compiler_01",
      templateId: "teacher-v3/compiler-v1",
      scope: "docs/architecture/compiler-persistence",
      profile: "default",
      idempotencyKey: "teacher-v3::compiler::docs/architecture/compiler-persistence",
      sourceBundleId: "bundle_compiler_01",
      parentProposalIds: ["prop_seed_01"],
    },
    subjectIds: ["doc:teacher-v3", "doc:compiled-artifacts"],
    evidence: [compilerEvidence],
    payload: {
      kind: "compiler-persistence",
      target: "teacher-v3",
      summary: "Persist compiler proposals with stable identity and lineage.",
    },
    expectedEffect: {
      retrieval: "better",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: 0.96,
    replaySuites: ["compiler-shape-smoke", "compiler-lineage-smoke"],
    rollbackKey: "rollback:teacher-v3:compiler:persistence",
    replayGate: describeTeacherProposalReplayGate("compiler"),
    createdAt: "2026-04-03T18:26:00Z",
  };
}

function makeLintProposal(): TeacherProposal {
  return {
    proposalId: "prop_lint_01",
    proposalClass: "lint",
    lane: "lint",
    status: "validated",
    lineage: {
      proposalClass: "lint",
      basePackVersion: 7,
      baseGraphHash: "graph_sha_lint_01",
      producerVersion: "teacher-v3@0.1.0",
      producerBuildId: "build_lint_01",
      promptHash: "prompt_sha_lint_01",
      templateId: "teacher-v3/lint-v1",
      scope: "release-drift",
      profile: "default",
      idempotencyKey: "teacher-v3::lint::release-drift",
      sourceBundleId: "bundle_lint_01",
      parentProposalIds: ["prop_seed_02"],
    },
    subjectIds: ["doc:README", "doc:HowItWorks", "doc:proof-page"],
    evidence: [lintEvidence],
    counterevidence: [lintCounterevidence],
    payload: {
      kind: "release-drift",
      currentVersion: "0.4.26",
      targetVersion: "0.4.27",
      surfaces: ["README", "HowItWorks", "proof-page"],
    },
    expectedEffect: {
      retrieval: "same",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: 0.89,
    replaySuites: ["release-drift-smoke"],
    rollbackKey: "rollback:teacher-v3:lint:release-drift",
    replayGate: describeTeacherProposalReplayGate("lint"),
    createdAt: "2026-04-03T18:27:00Z",
  };
}

function makeCompilerProofBundle(proposal: TeacherProposal): TeacherProposalProofBundleV1 {
  return {
    bundleId: "pb_compiler_01",
    proposalId: proposal.proposalId,
    proposalClass: proposal.proposalClass,
    status: "promoted",
    lineage: proposal.lineage,
    rollbackKey: proposal.rollbackKey,
    replaySuites: proposal.replaySuites,
    surfaceMap: [
      {
        id: "runtime_before",
        state: "shipped",
        phase: "before",
        kind: "runtime_truth",
        source: "openclawbrain status --detailed",
        note: "runtime surface before proposal persistence",
      },
      {
        id: "proposal_after",
        state: "target",
        phase: "after",
        kind: "proposal_truth",
        source: "brain_teacher_proposals",
        note: "stored proposal after persistence",
      },
    ],
    evidenceLinks: [
      {
        refId: compilerEvidence.evidenceId ?? compilerEvidence.sourceId,
        kind: "source",
        path: "docs/architecture/teacher-v3-proposals.md#proposal-envelope",
      },
    ],
    summary: "Persisted compiler proposal keeps lineage, rollback identity, and replay metadata intact.",
    createdAt: "2026-04-03T18:30:00Z",
  };
}

describe("BrainStore teacher proposals", () => {
  it("round-trips compiler and lint proposals with stable identity, lineage, and summaries", () => {
    const store = setup();
    const compiler = makeCompilerProposal();
    const lint = makeLintProposal();

    store.insertTeacherProposal(compiler);
    store.insertTeacherProposal(lint);

    store.updateTeacherProposalStatus({
      proposalId: compiler.proposalId,
      status: "promoted",
      resolvedAt: "2026-04-03T18:31:00Z",
      proofBundle: makeCompilerProofBundle(compiler),
    });

    const loadedCompiler = store.getTeacherProposal(compiler.proposalId);
    expect(loadedCompiler).not.toBeNull();
    expect(loadedCompiler).toMatchObject({
      proposalId: compiler.proposalId,
      proposalClass: "compiler",
      lane: "compiler",
      status: "promoted",
      rollbackKey: compiler.rollbackKey,
      replaySuites: compiler.replaySuites,
      resolvedAt: "2026-04-03T18:31:00Z",
    });
    expect(loadedCompiler?.proofBundle?.bundleId).toBe("pb_compiler_01");
    expect(loadedCompiler?.proofBundle?.status).toBe("promoted");
    expect(loadedCompiler?.proofBundle?.lineage.idempotencyKey).toBe(compiler.lineage.idempotencyKey);
    expect(loadedCompiler?.replayGate?.reviewMode).toBe("promotable");

    const compilerByKey = store.getTeacherProposalByIdempotencyKey(compiler.lineage.idempotencyKey);
    expect(compilerByKey?.proposalId).toBe(compiler.proposalId);
    expect(compilerByKey?.status).toBe("promoted");

    const compilerSummary = store.summarizeTeacherProposal(compiler.proposalId);
    expect(compilerSummary).toMatchObject({
      proposalId: compiler.proposalId,
      proposalClass: "compiler",
      status: "promoted",
      subjectCount: 2,
      evidenceCount: 1,
      counterevidenceCount: 0,
      replaySuiteCount: 2,
      rollbackKey: compiler.rollbackKey,
      hasProofBundle: true,
      proofBundleId: "pb_compiler_01",
      proofBundleStatus: "promoted",
      lineage: {
        proposalClass: "compiler",
        basePackVersion: 7,
        baseGraphHash: "graph_sha_compiler_01",
        producerVersion: "teacher-v3@0.1.0",
        producerBuildId: "build_compiler_01",
        promptHash: "prompt_sha_compiler_01",
        templateId: "teacher-v3/compiler-v1",
        scope: "docs/architecture/compiler-persistence",
        profile: "default",
        idempotencyKey: compiler.lineage.idempotencyKey,
        sourceBundleId: "bundle_compiler_01",
        parentProposalIds: ["prop_seed_01"],
      },
    });

    const lintSummary = store.summarizeTeacherProposal(lint.proposalId);
    expect(lintSummary).toMatchObject({
      proposalId: lint.proposalId,
      proposalClass: "lint",
      status: "validated",
      subjectCount: 3,
      evidenceCount: 1,
      counterevidenceCount: 1,
      replaySuiteCount: 1,
      rollbackKey: lint.rollbackKey,
      hasProofBundle: false,
      proofBundleId: undefined,
      proofBundleStatus: undefined,
      lineage: {
        proposalClass: "lint",
        basePackVersion: 7,
        baseGraphHash: "graph_sha_lint_01",
        producerVersion: "teacher-v3@0.1.0",
        producerBuildId: "build_lint_01",
        promptHash: "prompt_sha_lint_01",
        templateId: "teacher-v3/lint-v1",
        scope: "release-drift",
        profile: "default",
        idempotencyKey: lint.lineage.idempotencyKey,
        sourceBundleId: "bundle_lint_01",
        parentProposalIds: ["prop_seed_02"],
      },
    });

    const compilerRows = store.getTeacherProposalsByClass("compiler");
    expect(compilerRows).toHaveLength(1);
    expect(compilerRows[0]?.proposalId).toBe(compiler.proposalId);

    const diff = store.diffTeacherProposals(compiler.proposalId, lint.proposalId);
    expect(diff).not.toBeNull();
    expect(diff).toMatchObject({
      leftProposalId: compiler.proposalId,
      rightProposalId: lint.proposalId,
      sameProposalClass: false,
      sameIdempotencyKey: false,
      sameRollbackKey: false,
      sameStatus: false,
      subjectIds: {
        added: lint.subjectIds,
        removed: compiler.subjectIds,
      },
      evidenceIds: {
        added: [lintEvidence.evidenceId ?? lintEvidence.sourceId],
        removed: [compilerEvidence.evidenceId ?? compilerEvidence.sourceId],
      },
      counterevidenceIds: {
        added: [lintCounterevidence.evidenceId ?? lintCounterevidence.sourceId],
        removed: [],
      },
      replaySuites: {
        added: lint.replaySuites,
        removed: compiler.replaySuites,
      },
    });
    expect(diff?.changedFields).toEqual(
      expect.arrayContaining([
        "proposalClass",
        "status",
        "rollbackKey",
        "subjectIds",
        "evidenceIds",
        "counterevidenceIds",
        "replaySuites",
        "lineage.scope",
        "lineage.idempotencyKey",
      ]),
    );
    expect(diff?.summary).toContain("compiler prop_compiler_01 → prop_lint_01");
  });
});
