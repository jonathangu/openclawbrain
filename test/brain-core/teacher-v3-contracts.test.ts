import { describe, expect, it } from "vitest";
import type {
  CompiledArtifactMeta,
  EvidenceRef,
  ProposalLineage,
  TeacherProposal,
} from "../../src/brain-core/teacher-v3-contracts.js";
import {
  RETENTION_STATE_TRANSITIONS,
  evaluateRetentionTransitionV1,
} from "../../src/brain-core/teacher-v3-contracts.js";

const evidence: EvidenceRef = {
  evidenceId: "evi_01",
  sourceKind: "user_turn",
  sourceId: "turn_2026_04_03_001",
  span: { start: 12, end: 52 },
  authority: "user_explicit",
  derivation: "summary_navigation",
  excerpt: "Use the new Teacher v3 substrate.",
  sourceHash: "sha256:abc123",
  capturedAt: "2026-04-03T18:26:00Z",
};

const lineage: ProposalLineage = {
  proposalClass: "compiler",
  basePackVersion: 42,
  baseGraphHash: "graph_sha_01",
  producerVersion: "teacher-v3@0.1.0",
  producerBuildId: "build_abc123",
  promptHash: "prompt_sha_01",
  templateId: "teacher-v3/compiler-v1",
  scope: "docs/architecture",
  profile: "default",
  idempotencyKey: "teacher-v3::docs/architecture::compiled-artifacts",
  sourceBundleId: "bundle_01",
  parentProposalIds: ["prop_parent_01"],
};

const proposal: TeacherProposal = {
  proposalId: "prop_01",
  proposalClass: "compiler",
  status: "proposed",
  lineage,
  subjectIds: ["topic:teacher-v3", "doc:compiled-artifacts"],
  evidence: [evidence],
  counterevidence: [],
  payload: {
    kind: "compiled-artifact-substrate",
    title: "Teacher v3 compiled artifact substrate",
  },
  expectedEffect: {
    retrieval: "better",
    truthRisk: "low",
    tokenBudget: "same",
  },
  confidence: 0.94,
  replaySuites: ["teacher-v3/compiler-replay"],
  rollbackKey: "rollback::teacher-v3::compiler::docs-architecture",
  expiresAt: "2026-04-10T18:26:00Z",
  createdAt: "2026-04-03T18:26:00Z",
  artifacts: [
    {
      artifactId: "ca_01",
      kind: "concept_page",
      contentHash: "sha256:def456",
    },
  ],
};

const artifact: CompiledArtifactMeta = {
  schemaVersion: 1,
  artifactId: "ca_01",
  kind: "concept_page",
  title: "Teacher v3 compiled artifact substrate",
  status: "proposed",
  packId: "pack_01",
  proposalId: proposal.proposalId,
  proposalLane: "compiler",
  subjectIds: proposal.subjectIds,
  evidence: [evidence],
  counterevidence: [],
  provenance: {
    producer: "teacher-v3",
    producerVersion: "teacher-v3@0.1.0",
    promptHash: "prompt_sha_01",
    runId: "run_01",
    basePackId: "pack_01",
    baseGraphHash: "graph_sha_01",
    scope: "docs/architecture",
    idempotencyKey: lineage.idempotencyKey,
    sourceRoots: ["docs/architecture"],
    transformChain: ["extract", "cluster", "synthesize", "validate"],
  },
  contentHash: "sha256:def456",
  markdownPath: "compiled/packs/pack_01/artifacts/ca_01/artifact.md",
  metaPath: "compiled/packs/pack_01/artifacts/ca_01/artifact.meta.json",
  createdAt: "2026-04-03T18:26:00Z",
  updatedAt: "2026-04-03T18:27:00Z",
  confidence: 0.92,
  claims: [
    {
      claimId: "claim_01",
      text: "Teacher v3 compiled artifacts are derived, not authoritative.",
      evidenceIds: [evidence.evidenceId ?? ""],
      confidence: 0.98,
      status: "supported",
    },
  ],
  promotion: {
    replaySuites: ["teacher-v3/compiler-replay"],
    rollbackKey: proposal.rollbackKey,
  },
};

describe("teacher v3 contracts", () => {
  it("keeps evidence, proposal lineage, and compiled artifact metadata aligned", () => {
    expect(proposal.proposalClass).toBe("compiler");
    expect(proposal.lineage.idempotencyKey).toBe(lineage.idempotencyKey);
    expect(proposal.evidence[0]).toMatchObject({
      sourceKind: "user_turn",
      authority: "user_explicit",
      derivation: "summary_navigation",
    });
    expect(artifact.proposalId).toBe(proposal.proposalId);
    expect(artifact.provenance.basePackId).toBe("pack_01");
    expect(artifact.claims?.[0]?.evidenceIds).toContain("evi_01");
  });

  it("keeps teacher-forgetting retention fail-closed and protects user_explicit corrections from hard delete", () => {
    expect(RETENTION_STATE_TRANSITIONS).toMatchObject({
      retained: ["retained", "demoted", "archived", "tombstoned"],
      demoted: ["demoted", "archived", "tombstoned"],
      archived: ["archived", "tombstoned"],
      tombstoned: ["tombstoned", "deleted"],
      deleted: ["deleted"],
    });

    const userExplicitCorrection = {
      sourceId: "bn_correction_01",
      sourceKind: "correction" as const,
      authority: "user_explicit" as const,
    };
    const rawSourceMemory = {
      sourceId: "bn_source_01",
      sourceKind: "summary" as const,
      authority: "raw_source" as const,
    };

    expect(
      evaluateRetentionTransitionV1({
        current: "retained",
        requested: "archive",
        target: rawSourceMemory,
      }),
    ).toMatchObject({
      allowed: true,
      to: "archived",
    });

    expect(
      evaluateRetentionTransitionV1({
        current: "retained",
        requested: "hard_delete",
        target: rawSourceMemory,
      }),
    ).toMatchObject({
      allowed: false,
      guardrail: "requires_tombstoned_prestate",
    });

    expect(
      evaluateRetentionTransitionV1({
        current: "tombstoned",
        requested: "hard_delete",
        target: rawSourceMemory,
      }),
    ).toMatchObject({
      allowed: true,
      to: "deleted",
    });

    const blocked = evaluateRetentionTransitionV1({
      current: "tombstoned",
      requested: "hard_delete",
      target: userExplicitCorrection,
    });
    expect(blocked.allowed).toBe(false);
    expect(blocked.guardrail).toBe("deny_hard_delete_user_explicit");
    expect(blocked.reason).toContain("user_explicit correction memory");
  });
});
