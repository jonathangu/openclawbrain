import { describe, expect, it } from "vitest";
import type {
  CompiledArtifactMeta,
  EvidenceRef,
  ProposalLineage,
  ProposalClass,
  TeacherCanaryRolloutPlanV1,
  TeacherProposalProofBundleV1,
  TeacherProposalReplayGateV1,
  TeacherProposal,
  TeacherV3LiveProofRungV1,
} from "../../src/brain-core/teacher-v3-contracts.js";
import {
  describeTeacherCanaryRolloutPlanV1,
  describeTeacherProposalReplayGate,
  isTeacherProposalPromotableClassV1,
  RETENTION_STATE_TRANSITIONS,
  TEACHER_CANARY_ROLLOUT_PLANS_V1,
  TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1,
  TEACHER_PROPOSAL_SHADOW_ONLY_CLASSES_V1,
  TEACHER_PROPOSAL_REPLAY_GATES_V1,
  evaluateRetentionTransitionV1,
  summarizeTeacherProposalProofBundleV1,
  summarizeTeacherV3LiveProofRungV1,
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
  status: "promoted",
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
  proofBundle: {
    bundleId: "pb_01",
    proposalId: "prop_01",
    proposalClass: "compiler",
    status: "promoted",
    lineage,
    rollbackKey: "rollback::teacher-v3::compiler::docs-architecture",
    replaySuites: ["teacher-v3/compiler-replay"],
    replayOutcomes: [
      {
        outcomeId: "replay_01",
        replaySuite: "teacher-v3/compiler-replay",
        proposalClass: "compiler",
        reviewMode: "promotable",
        result: "pass",
        source: "proposal_record",
        summary: "compiler replay passed",
        capturedAt: "2026-04-03T18:28:30Z",
      },
      {
        outcomeId: "replay_02",
        replaySuite: "teacher-v3/compiler-review",
        proposalClass: "compiler",
        reviewMode: "promotable",
        result: "warn",
        source: "proposal_record",
        summary: "compiler replay stayed bounded",
        capturedAt: "2026-04-03T18:28:45Z",
      },
    ],
    surfaceMap: [
      {
        id: "surface_runtime_status",
        state: "shipped",
        phase: "before",
        kind: "runtime_truth",
        source: "openclawbrain status --detailed",
        note: "canonical runtime truth surface",
      },
      {
        id: "surface_operator_proof",
        state: "shipped",
        phase: "before",
        kind: "proof_truth",
        source: "openclawbrain proof --openclaw-home ~/.openclaw",
        note: "operator proof bundle",
      },
      {
        id: "surface_teacher_v3_bundle",
        state: "target",
        phase: "after",
        kind: "proposal_truth",
        source: "artifacts/teacher-v3-proof/run-01",
        note: "proposal bundle overlay",
      },
    ],
    evidenceLinks: [
      {
        refId: evidence.evidenceId ?? "evi_01",
        kind: "source",
        path: "docs/architecture/teacher-v3-proposals.md#replay-and-rollback-rules",
      },
      {
        refId: "proof_verdict",
        kind: "operator-proof",
        path: "artifacts/2026-04-03/pb_01/verdict.json",
      },
    ],
    summary: "Promoted compiler proposal keeps a compact proof bundle with rollback identity and shipped-vs-target surface map.",
    createdAt: "2026-04-03T18:28:00Z",
    updatedAt: "2026-04-03T18:29:00Z",
  } satisfies TeacherProposalProofBundleV1,
  expiresAt: "2026-04-10T18:26:00Z",
  createdAt: "2026-04-03T18:26:00Z",
  resolvedAt: "2026-04-03T18:28:00Z",
  replayGate: describeTeacherProposalReplayGate("compiler"),
  canaryRollout: describeTeacherCanaryRolloutPlanV1("compiler"),
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
    proofBundle: proposal.proofBundle,
  },
};

const liveProofRung: TeacherV3LiveProofRungV1 = {
  rungId: "live-proof-rung-1",
  summary: "First live-proof rung for Teacher v3 progress",
  beforeSurfaces: [
    {
      id: "runtime-status-before",
      state: "shipped",
      phase: "before",
      kind: "runtime_truth",
      source: "openclawbrain status --openclaw-home ~/.openclaw --detailed",
      note: "canonical runtime truth before the change",
    },
    {
      id: "operator-proof-before",
      state: "shipped",
      phase: "before",
      kind: "proof_truth",
      source: "openclawbrain proof --openclaw-home ~/.openclaw",
      note: "host-anchored operator proof before the change",
    },
  ],
  afterSurfaces: [
    {
      id: "runtime-status-after",
      state: "shipped",
      phase: "after",
      kind: "runtime_truth",
      source: "openclawbrain status --openclaw-home ~/.openclaw --detailed",
      note: "canonical runtime truth after the change",
    },
    {
      id: "teacher-v3-target-after",
      state: "target",
      phase: "after",
      kind: "proposal_truth",
      source: "docs/architecture/teacher-v3-proof.md",
      note: "target-state teacher v3 proof bundle overlay",
    },
  ],
  checks: [
    {
      kind: "token",
      status: "pass",
      summary: "token budget stays explicit and bounded in the publication-safe proof view",
      evidenceSurfaceIds: ["runtime-status-before", "runtime-status-after"],
    },
    {
      kind: "latency",
      status: "pass",
      summary: "latency is surfaced as a bounded proof datum rather than a raw trace dump",
      evidenceSurfaceIds: ["operator-proof-before", "runtime-status-after"],
    },
    {
      kind: "truth",
      status: "pass",
      summary: "shipped status and proof remain authoritative while target-state surfaces stay labeled as target",
      evidenceSurfaceIds: ["runtime-status-before", "teacher-v3-target-after"],
    },
  ],
  publicationSafeArtifacts: [
    {
      artifactId: "teacher-v3-live-proof-summary",
      kind: "summary",
      path: "artifacts/teacher-v3-proof/live-proof-rung-1/summary.md",
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-live-proof-surface-map",
      kind: "surface-map",
      path: "artifacts/teacher-v3-proof/live-proof-rung-1/surface-map.json",
      redactions: ["raw source payloads"],
      containsRawLogs: false,
    },
  ],
  shippedStateNotes: [
    "The operator proof bundle is already shipped and cites host truth.",
    "The new live-proof rung remains target-state until it is wired into a writer.",
  ],
  targetStateNotes: [
    "Before/after surfaces stay explicit so the proof can show movement without implying authority changes.",
    "The publication-safe view stays bounded to token, latency, and truth checks plus redacted artifacts.",
  ],
};

const unsafePublicationArtifact = {
  artifactId: "teacher-v3-live-proof-unsafe",
  kind: "status",
  path: "artifacts/teacher-v3-proof/live-proof-rung-1/unsafe.json",
  redactions: [],
  containsRawLogs: true,
} as const;

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
    expect(proposal.replayGate?.reviewMode).toBe("promotable");
    expect(proposal.canaryRollout).toMatchObject({
      proposalClass: "compiler",
      surfaceState: "target",
      rolloutMode: "off",
      enabled: false,
    });
    expect(proposal.canaryRollout?.guardrails).toEqual(
      expect.arrayContaining([
        "Keep the rollout plan target-state only until it is explicitly shipped.",
        "Default rolloutMode stays off.",
        "Do not use the canary plan to change live serving without separate replay and rollback proof.",
        "Bind any candidate pack by durable version or id, never by ad hoc display labels.",
      ]),
    );
    expect(summarizeTeacherProposalProofBundleV1(proposal.proofBundle as TeacherProposalProofBundleV1)).toMatchObject({
      bundleId: "pb_01",
      proposalId: proposal.proposalId,
      proposalClass: "compiler",
      status: "promoted",
      rollbackKey: proposal.rollbackKey,
      replayOutcomeSummary: {
        replayOutcomeCount: 2,
        replaySuites: ["teacher-v3/compiler-replay", "teacher-v3/compiler-review"],
        resultCounts: { pass: 1, warn: 1, fail: 0 },
        reviewModeCounts: { promotable: 2, shadow_only: 0 },
        sourceCounts: { proposal_record: 2, proof_bundle: 0, derived: 0 },
      },
      surfaceCount: 3,
      shippedSurfaceCount: 2,
      targetSurfaceCount: 1,
      evidenceLinkCount: 2,
      counterevidenceLinkCount: 0,
      surfaceIds: ["surface_runtime_status", "surface_operator_proof", "surface_teacher_v3_bundle"],
    });
    expect(proposal.replayGate?.dimensions.truthInvariants.name).toBe("truth_invariants");
    expect(proposal.replayGate?.dimensions.attributionFloor.requirements).toContain(
      "Every proposal carries durable evidence refs.",
    );
    expect(TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1).toEqual(["compiler", "lint"]);
    expect(TEACHER_PROPOSAL_SHADOW_ONLY_CLASSES_V1).toEqual([
      "mutation",
      "forgetting",
      "correction",
    ]);
  });

  it("keeps canary rollout plans target-state only and off by default for proposal classes and candidate packs", () => {
    const proposalClasses: ProposalClass[] = [
      "compiler",
      "lint",
      "mutation",
      "forgetting",
      "correction",
    ];

    expect(Object.keys(TEACHER_CANARY_ROLLOUT_PLANS_V1).sort()).toEqual(
      [...proposalClasses].sort(),
    );

    for (const proposalClass of proposalClasses) {
      const plan: TeacherCanaryRolloutPlanV1 = describeTeacherCanaryRolloutPlanV1(proposalClass);

      expect(plan.proposalClass).toBe(proposalClass);
      expect(plan.surfaceState).toBe("target");
      expect(plan.rolloutMode).toBe("off");
      expect(plan.enabled).toBe(false);
      expect(plan.candidatePackVersion).toBeUndefined();
      expect(plan.shippedStateSummary).toContain("promoted packs");
      expect(plan.targetStateSummary).toContain("off by default");
      expect(plan.guardrails).toEqual(
        expect.arrayContaining([
          "Keep the rollout plan target-state only until it is explicitly shipped.",
          "Default rolloutMode stays off.",
          "Do not use the canary plan to change live serving without separate replay and rollback proof.",
          "Bind any candidate pack by durable version or id, never by ad hoc display labels.",
        ]),
      );

      const candidatePackScopedPlan = describeTeacherCanaryRolloutPlanV1(proposalClass, 9, "pack_09");
      expect(candidatePackScopedPlan).toMatchObject({
        proposalClass,
        surfaceState: "target",
        rolloutMode: "off",
        enabled: false,
        candidatePackVersion: 9,
        candidatePackId: "pack_09",
      });
    }
  });

  it("exposes class-specific replay gate modes and the four shared dimensions", () => {
    const proposalClasses: ProposalClass[] = [
      "compiler",
      "lint",
      "mutation",
      "forgetting",
      "correction",
    ];
    const promotableClasses = new Set(TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1);

    expect(Object.keys(TEACHER_PROPOSAL_REPLAY_GATES_V1).sort()).toEqual(
      [...proposalClasses].sort(),
    );

    for (const proposalClass of proposalClasses) {
      const gate: TeacherProposalReplayGateV1 = describeTeacherProposalReplayGate(proposalClass);
      const promotable = isTeacherProposalPromotableClassV1(proposalClass);
      const isPromotableClass = promotableClasses.has(
        proposalClass as (typeof TEACHER_PROPOSAL_PROMOTABLE_CLASSES_V1)[number],
      );

      expect(gate.proposalClass).toBe(proposalClass);
      expect(gate.reviewMode).toBe(isPromotableClass ? "promotable" : "shadow_only");
      expect(Object.keys(gate.dimensions).sort()).toEqual([
        "attributionFloor",
        "boundedness",
        "reversibility",
        "truthInvariants",
      ]);
      expect(gate.dimensions.truthInvariants.requirements).toEqual(
        expect.arrayContaining([
          "Explicit correction memory still outranks teacher synthesis.",
          "The live path stays read-only to the proposal.",
          "Evidence refs stay attached to any non-trivial claim.",
        ]),
      );
      expect(gate.dimensions.attributionFloor.requirements).toEqual(
        expect.arrayContaining([
          "Every proposal carries durable evidence refs.",
          "Source ids must be stable record ids, not display labels.",
          "Unattributed payload stays out of promotion.",
        ]),
      );
      expect(gate.dimensions.boundedness.requirements).toEqual(
        expect.arrayContaining([
          "Proposal subject sets stay finite and small.",
          "Payloads avoid raw corpus dumps and unbounded excerpts.",
          "Replay fits inside a single review pass.",
        ]),
      );
      expect(gate.dimensions.reversibility.requirements).toEqual(
        expect.arrayContaining([
          "RollbackKey identifies the reversible path.",
          "Prior state remains recoverable for replay.",
          "Rejected or superseded proposals keep lineage.",
        ]),
      );
      expect(promotable).toBe(isPromotableClass);
    }
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

  it("summarizes the first live-proof rung as a bounded publication-safe view", () => {
    const summary = summarizeTeacherV3LiveProofRungV1(liveProofRung);

    expect(summary.rungId).toBe("live-proof-rung-1");
    expect(summary.before).toMatchObject({
      count: 2,
      shippedCount: 2,
      targetCount: 0,
      ids: ["runtime-status-before", "operator-proof-before"],
    });
    expect(summary.after).toMatchObject({
      count: 2,
      shippedCount: 1,
      targetCount: 1,
      ids: ["runtime-status-after", "teacher-v3-target-after"],
    });
    expect(summary.checks.token.status).toBe("pass");
    expect(summary.checks.latency.summary).toContain("bounded proof datum");
    expect(summary.checks.truth.evidenceSurfaceIds).toContain("teacher-v3-target-after");
    expect(summary.publicationSafeArtifacts).toMatchObject({
      count: 2,
      ids: ["teacher-v3-live-proof-summary", "teacher-v3-live-proof-surface-map"],
      kinds: ["summary", "surface-map"],
    });
    expect(summary.publicationSafeArtifacts.redactions).toEqual([
      "raw stdout/stderr",
      "secret-bearing values",
      "raw source payloads",
    ]);
  });

  it("fails closed when a live-proof rung is missing a check or contains raw logs", () => {
    expect(() =>
      summarizeTeacherV3LiveProofRungV1({
        ...liveProofRung,
        checks: liveProofRung.checks.filter((check) => check.kind !== "truth"),
      }),
    ).toThrow(/missing teacher v3 live-proof truth check/);

    expect(() =>
      summarizeTeacherV3LiveProofRungV1({
        ...liveProofRung,
        publicationSafeArtifacts: [
          ...liveProofRung.publicationSafeArtifacts,
          unsafePublicationArtifact as never,
        ],
      }),
    ).toThrow(/publication-safe artifact must not contain raw logs/);
  });
});
