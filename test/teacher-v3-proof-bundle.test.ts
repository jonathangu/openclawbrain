import { mkdirSync, readFileSync, rmSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildTeacherV3ProofBundle,
  TEACHER_V3_PROOF_BUNDLE_LAYOUT,
  writeTeacherV3ProofBundle,
} from "../scripts/teacher-v3-proof-bundle.mjs";

const runtimeStatus = {
  serveState: "serving_active_pack",
  currentPackVersion: 7,
  currentPackPromotedAt: "2026-04-03T18:00:00Z",
  currentPackMetadata: {
    reason: "current promoted pack",
    kind: "promoted_pack",
  },
  teacherConfigured: true,
  teacherProvider: "openai",
  teacherModel: "gpt-4.1",
  operatorHealth: {
    status: "healthy",
    detail: "runtime and proof surface are aligned",
  },
  learningHealth: {
    status: "healthy",
    detail: "learning loop is steady",
  },
  routeTraceCount: 12,
  supervisionCount: 4,
  recentTraceCount: 2,
  pendingLabels: 1,
  pendingObservations: 3,
  lastCompileReportSummary: "compile report is bounded",
  lastAssemblyDecision: {
    summary: "assembly summary",
    verdict: "approved",
  },
  lastPrefetchDecision: {
    summary: "prefetch summary",
    verdict: "ready",
  },
  lastPromotionReason: "current pack promoted",
  lastPromotionVerdict: {
    verdict: "promoted",
    summary: "promotion succeeded",
  },
  lastReplayFailureReason: null,
  lastReplayGateVerdict: {
    verdict: "pass",
    summary: "replay gate passed",
  },
};

const operatorProof = {
  bundleDir: "artifacts/operator-proof-20260403-182600Z",
  command: "openclawbrain proof --openclaw-home ~/.openclaw",
  summary: "operator proof summary",
  verdict: {
    verdict: "success_and_proven",
    severity: "info",
    why: "runtime truth and proof truth were aligned",
    missingProofs: [],
  },
  runtimeLoadProofPath: "~/.openclaw/activation/attachment-truth/runtime-load-proofs.json",
  runtimeLoadProofExists: true,
  stepCount: 5,
  postBundleCount: 2,
};

const docsTruth = {
  path: "docs/architecture/teacher-v3-proof.md",
  title: "Teacher v3 proposal reporting / proof surfaces",
  summary: "Design-only mapping of shipped truth to target-state proof surfaces.",
};

describe("teacher v3 proof bundle writer", () => {
  it("emits the bounded five-file bundle from real runtime/proof inputs", () => {
    const outputDir = path.join(process.cwd(), "scratch", "teacher-v3-proof-bundle-test");
    rmSync(outputDir, { recursive: true, force: true });
    mkdirSync(path.dirname(outputDir), { recursive: true });

    const bundle = buildTeacherV3ProofBundle({
      bundleStartedAt: "2026-04-03T18:26:00Z",
      outputDir,
      runtimeStatusCommand: "npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed",
      runtimeStatus,
      operatorProofCommand: "openclawbrain proof --openclaw-home ~/.openclaw",
      operatorProof,
      docsTruth,
      producerVersion: "openclawbrain@0.3.8",
      proposalClass: "compiler",
      proposalLane: "compiler",
      proposalStatus: "promotable",
      proposalRecord: {
        recordSource: "stored-proposal",
        evidence: [
          {
            evidenceId: "evi_replay_01",
            sourceKind: "file",
            sourceId: "docs/architecture/teacher-v3-replay.md",
            authority: "raw_source",
            derivation: "teacher_compilation",
            excerpt: "Replay over real candidate state for compiler proposals.",
            sourceHash: "sha256:replay-01",
          },
        ],
        counterevidence: [],
        replaySuites: ["teacher-v3/compiler-replay"],
        confidence: 0.94,
        rollbackKey: "rollback:teacher-v3:compiler:replay",
        replaySummary: {
          replayId: "treplay_01",
          proposalId: "prop_01",
          proposalClass: "compiler",
          status: "promotable",
          reviewMode: "promotable",
          basePackVersion: 7,
          baseGraphHash: "graph_sha_07",
          candidatePackVersion: 8,
          candidatePackId: "candidate_pack_08",
          candidateGraphHash: "graph_sha_08",
          beforeScore: 0.61,
          afterScore: 0.79,
          scoreDelta: 0.18,
          before: {
            phase: "before",
            surfaceState: "shipped",
            packVersion: 7,
            packId: null,
            graphHash: "graph_sha_07",
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
            graphHash: "graph_sha_08",
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
            replaySuites: ["teacher-v3/compiler-replay"],
            candidatePackVersion: 8,
            candidatePackId: "candidate_pack_08",
            candidateGraphHash: "graph_sha_08",
            summary: "Compiler replay is promotable on candidate pack candidate_pack_08; evidence-backed lineage stays intact and the candidate graph is distinct from base state.",
            notes: [
              "basePackVersion=7",
              "baseGraphHash=graph_sha_07",
              "beforeScore=0.610",
              "afterScore=0.790",
              "scoreDelta=0.180",
              "compiler proposals may be evaluated as promotable, but they are not auto-promoted",
            ],
          },
          summary: "compiler replay accepted on candidate_pack_08; before=0.610 after=0.790 delta=0.180",
          createdAt: "2026-04-03T18:33:00Z",
          updatedAt: "2026-04-03T18:33:00Z",
        },
      },
    });

    expect(Object.keys(bundle.files).sort()).toEqual([
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.status,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict,
    ].sort());
    expect(bundle.surfaceMap.counts).toMatchObject({
      observedSurfaceCount: 3,
      shippedSurfaceCount: 3,
      targetSurfaceCount: 5,
      totalSurfaceCount: 8,
    });
    expect(bundle.statusReport.recommendations).toHaveLength(3);
    expect(bundle.proposalReport.proposal).toMatchObject({
      proposalClass: "compiler",
      proposalLane: "compiler",
      status: "promotable",
      reviewMode: "promotable",
      recordSource: "stored-proposal",
    });
    expect(bundle.proposalReport.replaySummary).toMatchObject({
      replayId: "treplay_01",
      proposalClass: "compiler",
      status: "promotable",
      candidatePackId: "candidate_pack_08",
    });
    expect(bundle.proposalReport.gate1Seam).toMatchObject({
      present: false,
      recordSource: "runtime-capture",
    });
    expect(bundle.proposalReport.replayGate).toMatchObject({
      proposalClass: "compiler",
      reviewMode: "promotable",
    });
    expect(Object.keys(bundle.proposalReport.replayGate.dimensions).sort()).toEqual([
      "attributionFloor",
      "boundedness",
      "reversibility",
      "truthInvariants",
    ]);
    expect(bundle.proposalReport.publicationSafeArtifacts).toHaveLength(5);
    expect(bundle.verdictReport).toMatchObject({
      verdict: "reviewable",
      severity: "info",
      targetStateOnly: true,
    });
    expect(bundle.summaryMarkdown).toContain("Teacher v3 proof bundle");
    expect(bundle.summaryMarkdown).toContain("Gate 1 seam");
    expect(bundle.summaryMarkdown).toContain("replay status: **promotable**");
    expect(bundle.summaryMarkdown).toContain("candidate_pack_08");

    const writeResult = writeTeacherV3ProofBundle(outputDir, bundle);
    expect(writeResult.writtenFiles).toHaveLength(5);

    const summaryPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary);
    const statusPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.status);
    const surfaceMapPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap);
    const proposalReportPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport);
    const verdictPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict);

    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.status])).not.toThrow();
    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap])).not.toThrow();
    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport])).not.toThrow();
    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict])).not.toThrow();

    expect(readFileSync(summaryPath, "utf8")).toContain("runtime truth");
    expect(readFileSync(statusPath, "utf8")).toContain("teacher_v3_proof_bundle_status.v1");
    expect(readFileSync(surfaceMapPath, "utf8")).toContain("runtime-truth");
    expect(readFileSync(proposalReportPath, "utf8")).toContain("teacher_v3_proposal_report.v1");
    expect(readFileSync(verdictPath, "utf8")).toContain("reviewable");
  });

  it("captures explicit replay outcomes for a shadow-only proposal record seam", () => {
    const outputDir = path.join(process.cwd(), "scratch", "teacher-v3-proof-bundle-shadow-test");
    rmSync(outputDir, { recursive: true, force: true });
    mkdirSync(path.dirname(outputDir), { recursive: true });

    const bundle = buildTeacherV3ProofBundle({
      bundleStartedAt: "2026-04-03T19:05:00Z",
      outputDir,
      bundleId: "teacher-v3-proof-shadow-01",
      proposalId: "prop_shadow_01",
      proposalClass: "mutation",
      proposalLane: "mutation",
      proposalStatus: "shadow_scored",
      subjectIds: ["memory:shadow-01"],
      proposalRecord: {
        recordSource: "brain_teacher_proposals",
        proposalClass: "mutation",
        proposalLane: "mutation",
        status: "shadow_scored",
        confidence: 0.81,
        rollbackKey: "rollback:teacher-v3:mutation:shadow",
        createdAt: "2026-04-03T19:00:00Z",
        resolvedAt: "2026-04-03T19:01:00Z",
        updatedAt: "2026-04-03T19:01:30Z",
        lineage: {
          proposalClass: "mutation",
          basePackVersion: 7,
          baseGraphHash: "graph_sha_shadow_01",
          producerVersion: "teacher-v3@0.1.0",
          scope: "mutation-shadow",
          idempotencyKey: "teacher-v3::mutation::mutation-shadow",
        },
        replaySuites: ["mutation-shadow-smoke", "mutation-rollback-smoke"],
        replayOutcomes: [
          {
            outcomeId: "shadow_replay_01",
            replaySuite: "mutation-shadow-smoke",
            proposalClass: "mutation",
            reviewMode: "shadow_only",
            result: "pass",
            source: "proposal_record",
            summary: "mutation shadow replay stayed bounded",
            capturedAt: "2026-04-03T19:00:30Z",
          },
          {
            outcomeId: "shadow_replay_02",
            replaySuite: "mutation-rollback-smoke",
            proposalClass: "mutation",
            reviewMode: "shadow_only",
            result: "warn",
            source: "proposal_record",
            summary: "rollback path preserved inspectable lineage",
            capturedAt: "2026-04-03T19:00:45Z",
          },
        ],
        gate1Seam: {
          present: true,
          recordSource: "brain_teacher_proposals",
          note: "loaded from the persisted proposal store seam",
        },
      },
      runtimeStatusCommand: "npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed",
      runtimeStatus,
      operatorProofCommand: "openclawbrain proof --openclaw-home ~/.openclaw",
      operatorProof,
      docsTruth,
      producerVersion: "openclawbrain@0.3.8",
    });

    expect(bundle.proposalReport.proposal).toMatchObject({
      proposalClass: "mutation",
      proposalLane: "mutation",
      status: "shadow_scored",
      reviewMode: "shadow_only",
      recordSource: "brain_teacher_proposals",
      replayOutcomeCount: 2,
    });
    expect(bundle.proposalReport.gate1Seam).toMatchObject({
      present: true,
      recordSource: "brain_teacher_proposals",
    });
    expect(bundle.proposalReport.replayOutcomes).toHaveLength(2);
    expect(bundle.proposalReport.replayOutcomeSummary).toMatchObject({
      replayOutcomeCount: 2,
      replaySuites: ["mutation-shadow-smoke", "mutation-rollback-smoke"],
      resultCounts: { pass: 1, warn: 1, fail: 0 },
      reviewModeCounts: { promotable: 0, shadow_only: 2 },
      sourceCounts: { proposal_record: 2, proof_bundle: 0, derived: 0 },
    });
    expect(bundle.statusReport.replayOutcomeSummary).toMatchObject({
      replayOutcomeCount: 2,
      reviewModeCounts: { promotable: 0, shadow_only: 2 },
    });
    expect(bundle.verdictReport).toMatchObject({
      verdict: "reviewable",
      targetStateOnly: false,
    });
    expect(bundle.summaryMarkdown).toContain("Replay outcomes");
    expect(bundle.summaryMarkdown).toContain("shadow_only=2");

    const writeResult = writeTeacherV3ProofBundle(outputDir, bundle);
    expect(writeResult.writtenFiles).toHaveLength(5);
    expect(readFileSync(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport), "utf8")).toContain("mutation-shadow-smoke");
  });
});
