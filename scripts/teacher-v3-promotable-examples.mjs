#!/usr/bin/env node

import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { BrainGraph } from "../src/brain-core/graph.js";
import { runBrainMigrations } from "../src/brain-store/migrations.js";
import { BrainStore } from "../src/brain-store/store.js";
import {
  describeTeacherCanaryRolloutPlanV1,
  describeTeacherProposalReplayGate,
  summarizeTeacherProposalV1,
} from "../src/brain-core/teacher-v3-contracts.js";
import { buildTeacherProposalReplaySummaryV1 } from "../src/brain-core/teacher-v3-replay.js";
import {
  buildTeacherV3ProofBundle,
  buildTeacherV3ProofBundleDigest,
  writeTeacherV3ProofBundle,
} from "./teacher-v3-proof-bundle.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const outputRoot = path.join(repoRoot, "artifacts", "teacher-v3-promotable-examples");

const GENERATED_AT = "2026-04-03T22:17:00Z";
const FIXED_GRAPH_TIME = Date.parse("2026-04-03T18:30:00Z");
const DEFAULT_RUNTIME_STATUS = {
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

const DEFAULT_OPERATOR_PROOF = {
  bundleDir: "artifacts/operator-proof-20260403-221700Z",
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

const DEFAULT_DOCS_TRUTH = {
  path: "docs/architecture/teacher-v3-proof.md",
  title: "Teacher v3 proposal reporting / proof surfaces",
  summary: "Design-only mapping of shipped truth to target-state proof surfaces.",
};

function sha256Text(text) {
  return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}

function ensureDir(dirPath) {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function makeNode(id, kind = "chunk", timestamp = FIXED_GRAPH_TIME) {
  return {
    id,
    kind,
    content: id,
    embedding: null,
    sourceUri: null,
    trust: "scanner",
    tags: [id],
    tokenCount: id.length,
    metadata: {},
    createdAt: timestamp,
    updatedAt: timestamp,
  };
}

function makeGraph(prefix) {
  const graph = new BrainGraph();
  graph.addNode(makeNode(`${prefix}_a`));
  graph.addNode(makeNode(`${prefix}_b`));
  graph.addNode(makeNode(`${prefix}_c`));
  graph.addEdge({
    source: `${prefix}_a`,
    target: `${prefix}_b`,
    kind: "learned",
    weight: 0.8,
    prior: 0.5,
    metadata: { seed: true },
    decayedAt: FIXED_GRAPH_TIME,
    createdAt: FIXED_GRAPH_TIME,
  });
  return graph;
}

function makeCandidatePack(version, prefix) {
  return {
    version,
    nodeCount: 3,
    edgeCount: 1,
    healthJson: JSON.stringify({
      nodeCount: 3,
      edgeCount: 1,
      firedPerQuery: prefix === "compiler" ? 1.8 : 1.7,
      dormantPercent: prefix === "compiler" ? 0.1 : 0.12,
      orphanCount: 0,
    }),
    promotedAt: null,
    rolledBack: false,
    createdAt: FIXED_GRAPH_TIME,
  };
}

function makeEvidence({ evidenceId, sourceKind, sourceId, authority, derivation, excerpt }) {
  return {
    evidenceId,
    sourceKind,
    sourceId,
    authority,
    derivation,
    excerpt,
    sourceHash: sha256Text(`${sourceId}\n${excerpt}`),
  };
}

function makeStore() {
  const db = new DatabaseSync(":memory:");
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);
  return new BrainStore(db);
}

function makeReplaySummary(proposal, candidateState, evaluatedAt) {
  return buildTeacherProposalReplaySummaryV1({
    proposal,
    candidateState,
    evaluatedAt,
  });
}

function makeCompilerProposal() {
  return {
    proposalId: "prop_teacher_v3_compiler_worked_example",
    proposalClass: "compiler",
    lane: "compiler",
    status: "promotable",
    lineage: {
      proposalClass: "compiler",
      basePackVersion: 7,
      baseGraphHash: "graph_sha_compiler_worked_example",
      producerVersion: "teacher-v3-worked-examples@0.1.0",
      producerBuildId: "build_compiler_worked_example",
      promptHash: "prompt_sha_compiler_worked_example",
      templateId: "teacher-v3/compiler-v1",
      scope: "compiler-persistence",
      profile: "worked-example",
      idempotencyKey: "teacher-v3::compiler::worked-example",
      sourceBundleId: "bundle_compiler_worked_example",
      parentProposalIds: ["prop_seed_compiler_00"],
    },
    subjectIds: [
      "proposal:compiler-persistence",
      "store:round-trip",
      "proof-bundle:five-file-layout",
    ],
    evidence: [
      makeEvidence({
        evidenceId: "evi_compiler_store_01",
        sourceKind: "file",
        sourceId: "src/brain-store/store.ts#updateTeacherProposalStatus",
        authority: "raw_source",
        derivation: "teacher_compilation",
        excerpt: "updateTeacherProposalStatus({ proposalId, status, proofBundle, replaySummary, canaryRollout })",
      }),
      makeEvidence({
        evidenceId: "evi_compiler_replay_01",
        sourceKind: "file",
        sourceId: "src/brain-core/teacher-v3-replay.ts#buildTeacherProposalReplaySummaryV1",
        authority: "raw_source",
        derivation: "teacher_compilation",
        excerpt: "Compiler replay is promotable on candidate pack ...",
      }),
      makeEvidence({
        evidenceId: "evi_compiler_proof_01",
        sourceKind: "file",
        sourceId: "scripts/teacher-v3-proof-bundle.mjs#buildTeacherV3ProofBundle",
        authority: "raw_source",
        derivation: "teacher_compilation",
        excerpt: "summary.md, status.json, surface-map.json, proposal-report.json, verdict.json",
      }),
    ],
    counterevidence: [
      makeEvidence({
        evidenceId: "cevi_compiler_target_01",
        sourceKind: "file",
        sourceId: "docs/architecture/teacher-v3-proof.md#target-state",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "The target-state proof bundle is an overlay on top of the first three surfaces, not a replacement for them.",
      }),
    ],
    payload: {
      kind: "compiler-persistence",
      summary: "Persist compiler proposals with stable lineage, replay summaries, and bounded proof bundles.",
      target: "teacher-v3",
    },
    expectedEffect: {
      retrieval: "better",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: 0.96,
    replaySuites: ["compiler-persistence-smoke", "compiler-proof-bundle-smoke"],
    rollbackKey: "rollback:teacher-v3:compiler:worked-example",
    canaryRollout: describeTeacherCanaryRolloutPlanV1({
      proposalClass: "compiler",
      rollbackKey: "rollback:teacher-v3:compiler:worked-example",
      candidatePackVersion: 8,
      candidatePackId: "candidate_pack_compiler_08",
    }),
    replayGate: describeTeacherProposalReplayGate("compiler"),
    createdAt: "2026-04-03T18:26:00Z",
  };
}

function makeLintProposal() {
  return {
    proposalId: "prop_teacher_v3_lint_worked_example",
    proposalClass: "lint",
    lane: "lint",
    status: "promotable",
    lineage: {
      proposalClass: "lint",
      basePackVersion: 7,
      baseGraphHash: "graph_sha_lint_worked_example",
      producerVersion: "teacher-v3-worked-examples@0.1.0",
      producerBuildId: "build_lint_worked_example",
      promptHash: "prompt_sha_lint_worked_example",
      templateId: "teacher-v3/lint-v1",
      scope: "release-truth-sweep",
      profile: "worked-example",
      idempotencyKey: "teacher-v3::lint::worked-example",
      sourceBundleId: "bundle_lint_worked_example",
      parentProposalIds: ["prop_seed_lint_00"],
    },
    subjectIds: [
      "release:0.4.28",
      "docs:public-surface-truth",
      "docs:proof-surface-boundary",
    ],
    evidence: [
      makeEvidence({
        evidenceId: "evi_lint_release_01",
        sourceKind: "file",
        sourceId: "scripts/verify-release-docs-drift.mjs#verifyReleaseDocsDrift",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "This deterministic lint compares the current release version in CHANGELOG.md against public release-surfaces.",
      }),
      makeEvidence({
        evidenceId: "evi_lint_readme_01",
        sourceKind: "file",
        sourceId: "README.md#current-version",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "Current version: **0.4.28**",
      }),
      makeEvidence({
        evidenceId: "evi_lint_docs_01",
        sourceKind: "file",
        sourceId: "docs/README.md#current-release-notes",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "Current release notes (0.4.28)",
      }),
      makeEvidence({
        evidenceId: "evi_lint_endstate_01",
        sourceKind: "file",
        sourceId: "docs/END_STATE.md#split-package-story",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "split packages `@openclawbrain/openclaw@0.4.28` and `@openclawbrain/cli@0.4.28` are published",
      }),
    ],
    counterevidence: [
      makeEvidence({
        evidenceId: "cevi_lint_target_01",
        sourceKind: "file",
        sourceId: "docs/architecture/teacher-v3-proof.md#target-state-surfaces",
        authority: "raw_source",
        derivation: "teacher_lint",
        excerpt: "Teacher v3 reporting may summarize and cross-reference truth, but it must not become a new source of truth for the live runtime.",
      }),
    ],
    payload: {
      kind: "release-docs-truth-sweep",
      summary: "Keep the public release story aligned across README, docs index, changelog, and end-state docs.",
      currentVersion: "0.4.28",
      surfacedFiles: ["README.md", "docs/README.md", "CHANGELOG.md", "docs/END_STATE.md"],
    },
    expectedEffect: {
      retrieval: "same",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: 0.91,
    replaySuites: ["release-docs-drift-smoke", "teacher-v3-lint-proof-surface-smoke"],
    rollbackKey: "rollback:teacher-v3:lint:worked-example",
    canaryRollout: describeTeacherCanaryRolloutPlanV1({
      proposalClass: "lint",
      rollbackKey: "rollback:teacher-v3:lint:worked-example",
      candidatePackVersion: 8,
      candidatePackId: "candidate_pack_lint_08",
    }),
    replayGate: describeTeacherProposalReplayGate("lint"),
    createdAt: "2026-04-03T18:27:00Z",
  };
}

function buildCompilerExample(store, root = outputRoot) {
  const proposal = makeCompilerProposal();
  store.insertTeacherProposal(proposal);

  const candidateGraph = makeGraph("compiler");
  const replaySummary = makeReplaySummary(
    proposal,
    {
      candidatePack: makeCandidatePack(8, "compiler"),
      candidatePackId: "candidate_pack_compiler_08",
      candidateGraph,
    },
    "2026-04-03T18:33:00Z",
  );

  store.updateTeacherProposalStatus({
    proposalId: proposal.proposalId,
    status: "promoted",
    resolvedAt: "2026-04-03T18:34:00Z",
    replaySummary,
    canaryRollout: proposal.canaryRollout,
  });

  const loadedProposal = store.getTeacherProposal(proposal.proposalId);
  if (!loadedProposal) {
    throw new Error(`compiler proposal missing after insert: ${proposal.proposalId}`);
  }

  const exampleDir = path.join(root, "compiler");
  const proofBundleDir = path.join(exampleDir, "proof-bundle");
  const proofBundle = buildTeacherV3ProofBundle({
    bundleId: "teacher-v3-compiler-worked-example",
    bundleStartedAt: "2026-04-03T18:33:00Z",
    outputDir: proofBundleDir,
    runtimeStatusCommand: "openclawbrain status --detailed",
    runtimeStatus: DEFAULT_RUNTIME_STATUS,
    operatorProofCommand: DEFAULT_OPERATOR_PROOF.command,
    operatorProof: DEFAULT_OPERATOR_PROOF,
    docsTruth: DEFAULT_DOCS_TRUTH,
    producerVersion: "teacher-v3-worked-examples@0.1.0",
    proposalClass: "compiler",
    proposalLane: "compiler",
    proposalStatus: loadedProposal.status,
    proposalRecord: {
      ...loadedProposal,
      recordSource: "brain_store",
      proofBundle: {
        bundleId: "teacher-v3-compiler-worked-example",
        rollbackKey: loadedProposal.rollbackKey,
      },
      gate1Seam: {
        present: true,
        recordSource: "brain_store",
        note: "Compiler proposal round-tripped through BrainStore before proof-bundle emission.",
      },
      replayOutcomes: [
        {
          outcomeId: "compiler-persistence-smoke:1",
          replaySuite: "compiler-persistence-smoke",
          proposalClass: "compiler",
          reviewMode: "promotable",
          result: "pass",
          source: "proposal_record",
          summary: "Stored compiler proposal round-tripped through BrainStore without losing lineage or rollback binding.",
          capturedAt: "2026-04-03T18:33:00Z",
        },
        {
          outcomeId: "compiler-proof-bundle-smoke:1",
          replaySuite: "compiler-proof-bundle-smoke",
          proposalClass: "compiler",
          reviewMode: "promotable",
          result: "warn",
          source: "proposal_record",
          summary: "Bounded five-file proof bundle stayed publication-safe and target-state aware.",
          capturedAt: "2026-04-03T18:33:00Z",
        },
      ],
      recommendations: [
        "Keep the stored proposal seam coupled to replay summaries, not live mutation.",
        "Keep canary rollout off by default until a later tranche explicitly opts it in.",
        "Preserve the bounded five-file proof bundle layout and publication-safe redactions.",
      ],
    },
  });
  writeTeacherV3ProofBundle(proofBundleDir, proofBundle);

  const storedProposalSummary = summarizeTeacherProposalV1(loadedProposal);
  const proofBundleDigest = buildTeacherV3ProofBundleDigest(proofBundle);
  const example = buildExampleRecord({
    lane: "compiler",
    title: "Compiler worked example",
    store,
    proposal: loadedProposal,
    proposalSummary: storedProposalSummary,
    replaySummary,
    proofBundle,
    proofBundleDigest,
    proofBundleDir,
    proofBundleRelativeDir: "compiler/proof-bundle",
    proofBundleLabel: "compiler",
    targetStateNotes: [
      "The canary plan is explicit, off by default, and rollback-bound.",
      "The proof bundle is a review surface, not a live runtime truth source.",
      "Live serving still only uses promoted packs.",
    ],
  });

  return {
    lane: "compiler",
    proposal: loadedProposal,
    proposalSummary: storedProposalSummary,
    replaySummary,
    proofBundle,
    proofBundleDigest,
    example,
    exampleDir,
    proofBundleDir,
  };
}

function buildLintExample(store, root = outputRoot) {
  const proposal = makeLintProposal();
  store.insertTeacherProposal(proposal);

  const candidateGraph = makeGraph("lint");
  const replaySummary = makeReplaySummary(
    proposal,
    {
      candidatePack: makeCandidatePack(8, "lint"),
      candidatePackId: "candidate_pack_lint_08",
      candidateGraph,
    },
    "2026-04-03T18:34:00Z",
  );

  store.updateTeacherProposalStatus({
    proposalId: proposal.proposalId,
    status: "promotable",
    replaySummary,
    canaryRollout: proposal.canaryRollout,
  });

  const loadedProposal = store.getTeacherProposal(proposal.proposalId);
  if (!loadedProposal) {
    throw new Error(`lint proposal missing after insert: ${proposal.proposalId}`);
  }

  const exampleDir = path.join(root, "lint");
  const proofBundleDir = path.join(exampleDir, "proof-bundle");
  const proofBundle = buildTeacherV3ProofBundle({
    bundleId: "teacher-v3-lint-worked-example",
    bundleStartedAt: "2026-04-03T18:34:00Z",
    outputDir: proofBundleDir,
    runtimeStatusCommand: "openclawbrain status --detailed",
    runtimeStatus: DEFAULT_RUNTIME_STATUS,
    operatorProofCommand: DEFAULT_OPERATOR_PROOF.command,
    operatorProof: DEFAULT_OPERATOR_PROOF,
    docsTruth: DEFAULT_DOCS_TRUTH,
    producerVersion: "teacher-v3-worked-examples@0.1.0",
    proposalClass: "lint",
    proposalLane: "lint",
    proposalStatus: loadedProposal.status,
    proposalRecord: {
      ...loadedProposal,
      recordSource: "brain_store",
      proofBundle: {
        bundleId: "teacher-v3-lint-worked-example",
        rollbackKey: loadedProposal.rollbackKey,
      },
      gate1Seam: {
        present: true,
        recordSource: "brain_store",
        note: "Lint proposal round-tripped through BrainStore before proof-bundle emission.",
      },
      replayOutcomes: [
        {
          outcomeId: "release-docs-drift-smoke:1",
          replaySuite: "release-docs-drift-smoke",
          proposalClass: "lint",
          reviewMode: "promotable",
          result: "pass",
          source: "proposal_record",
          summary: "Release-docs drift lint remained bounded against the current 0.4.28 public story.",
          capturedAt: "2026-04-03T18:34:00Z",
        },
        {
          outcomeId: "teacher-v3-lint-proof-surface-smoke:1",
          replaySuite: "teacher-v3-lint-proof-surface-smoke",
          proposalClass: "lint",
          reviewMode: "promotable",
          result: "warn",
          source: "proposal_record",
          summary: "Teacher v3 proof/reporting surfaces stayed target-state only and reviewable.",
          capturedAt: "2026-04-03T18:34:00Z",
        },
      ],
      recommendations: [
        "Keep the release-docs drift lint deterministic and CI-first.",
        "Keep Teacher v3 proof/reporting surfaces target-state only until the runtime proof adoption tranche lands.",
        "Preserve the explicit rollback key even for review-only lint proposals.",
      ],
    },
  });
  writeTeacherV3ProofBundle(proofBundleDir, proofBundle);

  const storedProposalSummary = summarizeTeacherProposalV1(loadedProposal);
  const proofBundleDigest = buildTeacherV3ProofBundleDigest(proofBundle);
  const example = buildExampleRecord({
    lane: "lint",
    title: "Lint worked example",
    store,
    proposal: loadedProposal,
    proposalSummary: storedProposalSummary,
    replaySummary,
    proofBundle,
    proofBundleDigest,
    proofBundleDir,
    proofBundleRelativeDir: "lint/proof-bundle",
    proofBundleLabel: "lint",
    targetStateNotes: [
      "The release-docs lint is promotable, but it still only audits public surfaces; it does not mutate them.",
      "Teacher v3 proof/reporting surfaces remain target-state overlays, not the live runtime truth source.",
      "The rollback key is explicit even though the proposal stayed reviewable/promotable.",
    ],
  });

  return {
    lane: "lint",
    proposal: loadedProposal,
    proposalSummary: storedProposalSummary,
    replaySummary,
    proofBundle,
    proofBundleDigest,
    example,
    exampleDir,
    proofBundleDir,
  };
}

function buildExampleRecord({
  lane,
  title,
  proposal,
  proposalSummary,
  replaySummary,
  proofBundle,
  proofBundleDigest,
  proofBundleDir,
  proofBundleRelativeDir,
  proofBundleLabel,
  targetStateNotes,
}) {
  const proofBundleStatus = {
    verdict: proofBundle.verdictReport.verdict,
    severity: proofBundle.verdictReport.severity,
    why: proofBundle.verdictReport.why,
    gate1Seam: proofBundle.proposalReport.gate1Seam,
    replayOutcomeSummary: proofBundle.statusReport.replayOutcomeSummary,
    canaryRollout: proofBundle.statusReport.canaryRollout,
    canaryActivationGuard: proofBundle.statusReport.canaryActivationGuard,
    publicationSafeArtifacts: proofBundle.statusReport.publicationSafeArtifacts,
  };

  const example = {
    contract: "teacher_v3_promotable_example.v1",
    lane,
    title,
    reviewMode: proposal.replayGate.reviewMode,
    proposal: proposalSummary,
    proposalRecord: {
      proposalId: proposal.proposalId,
      proposalClass: proposal.proposalClass,
      lane: proposal.lane,
      status: proposal.status,
      rollbackKey: proposal.rollbackKey,
      confidence: proposal.confidence,
      createdAt: proposal.createdAt,
      resolvedAt: proposal.resolvedAt ?? null,
      lineage: proposal.lineage,
      subjectIds: proposal.subjectIds,
      evidence: proposal.evidence,
      counterevidence: proposal.counterevidence ?? [],
      payload: proposal.payload,
      replaySuites: proposal.replaySuites,
      canaryRollout: proposal.canaryRollout,
      replayGate: proposal.replayGate,
      recordSource: "brain_store",
      gate1Seam: {
        present: true,
        recordSource: "brain_store",
        note: `${proofBundleLabel} proposal was persisted, reloaded, and replay-scored before the proof bundle was emitted.`,
      },
    },
    replaySummary,
    proofBundle: {
      bundleId: proofBundle.bundleId,
      paths: {
        summary: `${proofBundleRelativeDir}/summary.md`,
        status: `${proofBundleRelativeDir}/status.json`,
        surfaceMap: `${proofBundleRelativeDir}/surface-map.json`,
        proposalReport: `${proofBundleRelativeDir}/proposal-report.json`,
        verdict: `${proofBundleRelativeDir}/verdict.json`,
      },
      digest: proofBundleDigest,
      status: proofBundleStatus,
    },
    targetStateNotes,
  };

  return example;
}

function renderExampleMarkdown(example) {
  const evidenceLines = example.proposalRecord.evidence.map((item) => `- \`${item.evidenceId ?? item.sourceId}\` → \`${item.sourceId}\`: ${item.excerpt}`);
  const counterevidenceLines = example.proposalRecord.counterevidence.map((item) => `- \`${item.evidenceId ?? item.sourceId}\` → \`${item.sourceId}\`: ${item.excerpt}`);
  const notesLines = example.targetStateNotes.map((note) => `- ${note}`);
  const proof = example.proofBundle;
  const replay = example.replaySummary;

  return [
    `# ${example.title}`,
    "",
    `- proposal: \`${example.proposal.proposalId}\` (${example.proposal.proposalClass}, ${example.proposal.status})`,
    `- review mode: **${example.reviewMode}**`,
    `- rollback key: \`${example.proposal.rollbackKey}\``,
    `- replay: **${replay.classSummary.reviewMode}** / **${replay.status}**`,
    `- replay summary: ${replay.summary}`,
    `- proof verdict: **${proof.status.verdict}** (${proof.status.severity})`,
    `- proof bundle: \`${proof.bundleId}\``,
    `- proof bundle files: \`${proof.paths.summary}\`, \`${proof.paths.status}\`, \`${proof.paths.surfaceMap}\`, \`${proof.paths.proposalReport}\`, \`${proof.paths.verdict}\``,
    `- gate 1 seam: ${proof.status.gate1Seam.present ? "yes" : "no"}`,
    "",
    "## What was proposed",
    `- ${example.proposalRecord.payload.summary}`,
    `- subjects: ${example.proposal.subjectIds.join(", ")}`,
    "",
    "## Evidence",
    ...evidenceLines,
    "",
    "## Counterevidence / boundary",
    ...counterevidenceLines,
    "",
    "## Replay summary",
    `- before score: ${replay.beforeScore.toFixed(3)}`,
    `- after score: ${replay.afterScore.toFixed(3)}`,
    `- score delta: ${replay.scoreDelta.toFixed(3)}`,
    `- class summary: ${replay.classSummary.summary}`,
    `- replay suites: ${replay.classSummary.replaySuites.join(", ")}`,
    "",
    "## Proof bundle + verdict surface",
    `- verdict: **${proof.status.verdict}** (${proof.status.severity})`,
    `- why: ${proof.status.why}`,
    `- review mode: ${example.reviewMode}`,
    `- publication-safe artifacts: ${proof.status.publicationSafeArtifacts.map((item) => item.artifactId).join(", ")}`,
    `- rollback-bound canary: ${proof.status.canaryRollout.rollbackBound ? "yes" : "no"}`,
    `- rollout mode: ${proof.status.canaryRollout.rolloutMode}`,
    `- enabled: ${proof.status.canaryRollout.enabled ? "yes" : "no"}`,
    `- activation guard: ${proof.status.canaryActivationGuard.summary}`,
    "",
    "## What remains target-state",
    ...notesLines,
  ].join("\n") + "\n";
}

function writeExampleArtifacts(example, outputDir) {
  ensureDir(outputDir);
  const proofBundleDir = path.join(outputDir, "proof-bundle");
  ensureDir(proofBundleDir);
  writeJson(path.join(outputDir, "example.json"), example.example ?? example);
  writeFileSync(path.join(outputDir, "example.md"), renderExampleMarkdown(example.example ?? example), "utf8");
}

function buildManifest(generatedAt, compiler, lint) {
  return {
    contract: "teacher_v3_promotable_examples_manifest.v1",
    generatedAt,
    outputRoot: "artifacts/teacher-v3-promotable-examples",
    lanes: {
      compiler: {
        proposalId: compiler.proposal.proposalId,
        proposalStatus: compiler.proposal.status,
        reviewMode: compiler.proposal.replayGate.reviewMode,
        proofBundleDir: "compiler/proof-bundle",
        exampleJson: "compiler/example.json",
        exampleMd: "compiler/example.md",
        proofBundleDigest: compiler.proofBundleDigest,
      },
      lint: {
        proposalId: lint.proposal.proposalId,
        proposalStatus: lint.proposal.status,
        reviewMode: lint.proposal.replayGate.reviewMode,
        proofBundleDir: "lint/proof-bundle",
        exampleJson: "lint/example.json",
        exampleMd: "lint/example.md",
        proofBundleDigest: lint.proofBundleDigest,
      },
    },
  };
}

export function buildTeacherV3PromotableExamples(targetRoot = outputRoot) {
  const root = path.resolve(targetRoot);
  const store = makeStore();
  const compiler = buildCompilerExample(store, root);
  const lint = buildLintExample(store, root);
  const manifest = buildManifest(GENERATED_AT, compiler, lint);

  return {
    generatedAt: GENERATED_AT,
    outputRoot: root,
    manifest,
    compiler,
    lint,
  };
}

export function writeTeacherV3PromotableExamples(targetRoot = outputRoot) {
  const root = path.resolve(targetRoot);
  if (existsSync(root)) {
    rmSync(root, { recursive: true, force: true });
  }
  ensureDir(root);

  const examples = buildTeacherV3PromotableExamples(root);
  const compilerDir = path.join(root, "compiler");
  const lintDir = path.join(root, "lint");
  ensureDir(compilerDir);
  ensureDir(lintDir);

  writeExampleArtifacts(examples.compiler, compilerDir);
  writeExampleArtifacts(examples.lint, lintDir);
  writeJson(path.join(root, "manifest.json"), buildManifest(examples.generatedAt, examples.compiler, examples.lint));
  writeFileSync(
    path.join(root, "README.md"),
    [
      "# Teacher v3 promotable worked examples",
      "",
      "This directory holds one honest full worked example each for the promotable Teacher v3 compiler and lint lanes.",
      "",
      "- Compiler: stored proposal promoted after replay, proof bundle reviewable, canary off by default.",
      "- Lint: stored proposal remains reviewable/promotable, proof bundle reviewable, public release truth stays bounded.",
      "",
      "See `manifest.json` for the exact file layout and per-lane proof bundle digests.",
      "",
    ].join("\n"),
    "utf8",
  );

  return {
    ...examples,
    root,
  };
}

function main() {
  const targetRoot = process.argv[2] ? path.resolve(process.argv[2]) : outputRoot;
  const result = writeTeacherV3PromotableExamples(targetRoot);
  process.stdout.write(`${JSON.stringify({
    generatedAt: result.generatedAt,
    root: result.root,
    manifestPath: path.join(result.root, "manifest.json"),
    compilerExamplePath: path.join(result.root, "compiler", "example.json"),
    lintExamplePath: path.join(result.root, "lint", "example.json"),
  }, null, 2)}\n`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
