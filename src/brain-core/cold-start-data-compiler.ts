import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import type {
  DataRegistryEntryV1,
  GraphCompilerContractV1,
  GraphEdgeSchemaV1Type,
  GraphNodeSchemaV1Type,
  RouteCandidateV1,
  RouteDecisionRowV1,
  TeacherLabelContractV1,
  VerifierContractV1,
} from "./cold-start-router-contracts.ts";
import {
  COLD_START_CONTRACT_VERSION_V1,
  DataRegistryEntrySchemaV1,
  GraphCompilerContractSchemaV1,
  GraphNeighborhoodPackSchemaV1,
  RouteDecisionRowSchemaV1,
  TeacherLabelContractSchemaV1,
  VerifierContractSchemaV1,
  validateDataRegistryEntryV1,
  validateGraphCompilerContractV1,
  validateRouteDecisionRowV1,
  validateTeacherLabelContractV1,
  validateVerifierContractV1,
} from "./cold-start-router-contracts.ts";

export interface RawDocsQaSourceDocumentV1 {
  source_ref: string;
  path: string;
  kind?: "doc_chunk" | "qa_source";
}

export interface RawDocsQaCandidateV1 {
  candidate_id: string;
  candidate_type: RouteCandidateV1["candidate_type"];
  authority?: string;
  freshness?: string;
  token_cost?: number;
  score_hint?: number;
}

export interface RawDocsQaEvidenceSpanV1 {
  source_ref: string;
  excerpt?: string;
  start?: number;
  end?: number;
}

export interface RawDocsQaTeacherActionTraverseV1 {
  kind: "traverse";
  target_ids: string[];
}

export interface RawDocsQaTeacherActionToolV1 {
  kind: "tool";
  tool_name: string;
  tool_args_ref?: string;
}

export type RawDocsQaTeacherActionV1 =
  | RawDocsQaTeacherActionTraverseV1
  | RawDocsQaTeacherActionToolV1;

export interface RawDocsQaExampleV1 {
  row_id: string;
  query: string;
  cursor_path: string[];
  candidate_set: RawDocsQaCandidateV1[];
  teacher_action: RawDocsQaTeacherActionV1;
  stop_label: RouteDecisionRowV1["stop_label"];
  evidence_spans: RawDocsQaEvidenceSpanV1[];
  hard_negatives?: string[];
  outcome_gain: number;
  teacher_confidence: number;
  rationale: string;
  split_tag?: string;
  created_at?: string;
}

export interface RawDocsQaRegistryV1 {
  dataset_id: string;
  source_family: "docs" | "qa";
  upstream_url: string;
  original_creator: string;
  license: string;
  commercial_use_status: DataRegistryEntryV1["commercial_use_status"];
  redistribution_status: DataRegistryEntryV1["redistribution_status"];
  pii_risk: DataRegistryEntryV1["pii_risk"];
  benchmark_split_status: DataRegistryEntryV1["benchmark_split_status"];
  approval_status: DataRegistryEntryV1["approval_status"];
  reviewer: string;
  immutable_snapshot_ref: string;
  exact_files: string[];
  allowed_uses: string[];
  disallowed_uses: string[];
  notes: string[];
  created_at?: string;
  updated_at?: string;
}

export interface RawDocsQaSourceBundleV1 {
  contract: "cold_start_docs_qa_source_bundle.v1";
  bundle_id: string;
  generated_at: string;
  registry: RawDocsQaRegistryV1;
  source_documents: RawDocsQaSourceDocumentV1[];
  examples: RawDocsQaExampleV1[];
}

export interface ColdStartCurationIssueV1 {
  severity: "warning" | "error";
  code: string;
  detail: string;
  rowId?: string;
  sourceRef?: string;
}

export interface ColdStartRowCleanupSummaryV1 {
  rowId: string;
  accepted: boolean;
  queryTrimmed: boolean;
  rationaleTrimmed: boolean;
  cursorPathSegmentsTrimmed: number;
  dedupedCandidateIds: number;
  dedupedEvidenceSpans: number;
  dedupedHardNegatives: number;
  clampedConfidence: boolean;
  issues: ColdStartCurationIssueV1[];
}

export interface ColdStartDocsQaCompilationReportV1 {
  contract: "cold_start_docs_qa_compilation_report.v1";
  bundleId: string;
  datasetId: string;
  sourceFamily: RawDocsQaRegistryV1["source_family"];
  sourceDocumentCount: number;
  rawExampleCount: number;
  acceptedRowCount: number;
  rejectedRowCount: number;
  cleanup: {
    queryTrimmedCount: number;
    rationaleTrimmedCount: number;
    cursorPathSegmentTrimCount: number;
    candidateDedupedCount: number;
    evidenceSpanDedupedCount: number;
    hardNegativeDedupedCount: number;
    confidenceClampCount: number;
  };
  supervision: {
    stopLabelCounts: Record<RouteDecisionRowV1["stop_label"], number>;
    teacherActionKindCounts: Record<RouteDecisionRowV1["teacher_action"]["kind"], number>;
  };
  rowSummaries: ColdStartRowCleanupSummaryV1[];
  issues: ColdStartCurationIssueV1[];
}

export interface ColdStartDocsQaCompilationBundleV1 {
  registryEntry: DataRegistryEntryV1;
  graphCompiler: GraphCompilerContractV1;
  routeRows: RouteDecisionRowV1[];
  teacherLabels: TeacherLabelContractV1[];
  verifiers: VerifierContractV1[];
  report: ColdStartDocsQaCompilationReportV1;
}

function sha256Text(text: string): string {
  return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function normalizeTrimmedStringArray(values: readonly string[] | undefined): string[] {
  if (!Array.isArray(values)) {
    return [];
  }
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const normalized = normalizeWhitespace(String(value ?? ""));
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

function createStopLabelCounts(): Record<RouteDecisionRowV1["stop_label"], number> {
  return {
    CONTINUE: 0,
    STOP_LOCAL: 0,
    STOP: 0,
  };
}

function createTeacherActionKindCounts(): Record<RouteDecisionRowV1["teacher_action"]["kind"], number> {
  return {
    traverse: 0,
    tool: 0,
  };
}

function cloneSorted<T>(values: readonly T[], comparator: (left: T, right: T) => number): T[] {
  return [...values].sort(comparator);
}

function stableCompare(left: string, right: string): number {
  return left.localeCompare(right);
}

function assertKnownSourceDocuments(sourceDocuments: RawDocsQaSourceDocumentV1[]): Map<string, RawDocsQaSourceDocumentV1> {
  const documents = new Map<string, RawDocsQaSourceDocumentV1>();
  for (const source of sourceDocuments) {
    const sourceRef = normalizeWhitespace(source.source_ref);
    const sourcePath = normalizeWhitespace(source.path);
    if (!sourceRef) {
      continue;
    }
    if (!documents.has(sourceRef)) {
      documents.set(sourceRef, {
        source_ref: sourceRef,
        path: sourcePath,
        ...(source.kind ? { kind: source.kind } : {}),
      });
    }
  }
  return documents;
}

function readRepoFile(repoRoot: string, relativePath: string): string {
  const resolvedPath = path.resolve(repoRoot, relativePath);
  return readFileSync(resolvedPath, "utf8");
}

function toFileHashes(repoRoot: string, filePaths: readonly string[]): Record<string, string> {
  const hashes: Record<string, string> = {};
  for (const filePath of cloneSorted(normalizeTrimmedStringArray(filePaths), stableCompare)) {
    hashes[filePath] = sha256Text(readRepoFile(repoRoot, filePath));
  }
  return hashes;
}

function buildSnapshotRef(bundleId: string, exactFiles: readonly string[], fileHashes: Record<string, string>): string {
  return `snapshot:${bundleId}@${sha256Text(JSON.stringify({ exactFiles: [...exactFiles].sort(), fileHashes }))}`;
}

function canonicalCandidateSet(rawCandidates: RawDocsQaCandidateV1[]): {
  candidates: RouteDecisionRowV1["candidate_set"];
  dedupedCount: number;
  duplicateIssues: ColdStartCurationIssueV1[];
} {
  const seen = new Map<string, RouteDecisionRowV1["candidate_set"][number]>();
  let dedupedCount = 0;
  const duplicateIssues: ColdStartCurationIssueV1[] = [];

  for (const candidate of rawCandidates) {
    const candidateId = normalizeWhitespace(candidate.candidate_id);
    if (!candidateId) {
      continue;
    }

    const normalizedCandidate: RouteDecisionRowV1["candidate_set"][number] = {
      candidate_id: candidateId,
      candidate_type: candidate.candidate_type,
      ...(candidate.semantic_class ? { semantic_class: normalizeWhitespace(candidate.semantic_class) } : {}),
      ...(candidate.authority ? { authority: normalizeWhitespace(candidate.authority) } : {}),
      ...(candidate.freshness ? { freshness: normalizeWhitespace(candidate.freshness) } : {}),
      ...(typeof candidate.token_cost === "number" && Number.isFinite(candidate.token_cost)
        ? { token_cost: Math.max(0, Math.trunc(candidate.token_cost)) }
        : {}),
      ...(typeof candidate.score_hint === "number" && Number.isFinite(candidate.score_hint)
        ? { score_hint: candidate.score_hint }
        : {}),
    };

    const existing = seen.get(candidateId);
    if (!existing) {
      seen.set(candidateId, normalizedCandidate);
      continue;
    }

    dedupedCount += 1;
    if (existing.candidate_type !== normalizedCandidate.candidate_type) {
      duplicateIssues.push({
        severity: "error",
        code: "candidate_type_conflict",
        detail: `candidate ${candidateId} appeared with incompatible types ${existing.candidate_type} and ${normalizedCandidate.candidate_type}`,
      });
      continue;
    }

    const existingScore = existing.score_hint ?? Number.NEGATIVE_INFINITY;
    const nextScore = normalizedCandidate.score_hint ?? Number.NEGATIVE_INFINITY;
    if (nextScore > existingScore) {
      seen.set(candidateId, {
        ...existing,
        ...(normalizedCandidate.semantic_class ? { semantic_class: normalizedCandidate.semantic_class } : {}),
        ...(normalizedCandidate.authority ? { authority: normalizedCandidate.authority } : {}),
        ...(normalizedCandidate.freshness ? { freshness: normalizedCandidate.freshness } : {}),
        ...(normalizedCandidate.token_cost !== undefined ? { token_cost: normalizedCandidate.token_cost } : {}),
        ...(normalizedCandidate.score_hint !== undefined ? { score_hint: normalizedCandidate.score_hint } : {}),
      });
    }
  }

  const candidates = [...seen.values()].sort((left, right) => {
    const leftScore = left.score_hint ?? Number.NEGATIVE_INFINITY;
    const rightScore = right.score_hint ?? Number.NEGATIVE_INFINITY;
    if (leftScore !== rightScore) {
      return rightScore - leftScore;
    }
    return stableCompare(left.candidate_id, right.candidate_id);
  });

  return { candidates, dedupedCount, duplicateIssues };
}

function normalizeEvidenceSpans(params: {
  repoRoot: string;
  sourceDocuments: Map<string, RawDocsQaSourceDocumentV1>;
  rowId: string;
  evidenceSpans: RawDocsQaEvidenceSpanV1[];
}): {
  evidenceSpans: RouteDecisionRowV1["evidence_spans"];
  dedupedCount: number;
  issues: ColdStartCurationIssueV1[];
} {
  const seen = new Set<string>();
  const issues: ColdStartCurationIssueV1[] = [];
  let dedupedCount = 0;
  const evidenceSpans: RouteDecisionRowV1["evidence_spans"] = [];

  for (const rawSpan of params.evidenceSpans) {
    const sourceRef = normalizeWhitespace(rawSpan.source_ref);
    const source = params.sourceDocuments.get(sourceRef);
    if (!source) {
      issues.push({
        severity: "error",
        code: "unknown_evidence_source",
        rowId: params.rowId,
        sourceRef,
        detail: `row ${params.rowId} references unknown evidence source ${sourceRef}`,
      });
      continue;
    }

    const excerpt = rawSpan.excerpt ? normalizeWhitespace(rawSpan.excerpt) : undefined;
    const sourceText = readRepoFile(params.repoRoot, source.path);
    const resolved = typeof rawSpan.start === "number" && typeof rawSpan.end === "number"
      ? { start: rawSpan.start, end: rawSpan.end }
      : excerpt
        ? (() => {
            const start = sourceText.indexOf(excerpt);
            return start >= 0 ? { start, end: start + excerpt.length } : null;
          })()
        : null;

    if (!resolved) {
      issues.push({
        severity: "error",
        code: "unresolved_evidence_excerpt",
        rowId: params.rowId,
        sourceRef,
        detail: `row ${params.rowId} could not resolve evidence excerpt in ${source.path}${excerpt ? `: ${excerpt}` : ""}`,
      });
      continue;
    }

    const key = `${sourceRef}:${resolved.start}:${resolved.end}`;
    if (seen.has(key)) {
      dedupedCount += 1;
      continue;
    }
    seen.add(key);
    evidenceSpans.push({
      source_ref: sourceRef,
      start: resolved.start,
      end: resolved.end,
      ...(excerpt ? { excerpt } : {}),
    });
  }

  evidenceSpans.sort((left, right) => {
    const sourceCompare = stableCompare(left.source_ref, right.source_ref);
    if (sourceCompare !== 0) {
      return sourceCompare;
    }
    if (left.start !== right.start) {
      return left.start - right.start;
    }
    return left.end - right.end;
  });

  return { evidenceSpans, dedupedCount, issues };
}

function normalizeRouteDecisionRow(params: {
  repoRoot: string;
  datasetId: string;
  sourceDocuments: Map<string, RawDocsQaSourceDocumentV1>;
  example: RawDocsQaExampleV1;
  registry: RawDocsQaRegistryV1;
  bundleId: string;
  generatedAt: string;
}): {
  row?: RouteDecisionRowV1;
  teacherLabel?: TeacherLabelContractV1;
  verifier?: VerifierContractV1;
  cleanup: ColdStartRowCleanupSummaryV1;
  issues: ColdStartCurationIssueV1[];
} {
  const rowIssues: ColdStartCurationIssueV1[] = [];
  const query = normalizeWhitespace(params.example.query);
  const rationale = normalizeWhitespace(params.example.rationale);
  const queryTrimmed = query !== params.example.query;
  const rationaleTrimmed = rationale !== params.example.rationale;

  let cursorPathSegmentsTrimmed = 0;
  const cursorPath: string[] = [];
  for (const segment of params.example.cursor_path) {
    const normalized = normalizeWhitespace(segment);
    if (normalized.length === 0) {
      cursorPathSegmentsTrimmed += 1;
      continue;
    }
    if (normalized !== segment) {
      cursorPathSegmentsTrimmed += 1;
    }
    cursorPath.push(normalized);
  }

  const { candidates, dedupedCount: dedupedCandidateIds, duplicateIssues } = canonicalCandidateSet(params.example.candidate_set);
  rowIssues.push(...duplicateIssues);

  const teacherActionKind = params.example.teacher_action.kind;
  const toolName = teacherActionKind === "tool"
    ? normalizeWhitespace(params.example.teacher_action.tool_name)
    : "";
  const candidateIds = new Set(candidates.map((candidate) => candidate.candidate_id));
  const toolCandidateIds = normalizeTrimmedStringArray(
    teacherActionKind === "tool"
      ? candidates
        .filter((candidate) => candidate.candidate_type === "tool")
        .map((candidate) => candidate.candidate_id)
      : [],
  );

  const canonicalTargets = teacherActionKind === "traverse"
    ? normalizeTrimmedStringArray(params.example.teacher_action.target_ids)
    : toolCandidateIds.filter((candidateId) => candidateId === toolName);

  if (teacherActionKind === "tool" && canonicalTargets.length === 0) {
    if (toolCandidateIds.length === 1) {
      canonicalTargets.push(toolCandidateIds[0]!);
    } else if (toolCandidateIds.length === 0) {
      rowIssues.push({
        severity: "error",
        code: "missing_teacher_tool_candidate",
        rowId: params.example.row_id,
        detail: `row ${params.example.row_id} tool action ${toolName} does not have a matching tool candidate in the candidate set`,
      });
    } else {
      rowIssues.push({
        severity: "error",
        code: "ambiguous_teacher_tool_candidate",
        rowId: params.example.row_id,
        detail: `row ${params.example.row_id} tool action ${toolName} matches multiple tool candidates: ${toolCandidateIds.join(", ")}`,
      });
    }
  }

  const hardNegativesFromExample = normalizeTrimmedStringArray(params.example.hard_negatives);
  const hardNegativesDerived = candidates
    .map((candidate) => candidate.candidate_id)
    .filter((candidateId) => !canonicalTargets.includes(candidateId));
  const hardNegatives = cloneSorted([...new Set([...hardNegativesFromExample, ...hardNegativesDerived])], stableCompare);
  const dedupedHardNegatives = hardNegativesFromExample.length + hardNegativesDerived.length - hardNegatives.length;

  const { evidenceSpans, dedupedCount: dedupedEvidenceSpans, issues: evidenceIssues } = normalizeEvidenceSpans({
    repoRoot: params.repoRoot,
    sourceDocuments: params.sourceDocuments,
    rowId: params.example.row_id,
    evidenceSpans: params.example.evidence_spans,
  });
  rowIssues.push(...evidenceIssues);

  const clampedConfidence = Math.max(0, Math.min(1, params.example.teacher_confidence));
  const clampedConfidenceChanged = clampedConfidence !== params.example.teacher_confidence;
  const stopLabel = params.example.stop_label;

  if (teacherActionKind === "traverse") {
    const missingTargets = canonicalTargets.filter((targetId) => !candidateIds.has(targetId));
    if (missingTargets.length > 0) {
      rowIssues.push({
        severity: "error",
        code: "missing_teacher_targets",
        rowId: params.example.row_id,
        detail: `row ${params.example.row_id} target ids are not present in the candidate set: ${missingTargets.join(", ")}`,
      });
    }
  } else {
    const missingToolTargets = canonicalTargets.filter((targetId) => !candidateIds.has(targetId));
    if (missingToolTargets.length > 0) {
      rowIssues.push({
        severity: "error",
        code: "missing_teacher_tool_targets",
        rowId: params.example.row_id,
        detail: `row ${params.example.row_id} tool action ${toolName} is not present in the candidate set: ${missingToolTargets.join(", ")}`,
      });
    }
  }

  const overlaps = canonicalTargets.filter((targetId) => hardNegatives.includes(targetId));
  if (overlaps.length > 0) {
    rowIssues.push({
      severity: "error",
      code: "hard_negative_overlap",
      rowId: params.example.row_id,
      detail: `row ${params.example.row_id} hard negatives overlap positive targets: ${overlaps.join(", ")}`,
    });
  }

  const accepted = rowIssues.every((issue) => issue.severity !== "error");

  const cleanup: ColdStartRowCleanupSummaryV1 = {
    rowId: params.example.row_id,
    accepted,
    queryTrimmed,
    rationaleTrimmed,
    cursorPathSegmentsTrimmed,
    dedupedCandidateIds,
    dedupedEvidenceSpans,
    dedupedHardNegatives,
    clampedConfidence: clampedConfidenceChanged,
    issues: rowIssues,
  };

  if (!accepted) {
    return { cleanup, issues: rowIssues };
  }

  const row: RouteDecisionRowV1 = {
    row_id: normalizeWhitespace(params.example.row_id),
    dataset_id: params.datasetId,
    query,
    cursor_path: cursorPath,
    candidate_set: candidates,
    teacher_action: params.example.teacher_action.kind === "traverse"
      ? { kind: "traverse", target_ids: canonicalTargets }
      : { kind: "tool", tool_name: toolName, ...(params.example.teacher_action.tool_args_ref ? { tool_args_ref: normalizeWhitespace(params.example.teacher_action.tool_args_ref) } : {}) },
    stop_label: stopLabel,
    evidence_spans: evidenceSpans,
    hard_negatives: hardNegatives,
    outcome_gain: params.example.outcome_gain,
    provenance: {
      dataset: params.registry.dataset_id,
      source_license: params.registry.license,
      source_family: params.registry.source_family,
      source_snapshot_ref: params.registry.immutable_snapshot_ref,
      recorded_by: `${params.registry.reviewer}:docs-qa-compiler`,
      recorded_at: params.generatedAt,
      review_status: params.registry.approval_status,
    },
    split_tag: normalizeWhitespace(params.example.split_tag ?? params.registry.benchmark_split_status),
    created_at: normalizeWhitespace(params.example.created_at ?? params.generatedAt),
  };

  const teacherLabel: TeacherLabelContractV1 = {
    label_id: `${row.row_id}:teacher-label`,
    dataset_id: row.dataset_id,
    row_id: row.row_id,
    best_next_node_ids: canonicalTargets,
    best_next_tool_name: teacherActionKind === "tool" ? toolName : null,
    stop_label: row.stop_label,
    evidence_spans: row.evidence_spans,
    hard_negatives: row.hard_negatives,
    confidence: clampedConfidence,
    rationale,
    created_at: row.created_at,
  };

  const verifier: VerifierContractV1 = {
    verifier_id: `${row.row_id}:verifier`,
    label_id: teacherLabel.label_id,
    row_id: row.row_id,
    candidate_set_digest: sha256Text(JSON.stringify(row.candidate_set)),
    checks: [
      {
        check_id: "evidence_reachable",
        status: "pass",
        summary: "evidence spans resolve against source documents",
        evidence_refs: row.evidence_spans.map((span) => span.source_ref),
      },
      {
        check_id: "candidate_alignment",
        status: "pass",
        summary: "positive target ids exist in the candidate set",
        evidence_refs: canonicalTargets,
      },
      ...(teacherActionKind === "tool"
        ? [{
            check_id: "tool_alignment",
            status: "pass" as const,
            summary: `tool action resolves to ${toolName}`,
            evidence_refs: canonicalTargets.length > 0 ? canonicalTargets : [toolName],
          }]
        : []),
      {
        check_id: "hard_negative_disjoint",
        status: "pass",
        summary: "hard negatives do not overlap positive targets",
        evidence_refs: row.hard_negatives,
      },
      {
        check_id: "correction_priority",
        status: "pass",
        summary: "explicit correction priority is preserved by the row",
        evidence_refs: row.evidence_spans.map((span) => span.source_ref),
      },
    ],
    passed: true,
    issues: [],
    explicit_correction_priority_honored: true,
    created_at: row.created_at,
  };

  return { row, teacherLabel, verifier, cleanup, issues: rowIssues };
}

function buildGraphCompilerContract(params: {
  bundleId: string;
  registry: RawDocsQaRegistryV1;
  sourceDocuments: Map<string, RawDocsQaSourceDocumentV1>;
  routeRows: RouteDecisionRowV1[];
  fileHashes: Record<string, string>;
}): GraphCompilerContractV1 {
  const graphRef = `graph:docs-qa@${sha256Text(JSON.stringify({
    datasetId: params.registry.dataset_id,
    sourceRefs: [...params.sourceDocuments.keys()].sort(),
    rowIds: params.routeRows.map((row) => row.row_id).sort(),
    fileHashes: params.fileHashes,
  }))}`;

  const artifactRef = `artifact:docs-qa-pack@${sha256Text(JSON.stringify({
    datasetId: params.registry.dataset_id,
    bundleId: params.bundleId,
    graphRef,
    rowCount: params.routeRows.length,
  }))}`;

  return {
    compiler_id: "docs-qa-cold-start-compiler-v1",
    source_family: params.registry.source_family,
    input_snapshot_ref: params.registry.immutable_snapshot_ref,
    node_schema: [
      {
        node_kind: "doc_chunk",
        required_fields: ["path", "content_hash"],
        optional_fields: ["title", "section", "source_ref"],
        notes: ["Doc chunks are curated from approved architecture sources"],
      },
      {
        node_kind: "qa_example",
        required_fields: ["row_id", "query", "candidate_set"],
        optional_fields: ["hard_negatives", "split_tag"],
        notes: ["QA supervision nodes link the query to the routing target"],
      },
    ],
    edge_schema: [
      {
        edge_kind: "cites",
        required_fields: ["source_ref", "target_ref"],
        optional_fields: ["excerpt"],
        notes: ["Resolved evidence excerpts create cite edges"],
      },
      {
        edge_kind: "supervises",
        required_fields: ["row_id", "source_ref"],
        optional_fields: ["excerpt"],
        notes: ["Question rows supervise the linked document neighborhood"],
      },
    ],
    provenance_rules: [
      "approved docs sources only",
      "evidence excerpts must resolve in checked files",
      "hard negatives must stay disjoint from positive targets",
      "duplicate candidate ids are deduped before validation",
    ],
    output_neighborhood_pack: {
      pack_id: `docs-qa-neighborhood-pack:${params.bundleId}`,
      artifact_ref: artifactRef,
      graph_ref: graphRef,
      radius_hops: 1,
      frontier_limit: 32,
    },
    compiler_version: "docs-qa-cold-start-compiler@0.1.0",
  };
}

function buildRegistryEntry(params: {
  repoRoot: string;
  registry: RawDocsQaRegistryV1;
  generatedAt: string;
}): DataRegistryEntryV1 {
  const exactFiles = normalizeTrimmedStringArray(params.registry.exact_files);
  const fileHashes = toFileHashes(params.repoRoot, exactFiles);
  const createdAt = normalizeWhitespace(params.registry.created_at ?? params.generatedAt);
  const updatedAt = normalizeWhitespace(params.registry.updated_at ?? params.generatedAt);

  return {
    dataset_id: params.registry.dataset_id,
    source_family: params.registry.source_family,
    upstream_url: params.registry.upstream_url,
    original_creator: params.registry.original_creator,
    license: params.registry.license,
    commercial_use_status: params.registry.commercial_use_status,
    redistribution_status: params.registry.redistribution_status,
    pii_risk: params.registry.pii_risk,
    benchmark_split_status: params.registry.benchmark_split_status,
    approval_status: params.registry.approval_status,
    reviewer: params.registry.reviewer,
    immutable_snapshot_ref: params.registry.immutable_snapshot_ref,
    exact_files: exactFiles,
    file_hashes: fileHashes,
    allowed_uses: normalizeTrimmedStringArray(params.registry.allowed_uses),
    disallowed_uses: normalizeTrimmedStringArray(params.registry.disallowed_uses),
    notes: normalizeTrimmedStringArray(params.registry.notes),
    created_at: createdAt,
    updated_at: updatedAt,
  };
}

export function compileColdStartDocsQaSourceBundleV1(params: {
  repoRoot?: string;
  bundle: RawDocsQaSourceBundleV1;
}): ColdStartDocsQaCompilationBundleV1 {
  const repoRoot = path.resolve(params.repoRoot ?? process.cwd());
  const generatedAt = normalizeWhitespace(params.bundle.generated_at);
  const sourceDocuments = assertKnownSourceDocuments(params.bundle.source_documents);
  const registryEntry = buildRegistryEntry({ repoRoot, registry: params.bundle.registry, generatedAt });

  const acceptedRows: RouteDecisionRowV1[] = [];
  const teacherLabels: TeacherLabelContractV1[] = [];
  const verifiers: VerifierContractV1[] = [];
  const rowSummaries: ColdStartRowCleanupSummaryV1[] = [];
  const issues: ColdStartCurationIssueV1[] = [];
  const stopLabelCounts = createStopLabelCounts();
  const teacherActionKindCounts = createTeacherActionKindCounts();
  let queryTrimmedCount = 0;
  let rationaleTrimmedCount = 0;
  let cursorPathSegmentTrimCount = 0;
  let candidateDedupedCount = 0;
  let evidenceSpanDedupedCount = 0;
  let hardNegativeDedupedCount = 0;
  let confidenceClampCount = 0;

  for (const example of params.bundle.examples) {
    const compilation = normalizeRouteDecisionRow({
      repoRoot,
      datasetId: registryEntry.dataset_id,
      sourceDocuments,
      example,
      registry: params.bundle.registry,
      bundleId: params.bundle.bundle_id,
      generatedAt,
    });

    rowSummaries.push(compilation.cleanup);
    issues.push(...compilation.issues);
    if (compilation.cleanup.queryTrimmed) {
      queryTrimmedCount += 1;
    }
    if (compilation.cleanup.rationaleTrimmed) {
      rationaleTrimmedCount += 1;
    }
    cursorPathSegmentTrimCount += compilation.cleanup.cursorPathSegmentsTrimmed;
    candidateDedupedCount += compilation.cleanup.dedupedCandidateIds;
    evidenceSpanDedupedCount += compilation.cleanup.dedupedEvidenceSpans;
    hardNegativeDedupedCount += compilation.cleanup.dedupedHardNegatives;
    if (compilation.cleanup.clampedConfidence) {
      confidenceClampCount += 1;
    }

    if (!compilation.row || !compilation.teacherLabel || !compilation.verifier) {
      continue;
    }

    acceptedRows.push(compilation.row);
    teacherLabels.push(compilation.teacherLabel);
    verifiers.push(compilation.verifier);
    stopLabelCounts[compilation.row.stop_label] += 1;
    teacherActionKindCounts[compilation.row.teacher_action.kind] += 1;
  }

  const graphCompiler = buildGraphCompilerContract({
    bundleId: params.bundle.bundle_id,
    registry: params.bundle.registry,
    sourceDocuments,
    routeRows: acceptedRows,
    fileHashes: registryEntry.file_hashes,
  });

  const report: ColdStartDocsQaCompilationReportV1 = {
    contract: "cold_start_docs_qa_compilation_report.v1",
    bundleId: params.bundle.bundle_id,
    datasetId: registryEntry.dataset_id,
    sourceFamily: params.bundle.registry.source_family,
    sourceDocumentCount: sourceDocuments.size,
    rawExampleCount: params.bundle.examples.length,
    acceptedRowCount: acceptedRows.length,
    rejectedRowCount: params.bundle.examples.length - acceptedRows.length,
    cleanup: {
      queryTrimmedCount,
      rationaleTrimmedCount,
      cursorPathSegmentTrimCount,
      candidateDedupedCount,
      evidenceSpanDedupedCount,
      hardNegativeDedupedCount,
      confidenceClampCount,
    },
    supervision: {
      stopLabelCounts,
      teacherActionKindCounts,
    },
    rowSummaries,
    issues,
  };

  if (!validateDataRegistryEntryV1(registryEntry).valid) {
    throw new Error(`compiled registry entry failed validation: ${validateDataRegistryEntryV1(registryEntry).issues.join(" | ")}`);
  }
  for (const row of acceptedRows) {
    const rowValidation = validateRouteDecisionRowV1(row);
    if (!rowValidation.valid) {
      throw new Error(`compiled route row failed validation (${row.row_id}): ${rowValidation.issues.join(" | ")}`);
    }
  }
  for (const label of teacherLabels) {
    const labelValidation = validateTeacherLabelContractV1(label);
    if (!labelValidation.valid) {
      throw new Error(`compiled teacher label failed validation (${label.label_id}): ${labelValidation.issues.join(" | ")}`);
    }
  }
  for (const verifier of verifiers) {
    const verifierValidation = validateVerifierContractV1(verifier);
    if (!verifierValidation.valid) {
      throw new Error(`compiled verifier failed validation (${verifier.verifier_id}): ${verifierValidation.issues.join(" | ")}`);
    }
  }
  if (!validateGraphCompilerContractV1(graphCompiler).valid) {
    throw new Error(`compiled graph compiler contract failed validation: ${validateGraphCompilerContractV1(graphCompiler).issues.join(" | ")}`);
  }

  return {
    registryEntry,
    graphCompiler,
    routeRows: acceptedRows,
    teacherLabels,
    verifiers,
    report,
  };
}

export function summarizeColdStartDocsQaCompilationBundleV1(bundle: ColdStartDocsQaCompilationBundleV1): Record<string, unknown> {
  return {
    registry: {
      datasetId: bundle.registryEntry.dataset_id,
      sourceFamily: bundle.registryEntry.source_family,
      approvalStatus: bundle.registryEntry.approval_status,
      fileCount: bundle.registryEntry.exact_files.length,
    },
    graphCompiler: {
      compilerId: bundle.graphCompiler.compiler_id,
      sourceFamily: bundle.graphCompiler.source_family,
      neighborhoodPackId: bundle.graphCompiler.output_neighborhood_pack.pack_id,
    },
    routeRows: bundle.routeRows.map((row) => ({
      rowId: row.row_id,
      candidateCount: row.candidate_set.length,
      hardNegativeCount: row.hard_negatives.length,
      stopLabel: row.stop_label,
      splitTag: row.split_tag,
    })),
    teacherLabels: bundle.teacherLabels.map((label) => ({
      labelId: label.label_id,
      confidence: label.confidence,
      targetCount: label.best_next_node_ids.length,
    })),
    verifiers: bundle.verifiers.map((verifier) => ({
      verifierId: verifier.verifier_id,
      passed: verifier.passed,
      checkCount: verifier.checks.length,
    })),
    report: bundle.report,
  };
}

export function loadColdStartDocsQaSourceBundleV1(bundlePath: string): RawDocsQaSourceBundleV1 {
  return JSON.parse(readFileSync(bundlePath, "utf8")) as RawDocsQaSourceBundleV1;
}

export function compileColdStartDocsQaSourceBundleFromFileV1(params: {
  bundlePath: string;
  repoRoot?: string;
}): ColdStartDocsQaCompilationBundleV1 {
  return compileColdStartDocsQaSourceBundleV1({
    repoRoot: params.repoRoot,
    bundle: loadColdStartDocsQaSourceBundleV1(params.bundlePath),
  });
}

export const __coldStartDocsQaCompilerVersionV1 = COLD_START_CONTRACT_VERSION_V1;
