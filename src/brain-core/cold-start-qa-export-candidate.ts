import { closeSync, existsSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { compileColdStartDocsQaSourceBundleV1, type ColdStartDocsQaCompilationBundleV1, type RawDocsQaExampleV1, type RawDocsQaSourceBundleV1, type RawDocsQaSourceDocumentV1 } from "./cold-start-data-compiler.ts";
import { type DataRegistryEntryV1, type RouteDecisionRowV1, validateDataRegistryEntryV1, validateRouteDecisionRowV1 } from "./cold-start-router-contracts.ts";
import { loadColdStartSourceIntakeRegistryV1, type ColdStartSourceIntakeCardV1 } from "./cold-start-source-intake.ts";

export const COLD_START_QA_SNAPSHOT_EXPORT_CANDIDATE_CONTRACT_V1 = "cold_start_qa_snapshot_export_candidate.v1" as const;

export type ColdStartQaSnapshotFamilyV1 = "hotpotqa" | "musique";

export interface ColdStartQaSnapshotDatasetResultV1 {
  datasetFamily: ColdStartQaSnapshotFamilyV1;
  datasetId: string;
  snapshotRef: string;
  sampleStrategy: string;
  sourceDocuments: RawDocsQaSourceDocumentV1[];
  rawBundle: RawDocsQaSourceBundleV1;
  compilation: ColdStartDocsQaCompilationBundleV1;
}

export interface ColdStartQaSnapshotExportCandidateV1 {
  contract: typeof COLD_START_QA_SNAPSHOT_EXPORT_CANDIDATE_CONTRACT_V1;
  export_id: string;
  generated_at: string;
  review_status: "under_review";
  source_registry_path: string;
  workspace_root: string;
  datasets: ColdStartQaSnapshotDatasetResultV1[];
  registry_entries: DataRegistryEntryV1[];
  route_rows: RouteDecisionRowV1[];
  notes: string[];
}

export interface ColdStartQaSnapshotExportCandidateSummaryV1 {
  exportId: string;
  generatedAt: string;
  reviewStatus: "under_review";
  datasetCount: number;
  routeRowCount: number;
  registryEntryCount: number;
  datasetIds: string[];
  snapshotRefs: string[];
  sampleStrategies: string[];
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_OPENCLAWBRAIN_REPO_ROOT = path.resolve(__dirname, "..", "..");
const DEFAULT_WORKSPACE_ROOT = path.resolve(DEFAULT_OPENCLAWBRAIN_REPO_ROOT, "..");

function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function slugify(value: string): string {
  return normalizeWhitespace(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "item";
}

function sampleFirstJsonArrayObjects<T>(filePath: string, limit: number): T[] {
  const fd = openSync(filePath, "r");
  const buffer = Buffer.alloc(128 * 1024);
  const results: T[] = [];
  let startedArray = false;
  let capturing = false;
  let depth = 0;
  let inString = false;
  let escape = false;
  let current = "";

  try {
    while (results.length < limit) {
      const bytesRead = readSync(fd, buffer, 0, buffer.length, null);
      if (bytesRead === 0) {
        break;
      }

      const chunk = buffer.toString("utf8", 0, bytesRead);
      for (let index = 0; index < chunk.length; index += 1) {
        const ch = chunk[index];
        if (!startedArray) {
          if (ch === "[") {
            startedArray = true;
          }
          continue;
        }

        if (!capturing) {
          if (ch === "{") {
            capturing = true;
            depth = 1;
            inString = false;
            escape = false;
            current = "{";
          } else if (ch === "]") {
            return results;
          }
          continue;
        }

        current += ch;
        if (inString) {
          if (escape) {
            escape = false;
          } else if (ch === "\\") {
            escape = true;
          } else if (ch === '"') {
            inString = false;
          }
          continue;
        }

        if (ch === '"') {
          inString = true;
          continue;
        }
        if (ch === "{") {
          depth += 1;
          continue;
        }
        if (ch === "}") {
          depth -= 1;
          if (depth === 0) {
            results.push(JSON.parse(current) as T);
            capturing = false;
            current = "";
            if (results.length >= limit) {
              return results;
            }
          }
        }
      }
    }

    return results;
  } finally {
    closeSync(fd);
  }
}

function sampleFirstJsonlObjects<T>(filePath: string, limit: number): T[] {
  const fd = openSync(filePath, "r");
  const buffer = Buffer.alloc(128 * 1024);
  const results: T[] = [];
  let line = "";

  try {
    while (results.length < limit) {
      const bytesRead = readSync(fd, buffer, 0, buffer.length, null);
      if (bytesRead === 0) {
        break;
      }

      const chunk = buffer.toString("utf8", 0, bytesRead);
      for (let index = 0; index < chunk.length; index += 1) {
        const ch = chunk[index];
        if (ch === "\n") {
          const trimmed = line.trim();
          if (trimmed.length > 0) {
            results.push(JSON.parse(trimmed) as T);
            if (results.length >= limit) {
              return results;
            }
          }
          line = "";
        } else if (ch !== "\r") {
          line += ch;
        }
      }
    }

    const trimmed = line.trim();
    if (trimmed.length > 0 && results.length < limit) {
      results.push(JSON.parse(trimmed) as T);
    }

    return results;
  } finally {
    closeSync(fd);
  }
}

function buildCandidateId(datasetId: string, title: string): string {
  return `${datasetId}:support:${slugify(title)}`;
}

function normalizeRepoRelativePath(value: string): string {
  return value.replace(/^openclawbrain\//, "");
}

function candidateWorkspaceRoots(workspaceRoot: string): string[] {
  const resolvedWorkspaceRoot = path.resolve(workspaceRoot);
  return [...new Set([
    resolvedWorkspaceRoot,
    path.resolve(resolvedWorkspaceRoot, ".."),
    DEFAULT_WORKSPACE_ROOT,
    path.resolve(DEFAULT_WORKSPACE_ROOT, ".."),
  ])];
}

function resolveSnapshotFilePath(workspaceRoot: string, repoRelativePath: string): { repoRoot: string; filePath: string } {
  for (const repoRoot of candidateWorkspaceRoots(workspaceRoot)) {
    const candidatePath = path.resolve(repoRoot, repoRelativePath);
    if (existsSync(candidatePath)) {
      return { repoRoot, filePath: candidatePath };
    }
  }
  return {
    repoRoot: DEFAULT_WORKSPACE_ROOT,
    filePath: path.resolve(DEFAULT_WORKSPACE_ROOT, repoRelativePath),
  };
}

function getRegistryCard(registryPath: string, datasetId: string): ColdStartSourceIntakeCardV1 {
  const loaded = loadColdStartSourceIntakeRegistryV1(registryPath);
  const card = loaded.cards.find((entry) => entry.registry_entry.dataset_id === datasetId);
  if (!card) {
    throw new Error(`dataset ${datasetId} not found in ${registryPath}`);
  }
  return card;
}

function buildHotpotQaRawBundle(params: {
  workspaceRoot: string;
  card: ColdStartSourceIntakeCardV1;
  generatedAt: string;
  sampleCount: number;
}): ColdStartQaSnapshotDatasetResultV1 {
  const snapshotFile = "openclawbrain/data/cold-start/snapshots/hotpotqa_v1/hotpotqa.github.io__20260405/hotpot_dev_distractor_v1.json";
  const manifestFile = "openclawbrain/data/cold-start/snapshots/hotpotqa_v1/hotpotqa.github.io__20260405/manifest.json";
  const sumsFile = "openclawbrain/data/cold-start/snapshots/hotpotqa_v1/hotpotqa.github.io__20260405/SHA256SUMS";
  const { repoRoot, filePath: sourceFilePath } = resolveSnapshotFilePath(params.workspaceRoot, snapshotFile);
  const examples = sampleFirstJsonArrayObjects<{
    _id?: string;
    question: string;
    answer?: string;
    supporting_facts: Array<[string, number]>;
    level?: string;
    context: Array<[string, string[]]>;
  }>(sourceFilePath, params.sampleCount);

  if (examples.length === 0) {
    throw new Error(`no HotpotQA examples could be sampled from ${snapshotFile}`);
  }

  const sourceDocuments: RawDocsQaSourceDocumentV1[] = [
    {
      source_ref: snapshotFile,
      path: snapshotFile,
      kind: "qa_source",
    },
  ];

  const rawBundle: RawDocsQaSourceBundleV1 = {
    contract: "cold_start_docs_qa_source_bundle.v1",
    bundle_id: "hotpotqa-dev-distractor-export-candidate-v1",
    generated_at: params.generatedAt,
    registry: {
      dataset_id: params.card.registry_entry.dataset_id,
      source_family: "qa",
      upstream_url: params.card.registry_entry.upstream_url,
      original_creator: params.card.registry_entry.original_creator,
      license: params.card.registry_entry.license,
      commercial_use_status: params.card.registry_entry.commercial_use_status,
      redistribution_status: params.card.registry_entry.redistribution_status,
      pii_risk: params.card.registry_entry.pii_risk,
      benchmark_split_status: params.card.registry_entry.benchmark_split_status,
      approval_status: params.card.registry_entry.approval_status,
      reviewer: params.card.registry_entry.reviewer,
      immutable_snapshot_ref: params.card.registry_entry.immutable_snapshot_ref,
      exact_files: [snapshotFile, manifestFile, sumsFile],
      allowed_uses: params.card.registry_entry.allowed_uses,
      disallowed_uses: params.card.registry_entry.disallowed_uses,
      notes: [
        ...params.card.registry_entry.notes,
        "export candidate samples the frozen dev-distractor JSON only and keeps the rest of the snapshot bundle out of the candidate payload",
      ],
      ...(params.card.registry_entry.created_at ? { created_at: params.card.registry_entry.created_at } : {}),
      ...(params.card.registry_entry.updated_at ? { updated_at: params.card.registry_entry.updated_at } : {}),
    },
    source_documents: sourceDocuments,
    examples: examples.map((example, index) => {
      const targetTitles = example.supporting_facts.map(([title]) => normalizeWhitespace(title));
      const stopLabel: RawDocsQaExampleV1["stop_label"] = targetTitles.length === 1 ? "STOP_LOCAL" : "CONTINUE";
      const candidateSet = example.context.map(([title], candidateIndex) => ({
        candidate_id: buildCandidateId(params.card.registry_entry.dataset_id, title),
        candidate_type: "doc_chunk" as const,
        authority: targetTitles.includes(normalizeWhitespace(title)) ? "snapshot_supporting_fact" : "snapshot_context",
        freshness: params.card.registry_entry.benchmark_split_status,
        token_cost: 96 + candidateIndex,
        score_hint: targetTitles.includes(normalizeWhitespace(title)) ? 1 : 0.2,
      }));
      const supportingEvidence = example.supporting_facts.flatMap(([title]) => {
        const normalizedTitle = normalizeWhitespace(title);
        return normalizedTitle.length > 0
          ? [{ source_ref: snapshotFile, excerpt: normalizedTitle }]
          : [];
      });

      return {
        row_id: `hotpotqa-dev-distractor-export-candidate-${index + 1}`,
        query: normalizeWhitespace(example.question),
        cursor_path: [snapshotFile, example._id ?? `sample-${index + 1}`],
        candidate_set: candidateSet,
        teacher_action: {
          kind: "traverse",
          target_ids: targetTitles.map((title) => buildCandidateId(params.card.registry_entry.dataset_id, title)),
        },
        stop_label: stopLabel,
        evidence_spans: supportingEvidence,
        hard_negatives: [],
        outcome_gain: 1,
        teacher_confidence: 0.97,
        rationale: `HotpotQA snapshot evidence marks the supporting facts for ${example._id ?? `sample-${index + 1}`}.`,
        split_tag: "dev_distractor",
        created_at: params.generatedAt,
      } satisfies RawDocsQaExampleV1;
    }),
  };

  const compilation = compileColdStartDocsQaSourceBundleV1({
    repoRoot,
    bundle: rawBundle,
  });

  const hotpotCompiledHash = compilation.registryEntry.file_hashes[snapshotFile]?.replace(/^sha256:/, "");
  if (hotpotCompiledHash !== params.card.registry_entry.file_hashes[snapshotFile]) {
    throw new Error(`HotpotQA snapshot hash mismatch for ${snapshotFile}`);
  }

  return {
    datasetFamily: "hotpotqa",
    datasetId: params.card.registry_entry.dataset_id,
    snapshotRef: params.card.registry_entry.immutable_snapshot_ref,
    sampleStrategy: `first-${params.sampleCount}-examples-from-hotpotqa-dev-distractor-snapshot`,
    sourceDocuments,
    rawBundle,
    compilation,
  };
}

function buildMuSiQueRawBundle(params: {
  workspaceRoot: string;
  card: ColdStartSourceIntakeCardV1;
  generatedAt: string;
  sampleCount: number;
}): ColdStartQaSnapshotDatasetResultV1 {
  const snapshotFile = "openclawbrain/data/cold-start/snapshots/musique_v1/stonybrooknlp__musique__922ac98f19a2/data/musique_full_v1.0_dev.jsonl";
  const manifestFile = "openclawbrain/data/cold-start/snapshots/musique_v1/stonybrooknlp__musique__922ac98f19a2/manifest.json";
  const sumsFile = "openclawbrain/data/cold-start/snapshots/musique_v1/stonybrooknlp__musique__922ac98f19a2/SHA256SUMS";
  const { repoRoot, filePath: sourceFilePath } = resolveSnapshotFilePath(params.workspaceRoot, snapshotFile);
  const sampledExamples = sampleFirstJsonlObjects<{
    id: string;
    question: string;
    answer?: string;
    answerable?: boolean;
    paragraphs: Array<{
      idx: number;
      title: string;
      paragraph_text: string;
      is_supporting: boolean;
    }>;
    question_decomposition?: unknown;
    }>(sourceFilePath, Math.max(50, params.sampleCount * 100));

  const examples = sampledExamples.filter((candidate) => candidate.paragraphs.some((paragraph) => paragraph.is_supporting)).slice(0, params.sampleCount);
  if (examples.length === 0) {
    throw new Error(`no supporting MuSiQue examples could be sampled from ${snapshotFile}`);
  }

  if (examples.length < params.sampleCount) {
    throw new Error(`only ${examples.length} supporting MuSiQue examples could be sampled from ${snapshotFile}; requested ${params.sampleCount}`);
  }

  const sourceDocuments: RawDocsQaSourceDocumentV1[] = [
    {
      source_ref: snapshotFile,
      path: snapshotFile,
      kind: "qa_source",
    },
  ];

  const rawBundle: RawDocsQaSourceBundleV1 = {
    contract: "cold_start_docs_qa_source_bundle.v1",
    bundle_id: "musique-dev-export-candidate-v1",
    generated_at: params.generatedAt,
    registry: {
      dataset_id: params.card.registry_entry.dataset_id,
      source_family: "qa",
      upstream_url: params.card.registry_entry.upstream_url,
      original_creator: params.card.registry_entry.original_creator,
      license: params.card.registry_entry.license,
      commercial_use_status: params.card.registry_entry.commercial_use_status,
      redistribution_status: params.card.registry_entry.redistribution_status,
      pii_risk: params.card.registry_entry.pii_risk,
      benchmark_split_status: params.card.registry_entry.benchmark_split_status,
      approval_status: params.card.registry_entry.approval_status,
      reviewer: params.card.registry_entry.reviewer,
      immutable_snapshot_ref: params.card.registry_entry.immutable_snapshot_ref,
      exact_files: [snapshotFile, manifestFile, sumsFile],
      allowed_uses: params.card.registry_entry.allowed_uses,
      disallowed_uses: params.card.registry_entry.disallowed_uses,
      notes: [
        ...params.card.registry_entry.notes,
        "export candidate samples the frozen dev JSONL only and keeps the rest of the snapshot bundle out of the candidate payload",
      ],
      ...(params.card.registry_entry.created_at ? { created_at: params.card.registry_entry.created_at } : {}),
      ...(params.card.registry_entry.updated_at ? { updated_at: params.card.registry_entry.updated_at } : {}),
    },
    source_documents: sourceDocuments,
    examples: examples.map((example, index) => {
      const supportingParagraphs = example.paragraphs.filter((paragraph) => paragraph.is_supporting);
      const targetTitles = supportingParagraphs.map((paragraph) => normalizeWhitespace(paragraph.title));
      const stopLabel: RawDocsQaExampleV1["stop_label"] = supportingParagraphs.length === 1 ? "STOP_LOCAL" : "CONTINUE";

      return {
        row_id: `musique-dev-export-candidate-${index + 1}`,
        query: normalizeWhitespace(example.question),
        cursor_path: [snapshotFile, example.id],
        candidate_set: example.paragraphs.map((paragraph, candidateIndex) => ({
          candidate_id: buildCandidateId(params.card.registry_entry.dataset_id, paragraph.title),
          candidate_type: "doc_chunk" as const,
          authority: paragraph.is_supporting ? "snapshot_supporting_fact" : "snapshot_context",
          freshness: params.card.registry_entry.benchmark_split_status,
          token_cost: 72 + candidateIndex,
          score_hint: paragraph.is_supporting ? 1 : 0.15,
        })),
        teacher_action: {
          kind: "traverse",
          target_ids: targetTitles.map((title) => buildCandidateId(params.card.registry_entry.dataset_id, title)),
        },
        stop_label: stopLabel,
        evidence_spans: supportingParagraphs.map((paragraph) => ({ source_ref: snapshotFile, excerpt: normalizeWhitespace(paragraph.paragraph_text) })),
        hard_negatives: [],
        outcome_gain: 1,
        teacher_confidence: supportingParagraphs.length === 1 ? 0.98 : 0.96,
        rationale: `MuSiQue snapshot evidence marks the supporting paragraphs for ${example.id}.`,
        split_tag: "dev",
        created_at: params.generatedAt,
      } satisfies RawDocsQaExampleV1;
    }),
  };

  const compilation = compileColdStartDocsQaSourceBundleV1({
    repoRoot,
    bundle: rawBundle,
  });

  const musiqueCompiledHash = compilation.registryEntry.file_hashes[snapshotFile]?.replace(/^sha256:/, "");
  if (musiqueCompiledHash !== params.card.registry_entry.file_hashes[snapshotFile]) {
    throw new Error(`MuSiQue snapshot hash mismatch for ${snapshotFile}`);
  }

  return {
    datasetFamily: "musique",
    datasetId: params.card.registry_entry.dataset_id,
    snapshotRef: params.card.registry_entry.immutable_snapshot_ref,
    sampleStrategy: `first-${params.sampleCount}-supporting-examples-from-musique-dev-snapshot-stoplocal-aware`,
    sourceDocuments,
    rawBundle,
    compilation,
  };
}

export function buildColdStartQaSnapshotExportCandidateV1(params: {
  registryPath: string;
  workspaceRoot: string;
  generatedAt?: string;
  sampleCount?: number;
  hotpotSampleCount?: number;
  musiqueSampleCount?: number;
}): ColdStartQaSnapshotExportCandidateV1 {
  const generatedAt = params.generatedAt ?? new Date().toISOString();
  const sampleCount = params.sampleCount ?? 1;
  const hotpotSampleCount = params.hotpotSampleCount ?? sampleCount;
  const musiqueSampleCount = params.musiqueSampleCount ?? sampleCount;
  const hotpotCard = getRegistryCard(params.registryPath, "hotpotqa_v1");
  const musiqueCard = getRegistryCard(params.registryPath, "musique_v1");

  if (!hotpotCard.materialized_snapshot || !musiqueCard.materialized_snapshot) {
    throw new Error("snapshot-backed export candidate requires materialized HotpotQA and MuSiQue snapshots");
  }

  const hotpotqa = buildHotpotQaRawBundle({
    workspaceRoot: params.workspaceRoot,
    card: hotpotCard,
    generatedAt,
    sampleCount: hotpotSampleCount,
  });
  const musique = buildMuSiQueRawBundle({
    workspaceRoot: params.workspaceRoot,
    card: musiqueCard,
    generatedAt,
    sampleCount: musiqueSampleCount,
  });

  const registryEntries = [hotpotqa.compilation.registryEntry, musique.compilation.registryEntry];
  const routeRows = [...hotpotqa.compilation.routeRows, ...musique.compilation.routeRows];

  for (const entry of registryEntries) {
    const validation = validateDataRegistryEntryV1(entry);
    if (!validation.valid) {
      throw new Error(`compiled registry entry failed validation (${entry.dataset_id}): ${validation.issues.join("; ")}`);
    }
  }
  for (const row of routeRows) {
    const validation = validateRouteDecisionRowV1(row);
    if (!validation.valid) {
      throw new Error(`compiled route row failed validation (${row.row_id}): ${validation.issues.join("; ")}`);
    }
  }

  return {
    contract: COLD_START_QA_SNAPSHOT_EXPORT_CANDIDATE_CONTRACT_V1,
    export_id: "cold-start-qa-snapshot-export-candidate-v1",
    generated_at: generatedAt,
    review_status: "under_review",
    source_registry_path: params.registryPath,
    workspace_root: params.workspaceRoot,
    datasets: [hotpotqa, musique],
    registry_entries: registryEntries,
    route_rows: routeRows,
    notes: [
      "This export candidate is snapshot-backed and intentionally under review only.",
      "It compiles real HotpotQA and MuSiQue snapshot bytes into governed route supervision, but it does not claim approved_train.",
      "Snapshot integrity is checked against the frozen registry hashes before materialization.",
    ],
  };
}

export function summarizeColdStartQaSnapshotExportCandidateV1(
  candidate: ColdStartQaSnapshotExportCandidateV1,
): ColdStartQaSnapshotExportCandidateSummaryV1 {
  return {
    exportId: candidate.export_id,
    generatedAt: candidate.generated_at,
    reviewStatus: candidate.review_status,
    datasetCount: candidate.datasets.length,
    routeRowCount: candidate.route_rows.length,
    registryEntryCount: candidate.registry_entries.length,
    datasetIds: candidate.datasets.map((dataset) => dataset.datasetId),
    snapshotRefs: candidate.datasets.map((dataset) => dataset.snapshotRef),
    sampleStrategies: candidate.datasets.map((dataset) => dataset.sampleStrategy),
  };
}

export function loadColdStartQaSnapshotExportCandidateV1(filePath: string): ColdStartQaSnapshotExportCandidateV1 {
  return JSON.parse(readFileSync(filePath, "utf8")) as ColdStartQaSnapshotExportCandidateV1;
}

export function writeColdStartQaSnapshotExportCandidateV1(filePath: string, candidate: ColdStartQaSnapshotExportCandidateV1): void {
  writeFileSync(filePath, `${JSON.stringify(candidate, null, 2)}\n`, "utf8");
}
