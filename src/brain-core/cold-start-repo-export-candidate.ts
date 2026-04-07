import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type {
  DataRegistryEntryV1,
  GraphCompilerContractV1,
} from "./cold-start-router-contracts.js";
import {
  validateDataRegistryEntryV1,
  validateGraphCompilerContractV1,
} from "./cold-start-router-contracts.js";

export const COLD_START_REPO_EXPORT_CANDIDATE_CONTRACT_V1 =
  "cold_start_repo_export_candidate.v1" as const;

export const COLD_START_REPO_EXPORT_CANDIDATE_BRIDGE_STATUS_V1 = [
  "pending_row_extraction",
] as const;

export type ColdStartRepoExportCandidateBridgeStatusV1 =
  (typeof COLD_START_REPO_EXPORT_CANDIDATE_BRIDGE_STATUS_V1)[number];

export interface RepoSnapshotManifestFileV1 {
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface RepoSnapshotManifestV1 {
  schema_version: number;
  dataset_id: string;
  source_family: "repo";
  upstream_url: string;
  hf_repo_id: string;
  hf_revision: string;
  materialized_at: string;
  local_dir: string;
  file_count: number;
  files: RepoSnapshotManifestFileV1[];
  notes: string[];
}

export interface RepoSnapshotIndexEntryV1 {
  dataset_id: string;
  repo_id: string;
  revision: string;
  local_dir: string;
  manifest_path: string;
  manifest_sha256: string;
  files: RepoSnapshotManifestFileV1[];
}

export interface RepoSnapshotIndexV1 {
  schema_version: number;
  generated_at: string;
  snapshots: RepoSnapshotIndexEntryV1[];
}

export interface ColdStartRepoExportCandidateRouteSeedV1 {
  seed_id: string;
  dataset_id: string;
  split_name: string;
  file_path: string;
  file_sha256: string;
  size_bytes: number;
  snapshot_ref: string;
  manifest_path: string;
  bridge_status: ColdStartRepoExportCandidateBridgeStatusV1;
  route_supervision_hint: string;
}

export interface ColdStartRepoExportCandidateReportV1 {
  contract: typeof COLD_START_REPO_EXPORT_CANDIDATE_CONTRACT_V1;
  candidate_id: string;
  generated_at: string;
  repo_root: string;
  snapshot_index_path: string;
  snapshot_index_sha256: string;
  snapshot_count: number;
  manifest_count: number;
  registry_entry_count: number;
  exact_snapshot_file_count: number;
  data_file_count: number;
  materialized_byte_count: number;
  route_seed_count: number;
  materialized_snapshot_count: number;
  rights_caveat_count: number;
  approved_train_claim: false;
  route_row_count: 0;
  snapshot_backed: true;
  bridge_state: "snapshot-backed";
  caveats: string[];
  dataset_ids: string[];
}

export interface ColdStartRepoExportCandidateBundleV1 {
  contract: typeof COLD_START_REPO_EXPORT_CANDIDATE_CONTRACT_V1;
  candidate_id: string;
  generated_at: string;
  snapshot_index: RepoSnapshotIndexV1;
  registry_entries: DataRegistryEntryV1[];
  graph_compiler: GraphCompilerContractV1;
  route_seeds: ColdStartRepoExportCandidateRouteSeedV1[];
  report: ColdStartRepoExportCandidateReportV1;
}

export interface ColdStartRepoExportCandidateSummaryV1 {
  candidateId: string;
  generatedAt: string;
  snapshotCount: number;
  manifestCount: number;
  registryEntryCount: number;
  routeSeedCount: number;
  snapshotBacked: boolean;
  bridgeState: string;
  datasetIds: string[];
  dataFileCount: number;
  materializedByteCount: number;
  rightsCaveatCount: number;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_OPENCLAWBRAIN_REPO_ROOT = path.resolve(__dirname, "..", "..");

function sha256Text(text: string): string {
  return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

function stripSha256Prefix(value: string): string {
  return value.startsWith("sha256:") ? value.slice("sha256:".length) : value;
}

function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function candidateRepoRoots(repoRoot: string): string[] {
  const resolvedRepoRoot = path.resolve(repoRoot);
  return [...new Set([
    resolvedRepoRoot,
    path.resolve(resolvedRepoRoot, "openclawbrain"),
    path.resolve(resolvedRepoRoot, "..", "openclawbrain"),
    path.resolve(resolvedRepoRoot, "..", "..", "openclawbrain"),
    DEFAULT_OPENCLAWBRAIN_REPO_ROOT,
    path.resolve(DEFAULT_OPENCLAWBRAIN_REPO_ROOT, "..", "openclawbrain"),
  ])];
}

function resolveSnapshotIndexPath(repoRoot: string): { repoRoot: string; snapshotIndexPath: string } {
  for (const candidateRoot of candidateRepoRoots(repoRoot)) {
    const candidatePath = path.resolve(candidateRoot, "data/cold-start/snapshots/index.json");
    if (existsSync(candidatePath)) {
      return {
        repoRoot: candidateRoot,
        snapshotIndexPath: candidatePath,
      };
    }
  }

  return {
    repoRoot: DEFAULT_OPENCLAWBRAIN_REPO_ROOT,
    snapshotIndexPath: path.resolve(DEFAULT_OPENCLAWBRAIN_REPO_ROOT, "data/cold-start/snapshots/index.json"),
  };
}

function snapshotIndexPathFromRepoRoot(repoRoot: string): string {
  return resolveSnapshotIndexPath(repoRoot).snapshotIndexPath;
}

function snapshotRefFromDataset(datasetId: string, manifestSha256: string): string {
  return `snapshot:${datasetId}@${manifestSha256}`;
}

function splitNameFromFilePath(filePath: string): string {
  const baseName = path.basename(filePath);
  const prefixMatch = baseName.match(/^(.+?)-\d{5}-of-\d{5}/);
  if (prefixMatch?.[1]) {
    return prefixMatch[1];
  }
  return baseName.replace(/\.[^.]+$/, "");
}

function isRepoBenchmarkDatasetId(datasetId: string): boolean {
  return datasetId === "swe_bench_v1" || datasetId.startsWith("repo_bench_");
}

function routeSupervisionHint(datasetId: string): string {
  if (datasetId === "swe_bench_v1") {
    return "SWE-bench bridge: evaluation split bytes are frozen, but row-level route supervision still needs parquet extraction and rights review before any export claim.";
  }
  return "RepoBench bridge: benchmark shard bytes are frozen, but row-level route supervision still needs parquet extraction and rights review before any export claim.";
}

function datasetCaveat(datasetId: string): string {
  if (datasetId === "swe_bench_v1") {
    return "SWE-bench upstream tasks trace back to real GitHub issues/repos, so license mixing and holdout/export rights remain under review.";
  }
  return "RepoBench v1.1 is the current public release family, not the paper-era archive/v0 branch; rights approval for ingest/export remains pending.";
}

function buildRegistryEntry(params: {
  manifest: RepoSnapshotManifestV1;
  manifestPath: string;
  manifestSha256: string;
  generatedAt: string;
}): DataRegistryEntryV1 {
  const exactFiles = params.manifest.files.map((file) => file.path);
  const fileHashes = Object.fromEntries(
    params.manifest.files.map((file) => [file.path, file.sha256]),
  );

  return {
    dataset_id: params.manifest.dataset_id,
    source_family: "repo",
    upstream_url: params.manifest.upstream_url,
    original_creator: params.manifest.hf_repo_id,
    license: "under_review: governed snapshot of a public repo benchmark release family",
    commercial_use_status: "unknown",
    redistribution_status: "unknown",
    pii_risk: "unknown",
    benchmark_split_status: params.manifest.dataset_id === "swe_bench_v1" ? "holdout" : "eval_only",
    approval_status: "under_review",
    reviewer: "lane-r2-repo-snapshot-compiler",
    immutable_snapshot_ref: snapshotRefFromDataset(params.manifest.dataset_id, params.manifestSha256),
    exact_files: exactFiles,
    file_hashes: fileHashes,
    allowed_uses: [
      "route supervision bridge",
      "snapshot provenance",
      "rights review",
    ],
    disallowed_uses: [
      "approved_train",
      "unreviewed redistribution",
      "claiming row-level export before parquet extraction",
    ],
    notes: [
      `Materialized from local governed snapshot bytes at ${params.manifest.local_dir}`,
      `Manifest file: ${params.manifestPath}`,
      datasetCaveat(params.manifest.dataset_id),
      `Manifest sha256: ${params.manifestSha256}`,
    ],
    created_at: params.generatedAt,
    updated_at: params.generatedAt,
  };
}

function buildGraphCompilerContract(params: {
  candidateId: string;
  snapshotIndex: RepoSnapshotIndexV1;
  registryEntries: DataRegistryEntryV1[];
  routeSeeds: ColdStartRepoExportCandidateRouteSeedV1[];
}): GraphCompilerContractV1 {
  const inputSnapshotRef = `snapshot:repo-cold-start-index@${sha256Text(JSON.stringify({
    generatedAt: params.snapshotIndex.generated_at,
    datasetIds: params.snapshotIndex.snapshots.map((snapshot) => snapshot.dataset_id).sort(),
    manifestShas: params.snapshotIndex.snapshots.map((snapshot) => snapshot.manifest_sha256).sort(),
  }))}`;

  const graphRef = `graph:repo-snapshot-candidate@${sha256Text(JSON.stringify({
    candidateId: params.candidateId,
    datasetIds: params.registryEntries.map((entry) => entry.dataset_id).sort(),
    routeSeedIds: params.routeSeeds.map((seed) => seed.seed_id).sort(),
    snapshotIndexSha: sha256Text(JSON.stringify(params.snapshotIndex)),
  }))}`;

  const artifactRef = `artifact:repo-snapshot-neighborhood-pack@${sha256Text(JSON.stringify({
    candidateId: params.candidateId,
    graphRef,
    routeSeedCount: params.routeSeeds.length,
  }))}`;

  return {
    compiler_id: "repo-snapshot-cold-start-compiler-v1",
    source_family: "repo",
    input_snapshot_ref: inputSnapshotRef,
    node_schema: [
      {
        node_kind: "repo_file",
        required_fields: ["dataset_id", "path", "sha256", "snapshot_ref"],
        optional_fields: ["size_bytes", "split_name"],
        notes: ["Frozen repo bytes become provenance-bearing file nodes before row extraction."],
      },
      {
        node_kind: "benchmark_split",
        required_fields: ["dataset_id", "split_name", "manifest_path"],
        optional_fields: ["hint", "row_count"],
        notes: ["Benchmark split nodes mark the path from immutable snapshot files toward route supervision."],
      },
      {
        node_kind: "route_seed",
        required_fields: ["seed_id", "dataset_id", "file_path"],
        optional_fields: ["route_supervision_hint", "bridge_status"],
        notes: ["Route seed nodes intentionally stop short of pretending the parquet rows have already been extracted."],
      },
    ],
    edge_schema: [
      {
        edge_kind: "contains",
        required_fields: ["dataset_id", "path"],
        optional_fields: ["sha256"],
        notes: ["Snapshot manifests contain exact files."],
      },
      {
        edge_kind: "bridges_to",
        required_fields: ["seed_id", "snapshot_ref"],
        optional_fields: ["manifest_path"],
        notes: ["Bridge edges point from frozen snapshot bytes toward the later row compiler."],
      },
      {
        edge_kind: "supervises",
        required_fields: ["dataset_id", "split_name"],
        optional_fields: ["route_supervision_hint"],
        notes: ["Split-level supervision is still pending parquet extraction, so the compiler only marks the path."],
      },
    ],
    provenance_rules: [
      "real snapshot bytes only",
      "exact file hashes must match the local manifest",
      "approval remains under review; do not mark approved_train",
      "row-level route export stays pending until parquet extraction and rights review complete",
      "benchmark split and rights caveats must remain visible in every downstream artifact",
    ],
    output_neighborhood_pack: {
      pack_id: `repo-snapshot-neighborhood-pack:${params.candidateId}`,
      artifact_ref: artifactRef,
      graph_ref: graphRef,
      radius_hops: 1,
      frontier_limit: 64,
    },
    compiler_version: "repo-snapshot-cold-start-compiler@0.1.0",
  };
}

function buildRouteSeeds(params: {
  snapshotIndex: RepoSnapshotIndexV1;
}): ColdStartRepoExportCandidateRouteSeedV1[] {
  const routeSeeds: ColdStartRepoExportCandidateRouteSeedV1[] = [];
  for (const snapshot of params.snapshotIndex.snapshots) {
    const snapshotRef = snapshotRefFromDataset(snapshot.dataset_id, snapshot.manifest_sha256);
    for (const file of snapshot.files) {
      if (!file.path.startsWith("data/")) {
        continue;
      }
      const splitName = splitNameFromFilePath(file.path);
      routeSeeds.push({
        seed_id: `${snapshot.dataset_id}:${splitName}:${routeSeeds.length + 1}`,
        dataset_id: snapshot.dataset_id,
        split_name: splitName,
        file_path: file.path,
        file_sha256: file.sha256,
        size_bytes: file.size_bytes,
        snapshot_ref: snapshotRef,
        manifest_path: snapshot.manifest_path,
        bridge_status: "pending_row_extraction",
        route_supervision_hint: routeSupervisionHint(snapshot.dataset_id),
      });
    }
  }
  return routeSeeds;
}

export function compileColdStartRepoSnapshotExportCandidateV1(params: {
  repoRoot?: string;
  candidateId?: string;
  generatedAt?: string;
} = {}): ColdStartRepoExportCandidateBundleV1 {
  const requestedRepoRoot = path.resolve(params.repoRoot ?? process.cwd());
  const { repoRoot, snapshotIndexPath } = resolveSnapshotIndexPath(requestedRepoRoot);
  const snapshotIndex = readJsonFile<RepoSnapshotIndexV1>(snapshotIndexPath);
  const repoSnapshots = snapshotIndex.snapshots.filter((snapshot) => isRepoBenchmarkDatasetId(snapshot.dataset_id));
  const repoSnapshotIndex: RepoSnapshotIndexV1 = {
    ...snapshotIndex,
    snapshots: repoSnapshots,
  };
  const candidateId = normalizeWhitespace(
    params.candidateId ?? "cold-start-repo-snapshot-export-candidate-v1",
  );
  const generatedAt = normalizeWhitespace(
    params.generatedAt ?? snapshotIndex.generated_at,
  );
  const snapshotIndexSha256 = sha256Text(readFileSync(snapshotIndexPath, "utf8"));

  const registryEntries: DataRegistryEntryV1[] = [];
  const routeSeeds = buildRouteSeeds({ snapshotIndex: repoSnapshotIndex });
  let manifestCount = 0;
  let materializedByteCount = 0;
  let exactSnapshotFileCount = 0;

  for (const snapshotEntry of repoSnapshots) {
    const manifestJson = readFileSync(snapshotEntry.manifest_path, "utf8");
    const manifestSha256 = sha256Text(manifestJson);
    if (stripSha256Prefix(manifestSha256) !== snapshotEntry.manifest_sha256) {
      throw new Error(
        `manifest hash mismatch for ${snapshotEntry.dataset_id}: expected ${snapshotEntry.manifest_sha256}, got ${manifestSha256}`,
      );
    }

    const manifest = JSON.parse(manifestJson) as RepoSnapshotManifestV1;
    if (manifest.dataset_id !== snapshotEntry.dataset_id) {
      throw new Error(
        `manifest dataset_id mismatch for ${snapshotEntry.dataset_id}: got ${manifest.dataset_id}`,
      );
    }

    if (manifest.file_count !== manifest.files.length) {
      throw new Error(
        `manifest file_count mismatch for ${snapshotEntry.dataset_id}: expected ${manifest.file_count}, got ${manifest.files.length}`,
      );
    }

    const registryEntry = buildRegistryEntry({
      manifest,
      manifestPath: snapshotEntry.manifest_path,
      manifestSha256,
      generatedAt,
    });

    const validation = validateDataRegistryEntryV1(registryEntry);
    if (!validation.valid) {
      throw new Error(
        `compiled registry entry failed validation for ${registryEntry.dataset_id}: ${validation.issues.join(" | ")}`,
      );
    }

    registryEntries.push(registryEntry);
    manifestCount += 1;
    materializedByteCount += snapshotEntry.files.reduce((sum, file) => sum + file.size_bytes, 0);
    exactSnapshotFileCount += snapshotEntry.files.length;
  }

  const graphCompiler = buildGraphCompilerContract({
    candidateId,
    snapshotIndex,
    registryEntries,
    routeSeeds,
  });
  const graphValidation = validateGraphCompilerContractV1(graphCompiler);
  if (!graphValidation.valid) {
    throw new Error(`compiled graph compiler contract failed validation: ${graphValidation.issues.join(" | ")}`);
  }

  const report: ColdStartRepoExportCandidateReportV1 = {
    contract: COLD_START_REPO_EXPORT_CANDIDATE_CONTRACT_V1,
    candidate_id: candidateId,
    generated_at: generatedAt,
    repo_root: repoRoot,
    snapshot_index_path: snapshotIndexPath,
    snapshot_index_sha256: snapshotIndexSha256,
    snapshot_count: repoSnapshots.length,
    manifest_count: manifestCount,
    registry_entry_count: registryEntries.length,
    exact_snapshot_file_count: exactSnapshotFileCount,
    data_file_count: routeSeeds.length,
    materialized_byte_count: materializedByteCount,
    route_seed_count: routeSeeds.length,
    materialized_snapshot_count: repoSnapshots.length,
    rights_caveat_count: repoSnapshots.length,
    approved_train_claim: false,
    route_row_count: 0,
    snapshot_backed: true,
    bridge_state: "snapshot-backed",
    caveats: [
      "Real snapshot bytes are frozen locally, but route-row extraction is still pending and no approved_train claim is made.",
      "SWE-bench rights and RepoBench rights remain under review; this is a bridge artifact, not an ingest approval.",
      "Route supervision rows would need parquet extraction and a separate rights/approval pass before training ingest.",
    ],
    dataset_ids: registryEntries.map((entry) => entry.dataset_id),
  };

  return {
    contract: COLD_START_REPO_EXPORT_CANDIDATE_CONTRACT_V1,
    candidate_id: candidateId,
    generated_at: generatedAt,
    snapshot_index: repoSnapshotIndex,
    registry_entries: registryEntries,
    graph_compiler: graphCompiler,
    route_seeds: routeSeeds,
    report,
  };
}

export function summarizeColdStartRepoSnapshotExportCandidateV1(
  bundle: ColdStartRepoExportCandidateBundleV1,
): ColdStartRepoExportCandidateSummaryV1 {
  return {
    candidateId: bundle.candidate_id,
    generatedAt: bundle.generated_at,
    snapshotCount: bundle.report.snapshot_count,
    manifestCount: bundle.report.manifest_count,
    registryEntryCount: bundle.report.registry_entry_count,
    routeSeedCount: bundle.report.route_seed_count,
    snapshotBacked: bundle.report.snapshot_backed,
    bridgeState: bundle.report.bridge_state,
    datasetIds: bundle.report.dataset_ids,
    dataFileCount: bundle.report.data_file_count,
    materializedByteCount: bundle.report.materialized_byte_count,
    rightsCaveatCount: bundle.report.rights_caveat_count,
  };
}

export function loadColdStartRepoSnapshotIndexV1(
  repoRoot?: string,
): RepoSnapshotIndexV1 {
  return readJsonFile<RepoSnapshotIndexV1>(
    snapshotIndexPathFromRepoRoot(path.resolve(repoRoot ?? process.cwd())),
  );
}
