import { readFileSync } from "node:fs";

import { Type, type Static, type TSchema } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";

import {
  type ContractValidationResultV1,
  DataRegistryEntrySchemaV1,
  type DataRegistryEntryV1,
  RouteDecisionRowSchemaV1,
  type RouteDecisionRowV1,
  validateDataRegistryEntryV1,
  validateRouteDecisionRowAgainstDataRegistryEntryV1,
  validateRouteDecisionRowV1,
} from "./cold-start-router-contracts.ts";

export const COLD_START_ROUTER_APPROVED_EXPORT_CONTRACT_V1 = "cold_start_router_approved_export.v1" as const;

export const ColdStartRouterApprovedExportSchemaV1 = Type.Object(
  {
    contract: Type.Literal(COLD_START_ROUTER_APPROVED_EXPORT_CONTRACT_V1),
    export_id: Type.String({ minLength: 1 }),
    generated_at: Type.String({ minLength: 1 }),
    registry_entries: Type.Array(DataRegistryEntrySchemaV1, { minItems: 1 }),
    route_rows: Type.Array(RouteDecisionRowSchemaV1, { minItems: 1 }),
    notes: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
  },
  { additionalProperties: false },
);

export type ColdStartRouterApprovedExportV1 = Static<typeof ColdStartRouterApprovedExportSchemaV1>;

export interface ColdStartRouterApprovedExportRegistrySkipV1 {
  datasetId: string;
  reason: string;
}

export interface ColdStartRouterApprovedExportRowSkipV1 {
  rowId: string;
  datasetId: string;
  reason: string;
}

export interface ColdStartRouterApprovedExportSummaryV1 {
  exportId: string;
  generatedAt: string;
  rawRegistryEntryCount: number;
  approvedRegistryEntryCount: number;
  rawRowCount: number;
  approvedRowCount: number;
  skippedRegistryEntryCount: number;
  skippedRowCount: number;
  approvedDatasetIds: string[];
}

export interface ColdStartRouterApprovedExportLoadResultV1 {
  bundle: ColdStartRouterApprovedExportV1;
  registryEntries: DataRegistryEntryV1[];
  routeRows: RouteDecisionRowV1[];
  skippedRegistryEntries: ColdStartRouterApprovedExportRegistrySkipV1[];
  skippedRows: ColdStartRouterApprovedExportRowSkipV1[];
  summary: ColdStartRouterApprovedExportSummaryV1;
}

function validateSchema(contract: string, schema: TSchema, value: unknown): ContractValidationResultV1 {
  if (Value.Check(schema, value)) {
    return { contract, valid: true, issues: [] };
  }

  return {
    contract,
    valid: false,
    issues: [...Value.Errors(schema, value)].map((error) => {
      const path = error.path && error.path.length > 0 ? error.path : "/";
      return `${contract}${path}: ${error.message}`;
    }),
  };
}

function normalizeText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeStringArray(values: readonly string[] | undefined): string[] {
  if (!Array.isArray(values)) {
    return [];
  }
  return [...new Set(values.map((value) => normalizeText(value)).filter((value) => value.length > 0))].sort();
}

function isPendingSnapshotRef(snapshotRef: string): boolean {
  return snapshotRef.startsWith("pending:") || snapshotRef.startsWith("pending://");
}

function isApprovedRegistryEntry(entry: DataRegistryEntryV1): boolean {
  return entry.approval_status === "approved_train"
    && entry.commercial_use_status === "allowed"
    && entry.redistribution_status === "allowed"
    && (entry.pii_risk === "none" || entry.pii_risk === "low")
    && !isPendingSnapshotRef(entry.immutable_snapshot_ref);
}

function validateExportBundle(bundle: unknown): ColdStartRouterApprovedExportV1 {
  const validation = validateSchema("cold_start_router_approved_export.v1", ColdStartRouterApprovedExportSchemaV1, bundle);
  if (!validation.valid) {
    throw new Error(`invalid approved export bundle: ${validation.issues.join("; ")}`);
  }
  return bundle as ColdStartRouterApprovedExportV1;
}

export function loadColdStartRouterApprovedExportV1(filePath: string): ColdStartRouterApprovedExportV1 {
  return validateExportBundle(JSON.parse(readFileSync(filePath, "utf8")) as unknown);
}

export function reviewColdStartRouterApprovedTrainingSetV1(params: {
  exportId: string;
  generatedAt: string;
  registryEntries: DataRegistryEntryV1[];
  routeRows: RouteDecisionRowV1[];
}): Omit<ColdStartRouterApprovedExportLoadResultV1, "bundle"> {
  const skippedRegistryEntries: ColdStartRouterApprovedExportRegistrySkipV1[] = [];
  const registryEntries: DataRegistryEntryV1[] = [];
  const eligibleRegistryByDataset = new Map<string, DataRegistryEntryV1>();

  for (const registryEntry of params.registryEntries) {
    const validation = validateDataRegistryEntryV1(registryEntry);
    if (!validation.valid) {
      throw new Error(`invalid registry entry ${registryEntry.dataset_id}: ${validation.issues.join("; ")}`);
    }

    if (!isApprovedRegistryEntry(registryEntry)) {
      skippedRegistryEntries.push({
        datasetId: registryEntry.dataset_id,
        reason: `registry entry not eligible for training (${registryEntry.approval_status}, ${registryEntry.commercial_use_status}, ${registryEntry.redistribution_status}, ${registryEntry.pii_risk}, snapshot=${registryEntry.immutable_snapshot_ref})`,
      });
      continue;
    }

    registryEntries.push(registryEntry);
    eligibleRegistryByDataset.set(registryEntry.dataset_id, registryEntry);
  }

  const skippedRows: ColdStartRouterApprovedExportRowSkipV1[] = [];
  const routeRows: RouteDecisionRowV1[] = [];
  for (const row of params.routeRows) {
    const validation = validateRouteDecisionRowV1(row);
    if (!validation.valid) {
      throw new Error(`invalid route row ${row.row_id}: ${validation.issues.join("; ")}`);
    }

    const provenanceDataset = normalizeText(row.provenance.dataset);
    if (provenanceDataset.length === 0 || provenanceDataset !== row.dataset_id) {
      skippedRows.push({
        rowId: row.row_id,
        datasetId: row.dataset_id,
        reason: `row provenance dataset mismatch (${row.provenance.dataset})`,
      });
      continue;
    }

    if (row.provenance.review_status !== "approved_train") {
      skippedRows.push({
        rowId: row.row_id,
        datasetId: row.dataset_id,
        reason: `row review_status is ${row.provenance.review_status}; only approved_train rows are eligible`,
      });
      continue;
    }

    const registryEntry = eligibleRegistryByDataset.get(row.dataset_id);
    if (!registryEntry) {
      skippedRows.push({
        rowId: row.row_id,
        datasetId: row.dataset_id,
        reason: `dataset ${row.dataset_id} is not approved for training in the governed source-intake layer`,
      });
      continue;
    }

    const review = validateRouteDecisionRowAgainstDataRegistryEntryV1({ row, registryEntry });
    if (!review.valid) {
      skippedRows.push({
        rowId: row.row_id,
        datasetId: row.dataset_id,
        reason: `registry provenance review failed: ${review.issues.join("; ")}`,
      });
      continue;
    }

    routeRows.push(row);
  }

  const summary: ColdStartRouterApprovedExportSummaryV1 = {
    exportId: params.exportId,
    generatedAt: params.generatedAt,
    rawRegistryEntryCount: params.registryEntries.length,
    approvedRegistryEntryCount: registryEntries.length,
    rawRowCount: params.routeRows.length,
    approvedRowCount: routeRows.length,
    skippedRegistryEntryCount: skippedRegistryEntries.length,
    skippedRowCount: skippedRows.length,
    approvedDatasetIds: normalizeStringArray(registryEntries.map((entry) => entry.dataset_id)),
  };

  return {
    registryEntries,
    routeRows,
    skippedRegistryEntries,
    skippedRows,
    summary,
  };
}

export function loadAndFilterColdStartRouterApprovedExportV1(filePath: string): ColdStartRouterApprovedExportLoadResultV1 {
  const bundle = loadColdStartRouterApprovedExportV1(filePath);
  const reviewed = reviewColdStartRouterApprovedTrainingSetV1({
    exportId: bundle.export_id,
    generatedAt: bundle.generated_at,
    registryEntries: bundle.registry_entries,
    routeRows: bundle.route_rows,
  });

  return {
    bundle,
    ...reviewed,
  };
}

export function summarizeColdStartRouterApprovedExportV1(result: ColdStartRouterApprovedExportLoadResultV1): Record<string, unknown> {
  return {
    bundle: {
      exportId: result.bundle.export_id,
      generatedAt: result.bundle.generated_at,
      noteCount: result.bundle.notes.length,
    },
    summary: result.summary,
    registryEntries: result.registryEntries.map((entry) => ({
      datasetId: entry.dataset_id,
      approvalStatus: entry.approval_status,
      sourceFamily: entry.source_family,
    })),
    routeRows: result.routeRows.map((row) => ({
      rowId: row.row_id,
      datasetId: row.dataset_id,
      reviewStatus: row.provenance.review_status,
      stopLabel: row.stop_label,
    })),
    skippedRegistryEntries: result.skippedRegistryEntries,
    skippedRows: result.skippedRows,
  };
}
