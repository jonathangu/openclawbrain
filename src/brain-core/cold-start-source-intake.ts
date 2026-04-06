import { readFileSync } from "node:fs";

import { Type, type Static, type TSchema } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";

import {
  type ColdStartApprovalStatusV1,
  type ColdStartPiiRiskV1,
  type ColdStartSourceFamilyV1,
  type ContractValidationResultV1,
  DataRegistryEntrySchemaV1,
  validateDataRegistryEntryV1,
} from "./cold-start-router-contracts.js";

export const COLD_START_SOURCE_INTAKE_CONTRACT_VERSION_V1 = 1 as const;

export const COLD_START_SOURCE_INTAKE_REGISTRY_ID_V1 = "cold-start-source-intake-bootstrap-v1" as const;

export const COLD_START_SOURCE_INTAKE_STATES_V1 = ["candidate", "blocked", "ready"] as const;

export type ColdStartSourceIntakeStateV1 = (typeof COLD_START_SOURCE_INTAKE_STATES_V1)[number];

export const COLD_START_SOURCE_INTAKE_APPROVAL_PLACEHOLDER_RE_V1 = /\b(awaiting|pending|review|placeholder)\b/i;

export const ColdStartSourceIntakeCardSchemaV1 = Type.Object(
  {
    priority: Type.Integer({ minimum: 1 }),
    approval_placeholder: Type.String({ minLength: 1 }),
    registry_entry: DataRegistryEntrySchemaV1,
    readiness_notes: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    approval_checks: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    materialized_snapshot: Type.Boolean(),
  },
  { additionalProperties: false },
);

export type ColdStartSourceIntakeCardV1 = Static<typeof ColdStartSourceIntakeCardSchemaV1>;

export const ColdStartSourceIntakeApprovalPolicySchemaV1 = Type.Object(
  {
    default_state: Type.Literal("under_review"),
    rights_review_required: Type.Boolean(),
    manual_approval_required: Type.Boolean(),
    destructive_ingest_allowed: Type.Boolean(),
    snapshot_required: Type.Boolean(),
    placeholder_text: Type.String({ minLength: 1 }),
    review_checks: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
  },
  { additionalProperties: false },
);

export type ColdStartSourceIntakeApprovalPolicyV1 = Static<
  typeof ColdStartSourceIntakeApprovalPolicySchemaV1
>;

export const ColdStartSourceIntakeRegistrySchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(COLD_START_SOURCE_INTAKE_CONTRACT_VERSION_V1),
    registry_id: Type.Literal(COLD_START_SOURCE_INTAKE_REGISTRY_ID_V1),
    owner: Type.String({ minLength: 1 }),
    generated_at: Type.String({ minLength: 1 }),
    approval_policy: ColdStartSourceIntakeApprovalPolicySchemaV1,
    source_families: Type.Array(
      Type.Union([
        Type.Literal("docs"),
        Type.Literal("repo"),
        Type.Literal("memory"),
        Type.Literal("tools"),
        Type.Literal("qa"),
        Type.Literal("agent_traces"),
      ]),
      { minItems: 1 },
    ),
    intake_order: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    cards: Type.Array(ColdStartSourceIntakeCardSchemaV1, { minItems: 1 }),
    notes: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
  },
  { additionalProperties: false },
);

export type ColdStartSourceIntakeRegistryV1 = Static<typeof ColdStartSourceIntakeRegistrySchemaV1>;

export interface ColdStartSourceIntakeRegistrySummaryV1 {
  registryId: string;
  owner: string;
  entryCount: number;
  materializedSnapshotCount: number;
  datasetIds: string[];
  sourceFamilyCounts: Record<ColdStartSourceFamilyV1, number>;
  approvalStatusCounts: Record<ColdStartApprovalStatusV1, number>;
  piiRiskCounts: Record<ColdStartPiiRiskV1, number>;
  placeholderCount: number;
}

function toSet(values: readonly string[]): Set<string> {
  return new Set(values.map((value) => value.trim()).filter(Boolean));
}

function asCounts<T extends string>(keys: readonly T[]): Record<T, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<T, number>;
}

function hasExplicitPlaceholder(placeholder: string): boolean {
  return COLD_START_SOURCE_INTAKE_APPROVAL_PLACEHOLDER_RE_V1.test(placeholder);
}

export function validateColdStartSourceIntakeRegistryV1(
  value: unknown,
): ContractValidationResultV1 {
  const contract = "cold_start_source_intake_registry.v1";
  if (!value || typeof value !== "object") {
    return {
      contract,
      valid: false,
      issues: [`${contract}: value must be an object`],
    };
  }

  const base = validateSchema(contract, ColdStartSourceIntakeRegistrySchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const registry = value as ColdStartSourceIntakeRegistryV1;
  const issues: string[] = [];
  const datasetIds = registry.cards.map((card) => card.registry_entry.dataset_id);
  const uniqueDatasetIds = toSet(datasetIds);
  if (uniqueDatasetIds.size !== datasetIds.length) {
    issues.push(`${contract}/cards contains duplicate dataset_id values`);
  }

  const expectedOrder = [...datasetIds];
  const declaredOrder = [...registry.intake_order];
  if (expectedOrder.length !== declaredOrder.length || expectedOrder.some((datasetId, index) => datasetId !== declaredOrder[index])) {
    issues.push(`${contract}/intake_order must exactly match card dataset_id ordering`);
  }

  const uniquePriorities = toSet(registry.cards.map((card) => String(card.priority)));
  if (uniquePriorities.size !== registry.cards.length) {
    issues.push(`${contract}/cards contains duplicate priority values`);
  }

  const approvalStatusCounts = new Map<ColdStartApprovalStatusV1, number>();
  const piiRiskCounts = new Map<ColdStartPiiRiskV1, number>();

  for (const card of registry.cards) {
    const entryResult = validateDataRegistryEntryV1(card.registry_entry);
    if (!entryResult.valid) {
      issues.push(
        `${contract}/cards[${card.registry_entry.dataset_id}]/registry_entry: ${entryResult.issues.join("; ")}`,
      );
    }

    if (!hasExplicitPlaceholder(card.approval_placeholder)) {
      issues.push(
        `${contract}/cards[${card.registry_entry.dataset_id}]/approval_placeholder must clearly say pending/awaiting/review/placeholder`,
      );
    }

    approvalStatusCounts.set(
      card.registry_entry.approval_status,
      (approvalStatusCounts.get(card.registry_entry.approval_status) ?? 0) + 1,
    );
    piiRiskCounts.set(card.registry_entry.pii_risk, (piiRiskCounts.get(card.registry_entry.pii_risk) ?? 0) + 1);

    if (card.materialized_snapshot && card.registry_entry.immutable_snapshot_ref.startsWith("pending:")) {
      issues.push(
        `${contract}/cards[${card.registry_entry.dataset_id}]/materialized_snapshot cannot be true while immutable_snapshot_ref is pending`,
      );
    }
  }

  const missingFamilies = registry.cards
    .map((card) => card.registry_entry.source_family)
    .filter((family, index, array) => array.indexOf(family) === index)
    .filter((family) => !registry.source_families.includes(family));
  if (missingFamilies.length > 0) {
    issues.push(`${contract}/source_families missing used families: ${missingFamilies.join(", ")}`);
  }

  if (issues.length > 0) {
    return { contract, valid: false, issues: [...base.issues, ...issues] };
  }

  return base;
}

export function summarizeColdStartSourceIntakeRegistryV1(
  registry: ColdStartSourceIntakeRegistryV1,
): ColdStartSourceIntakeRegistrySummaryV1 {
  const sourceFamilyCounts = asCounts([
    "docs",
    "repo",
    "memory",
    "tools",
    "qa",
    "agent_traces",
  ]);
  const approvalStatusCounts = asCounts([
    "proposed",
    "under_review",
    "approved_train",
    "approved_eval_only",
    "rejected",
    "archived",
  ]);
  const piiRiskCounts = asCounts(["none", "low", "medium", "high", "unknown"]);

  let materializedSnapshotCount = 0;
  let placeholderCount = 0;

  for (const card of registry.cards) {
    sourceFamilyCounts[card.registry_entry.source_family] += 1;
    approvalStatusCounts[card.registry_entry.approval_status] += 1;
    piiRiskCounts[card.registry_entry.pii_risk] += 1;
    if (card.materialized_snapshot) {
      materializedSnapshotCount += 1;
    }
    if (hasExplicitPlaceholder(card.approval_placeholder)) {
      placeholderCount += 1;
    }
  }

  return {
    registryId: registry.registry_id,
    owner: registry.owner,
    entryCount: registry.cards.length,
    materializedSnapshotCount,
    datasetIds: registry.cards.map((card) => card.registry_entry.dataset_id),
    sourceFamilyCounts,
    approvalStatusCounts,
    piiRiskCounts,
    placeholderCount,
  };
}

export function loadColdStartSourceIntakeRegistryV1(filePath: string): ColdStartSourceIntakeRegistryV1 {
  return JSON.parse(readFileSync(filePath, "utf8")) as ColdStartSourceIntakeRegistryV1;
}

export function readAndValidateColdStartSourceIntakeRegistryV1(filePath: string): {
  registry: ColdStartSourceIntakeRegistryV1;
  validation: ContractValidationResultV1;
  summary: ColdStartSourceIntakeRegistrySummaryV1 | null;
} {
  const registry = loadColdStartSourceIntakeRegistryV1(filePath);
  const validation = validateColdStartSourceIntakeRegistryV1(registry);
  return {
    registry,
    validation,
    summary: validation.valid ? summarizeColdStartSourceIntakeRegistryV1(registry) : null,
  };
}

function validateSchema(contract: string, schema: unknown, value: unknown): ContractValidationResultV1 {
  if (typeof schema !== "object" || schema === null) {
    return { contract, valid: false, issues: [`${contract}: schema unavailable`] };
  }
  const typeSchema = schema as TSchema;
  // The schema is a TypeBox schema; route the actual checks through the typed validator.
  if (Value.Check(typeSchema, value)) {
    return { contract, valid: true, issues: [] };
  }

  return {
    contract,
    valid: false,
    issues: [...Value.Errors(typeSchema, value)].map((error) => {
      const path = error.path && error.path.length > 0 ? error.path : "/";
      return `${contract}${path}: ${error.message}`;
    }),
  };
}
