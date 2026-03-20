import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const ATTACHMENT_POLICY_DECLARATION_CONTRACT = "openclawbrain.attachment-policy-declaration.v1";
const ATTACHMENT_POLICY_DECLARATION_DIRNAME = "attachment-truth";
const ATTACHMENT_POLICY_DECLARATION_BASENAME = "policy-declaration.json";

function toErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function readRecord(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  return value;
}

function normalizeOptionalString(value) {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function normalizeIsoTimestamp(value, fieldName) {
  const normalized = normalizeOptionalString(value);

  if (normalized === null) {
    throw new Error(`${fieldName} is required`);
  }

  if (Number.isNaN(Date.parse(normalized))) {
    throw new Error(`${fieldName} must be an ISO timestamp`);
  }

  return new Date(normalized).toISOString();
}

function normalizeAttachmentPolicy(value, fieldName = "policy") {
  if (value === "undeclared" || value === "dedicated" || value === "shared") {
    return value;
  }

  throw new Error(`${fieldName} must be one of undeclared, dedicated, or shared`);
}

function normalizeOptionalAttachmentPolicy(value, fieldName) {
  if (value === null || value === undefined) {
    return null;
  }

  return normalizeAttachmentPolicy(value, fieldName);
}

function validateAttachmentPolicyDeclaration(activationRoot, value) {
  const record = readRecord(value);

  if (record === null) {
    throw new Error("attachment policy declaration must contain an object");
  }

  if (record.contract !== ATTACHMENT_POLICY_DECLARATION_CONTRACT) {
    throw new Error(`attachment policy declaration contract must be ${ATTACHMENT_POLICY_DECLARATION_CONTRACT}`);
  }

  if (typeof record.activationRoot !== "string" || record.activationRoot.trim().length === 0) {
    throw new Error("attachment policy declaration activationRoot must be a non-empty string");
  }

  const resolvedActivationRoot = path.resolve(record.activationRoot);
  if (resolvedActivationRoot !== activationRoot) {
    throw new Error(`attachment policy declaration activationRoot mismatch: expected ${activationRoot}, received ${resolvedActivationRoot}`);
  }

  return {
    contract: ATTACHMENT_POLICY_DECLARATION_CONTRACT,
    activationRoot,
    updatedAt: normalizeIsoTimestamp(record.updatedAt, "updatedAt"),
    policy: normalizeAttachmentPolicy(record.policy),
    source: normalizeOptionalString(record.source) ?? "unknown",
    openclawHome: normalizeOptionalString(record.openclawHome),
  };
}

export function resolveAttachmentPolicyDeclarationPath(activationRoot) {
  return path.resolve(activationRoot, ATTACHMENT_POLICY_DECLARATION_DIRNAME, ATTACHMENT_POLICY_DECLARATION_BASENAME);
}

export function loadAttachmentPolicyDeclaration(activationRoot) {
  const resolvedActivationRoot = path.resolve(activationRoot);
  const declarationPath = resolveAttachmentPolicyDeclarationPath(resolvedActivationRoot);

  if (!existsSync(declarationPath)) {
    return {
      path: declarationPath,
      declaration: null,
      error: null,
    };
  }

  try {
    return {
      path: declarationPath,
      declaration: validateAttachmentPolicyDeclaration(
        resolvedActivationRoot,
        JSON.parse(readFileSync(declarationPath, "utf8")),
      ),
      error: null,
    };
  } catch (error) {
    return {
      path: declarationPath,
      declaration: null,
      error: toErrorMessage(error),
    };
  }
}

export function writeAttachmentPolicyDeclaration(input) {
  const activationRoot = path.resolve(input.activationRoot);
  const declarationPath = resolveAttachmentPolicyDeclarationPath(activationRoot);
  const declaration = {
    contract: ATTACHMENT_POLICY_DECLARATION_CONTRACT,
    activationRoot,
    updatedAt: normalizeIsoTimestamp(input.updatedAt ?? new Date().toISOString(), "updatedAt"),
    policy: normalizeAttachmentPolicy(input.policy),
    source: normalizeOptionalString(input.source) ?? "cli",
    openclawHome: normalizeOptionalString(input.openclawHome),
  };

  mkdirSync(path.dirname(declarationPath), { recursive: true });
  writeFileSync(declarationPath, `${JSON.stringify(declaration, null, 2)}\n`, "utf8");

  return {
    path: declarationPath,
    declaration,
  };
}

export function resolveEffectiveAttachmentPolicyTruth(input) {
  const referenceCount =
    typeof input.referenceCount === "number" && Number.isInteger(input.referenceCount) && input.referenceCount >= 0
      ? input.referenceCount
      : 0;
  const discoverablePolicy = referenceCount > 1 ? "shared" : null;
  const statusPolicy = normalizeOptionalAttachmentPolicy(input.statusPolicy ?? null, "statusPolicy");
  const reportPolicy = normalizeOptionalAttachmentPolicy(input.reportPolicy ?? null, "reportPolicy");
  const declaredPolicy = normalizeOptionalAttachmentPolicy(input.declaredPolicy ?? null, "declaredPolicy");
  const effectivePolicy =
    discoverablePolicy ??
    (statusPolicy !== null && statusPolicy !== "undeclared"
      ? statusPolicy
      : reportPolicy !== null && reportPolicy !== "undeclared"
        ? reportPolicy
        : declaredPolicy);

  return {
    effectivePolicy,
    statusPolicy:
      effectivePolicy === null
        ? statusPolicy
        : discoverablePolicy !== null
          ? discoverablePolicy
          : statusPolicy === null || statusPolicy === "undeclared"
            ? effectivePolicy
            : statusPolicy,
    reportPolicy:
      effectivePolicy === null
        ? reportPolicy
        : discoverablePolicy !== null
          ? discoverablePolicy
          : reportPolicy === null || reportPolicy === "undeclared"
            ? effectivePolicy
            : reportPolicy,
  };
}
