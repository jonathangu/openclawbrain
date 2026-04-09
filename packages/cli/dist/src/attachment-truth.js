import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import path from "node:path";

const ATTACHMENT_RUNTIME_LOAD_PROOFS_CONTRACT = "openclaw_profile_runtime_load_proofs.v1";
const ATTACHMENT_TRUTH_DIRNAME = "attachment-truth";
const ATTACHMENT_RUNTIME_LOAD_PROOFS_BASENAME = "runtime-load-proofs.json";

function toErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function canonicalizeFilesystemPath(filePath) {
  const resolvedPath = path.resolve(filePath);

  try {
    return realpathSync(resolvedPath);
  } catch {
    return resolvedPath;
  }
}

function normalizeOptionalString(value) {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function formatPackageIdentity(packageName, packageVersion) {
  if (packageName !== null && packageVersion !== null) {
    return `${packageName}@${packageVersion}`;
  }

  return packageName ?? packageVersion ?? null;
}

function readPackageIdentityFromExtensionEntryPath(extensionEntryPath) {
  let currentDirectory = path.dirname(path.resolve(extensionEntryPath));

  while (true) {
    const packageJsonPath = path.join(currentDirectory, "package.json");
    if (existsSync(packageJsonPath)) {
      try {
        const parsed = readRecord(JSON.parse(readFileSync(packageJsonPath, "utf8")));
        const packageName = typeof parsed?.name === "string" && parsed.name.trim().length > 0 ? parsed.name.trim() : null;
        const packageVersion = typeof parsed?.version === "string" && parsed.version.trim().length > 0 ? parsed.version.trim() : null;

        return {
          packageJsonPath: canonicalizeFilesystemPath(packageJsonPath),
          packageName,
          packageVersion,
          packageIdentity: formatPackageIdentity(packageName, packageVersion)
        };
      } catch {
        return {
          packageJsonPath: canonicalizeFilesystemPath(packageJsonPath),
          packageName: null,
          packageVersion: null,
          packageIdentity: null
        };
      }
    }

    const parentDirectory = path.dirname(currentDirectory);
    if (parentDirectory === currentDirectory) {
      return {
        packageJsonPath: null,
        packageName: null,
        packageVersion: null,
        packageIdentity: null
      };
    }

    currentDirectory = parentDirectory;
  }
}

function normalizeIsoTimestamp(value, fieldName, fallbackValue) {
  const candidate = value ?? fallbackValue;

  if (candidate === undefined || candidate === null || candidate.trim().length === 0) {
    throw new Error(`${fieldName} is required`);
  }

  if (Number.isNaN(Date.parse(candidate))) {
    throw new Error(`${fieldName} must be an ISO timestamp`);
  }

  return new Date(candidate).toISOString();
}

function readRecord(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  return value;
}

function listConfiguredProfileIds(config) {
  const profiles = readRecord(config?.profiles);

  if (profiles === null) {
    return [];
  }

  return Object.keys(profiles)
    .map((profileId) => profileId.trim())
    .filter((profileId) => profileId.length > 0)
    .sort((left, right) => left.localeCompare(right));
}

function deriveDirectoryProfileId(openclawHome) {
  const basename = path.basename(openclawHome);

  if (!basename.startsWith(".openclaw-")) {
    return null;
  }

  const derived = basename.slice(".openclaw-".length).trim();
  return derived.length > 0 ? derived : null;
}

function inspectOpenClawHome(openclawHome) {
  const resolvedHome = path.resolve(openclawHome);
  const openclawJsonPath = path.join(resolvedHome, "openclaw.json");
  let config = null;

  if (existsSync(openclawJsonPath)) {
    try {
      config = readRecord(JSON.parse(readFileSync(openclawJsonPath, "utf8")));
    } catch {
      config = null;
    }
  }

  const configuredProfileIds = listConfiguredProfileIds(config);
  const directProfile = normalizeOptionalString(config?.profile);

  if (directProfile !== null) {
    return {
      profileId: directProfile,
      profileSource: "openclaw_json_profile",
    };
  }

  if (configuredProfileIds.length === 1) {
    return {
      profileId: configuredProfileIds[0],
      profileSource: "openclaw_json_single_profile_key",
    };
  }

  const directoryProfile = deriveDirectoryProfileId(resolvedHome);
  if (directoryProfile !== null) {
    return {
      profileId: directoryProfile,
      profileSource: "directory_name",
    };
  }

  return {
    profileId: null,
    profileSource: "none",
  };
}

function resolveOpenClawHomeFromExtensionEntryPath(extensionEntryPath) {
  let currentPath = path.resolve(extensionEntryPath);

  while (true) {
    const parentPath = path.dirname(currentPath);

    if (parentPath === currentPath) {
      break;
    }

    if (path.basename(parentPath) === "extensions") {
      return path.dirname(parentPath);
    }

    currentPath = parentPath;
  }

  return null;
}

function validateRuntimeLoadProofRecord(value, index) {
  const record = readRecord(value);

  if (record === null) {
    throw new Error(`profiles[${index}] must be an object`);
  }

  const openclawHome =
    typeof record.openclawHome === "string" && record.openclawHome.trim().length > 0
      ? canonicalizeFilesystemPath(record.openclawHome)
      : null;
  const profileId =
    record.profileId === null
      ? null
      : typeof record.profileId === "string" && record.profileId.trim().length > 0
        ? record.profileId.trim()
        : undefined;
  const profileSource =
    record.profileSource === "openclaw_json_profile" ||
    record.profileSource === "openclaw_json_single_profile_key" ||
    record.profileSource === "directory_name" ||
    record.profileSource === "none"
      ? record.profileSource
      : null;
  const extensionEntryPath =
    typeof record.extensionEntryPath === "string" && record.extensionEntryPath.trim().length > 0
      ? canonicalizeFilesystemPath(record.extensionEntryPath)
      : null;
  const loadedAt =
    typeof record.loadedAt === "string" && record.loadedAt.trim().length > 0
      ? normalizeIsoTimestamp(record.loadedAt, `profiles[${index}].loadedAt`)
      : null;
  const packageJsonPath =
    typeof record.packageJsonPath === "string" && record.packageJsonPath.trim().length > 0
      ? canonicalizeFilesystemPath(record.packageJsonPath)
      : null;
  const packageName =
    typeof record.packageName === "string" && record.packageName.trim().length > 0
      ? record.packageName.trim()
      : null;
  const packageVersion =
    typeof record.packageVersion === "string" && record.packageVersion.trim().length > 0
      ? record.packageVersion.trim()
      : null;
  const packageIdentity =
    typeof record.packageIdentity === "string" && record.packageIdentity.trim().length > 0
      ? record.packageIdentity.trim()
      : null;

  if (openclawHome === null) {
    throw new Error(`profiles[${index}].openclawHome must be a non-empty string`);
  }

  if (profileId === undefined) {
    throw new Error(`profiles[${index}].profileId must be null or a non-empty string`);
  }

  if (profileSource === null) {
    throw new Error(`profiles[${index}].profileSource must be a supported OpenClaw profile source`);
  }

  if (extensionEntryPath === null) {
    throw new Error(`profiles[${index}].extensionEntryPath must be a non-empty string`);
  }

  if (loadedAt === null) {
    throw new Error(`profiles[${index}].loadedAt must be a non-empty ISO timestamp`);
  }

  return {
    openclawHome,
    profileId,
    profileSource,
    extensionEntryPath,
    loadedAt,
    packageJsonPath,
    packageName,
    packageVersion,
    packageIdentity,
  };
}

function validateRuntimeLoadProofs(activationRoot, value) {
  const record = readRecord(value);

  if (record === null) {
    throw new Error("runtime load proof file must contain an object");
  }

  if (record.contract !== ATTACHMENT_RUNTIME_LOAD_PROOFS_CONTRACT) {
    throw new Error(`runtime load proof contract must be ${ATTACHMENT_RUNTIME_LOAD_PROOFS_CONTRACT}`);
  }

  if (record.runtimeOwner !== "openclaw") {
    throw new Error("runtime load proof runtimeOwner must be openclaw");
  }

  if (typeof record.activationRoot !== "string" || record.activationRoot.trim().length === 0) {
    throw new Error("runtime load proof activationRoot must be a non-empty string");
  }

  const resolvedActivationRoot = path.resolve(record.activationRoot);
  if (resolvedActivationRoot !== activationRoot) {
    throw new Error(
      `runtime load proof activationRoot mismatch: expected ${activationRoot}, received ${resolvedActivationRoot}`,
    );
  }

  const updatedAt =
    typeof record.updatedAt === "string" && record.updatedAt.trim().length > 0
      ? normalizeIsoTimestamp(record.updatedAt, "updatedAt")
      : null;

  if (updatedAt === null) {
    throw new Error("runtime load proof updatedAt must be a non-empty ISO timestamp");
  }

  if (!Array.isArray(record.profiles)) {
    throw new Error("runtime load proof profiles must be an array");
  }

  const profiles = record.profiles.map((entry, index) => validateRuntimeLoadProofRecord(entry, index));

  return {
    contract: ATTACHMENT_RUNTIME_LOAD_PROOFS_CONTRACT,
    runtimeOwner: "openclaw",
    activationRoot,
    updatedAt,
    profiles,
  };
}

function buildEmptyRuntimeLoadProofs(activationRoot, updatedAt) {
  return {
    contract: ATTACHMENT_RUNTIME_LOAD_PROOFS_CONTRACT,
    runtimeOwner: "openclaw",
    activationRoot,
    updatedAt,
    profiles: [],
  };
}

function writeRuntimeLoadProofs(proofPath, proofs) {
  mkdirSync(path.dirname(proofPath), { recursive: true });
  writeFileSync(proofPath, `${JSON.stringify(proofs, null, 2)}\n`, "utf8");
}

function deriveOpenClawHomeFromExtensionEntryPath(extensionEntryPath) {
  const openclawHome = resolveOpenClawHomeFromExtensionEntryPath(extensionEntryPath);

  if (openclawHome === null) {
    throw new Error(`extension entry path ${extensionEntryPath} is not nested under an OpenClaw extensions dir`);
  }

  return canonicalizeFilesystemPath(openclawHome);
}

export function resolveAttachmentRuntimeLoadProofsPath(activationRoot) {
  return path.join(path.resolve(activationRoot), ATTACHMENT_TRUTH_DIRNAME, ATTACHMENT_RUNTIME_LOAD_PROOFS_BASENAME);
}

export function listOpenClawProfileRuntimeLoadProofs(activationRoot) {
  const resolvedActivationRoot = path.resolve(activationRoot);
  const proofPath = resolveAttachmentRuntimeLoadProofsPath(resolvedActivationRoot);

  if (!existsSync(proofPath)) {
    return {
      path: proofPath,
      proofs: null,
      error: null,
    };
  }

  try {
    const parsed = JSON.parse(readFileSync(proofPath, "utf8"));

    return {
      path: proofPath,
      proofs: validateRuntimeLoadProofs(resolvedActivationRoot, parsed),
      error: null,
    };
  } catch (error) {
    return {
      path: proofPath,
      proofs: null,
      error: toErrorMessage(error),
    };
  }
}

export function recordOpenClawProfileRuntimeLoadProof(input) {
  const activationRoot = path.resolve(input.activationRoot);
  const loadedAt = normalizeIsoTimestamp(input.loadedAt, "loadedAt", new Date().toISOString());
  const extensionEntryPath = canonicalizeFilesystemPath(input.extensionEntryPath);
  const openclawHome = deriveOpenClawHomeFromExtensionEntryPath(extensionEntryPath);
  const inspection = inspectOpenClawHome(openclawHome);
  const packageIdentity = readPackageIdentityFromExtensionEntryPath(extensionEntryPath);
  const loadedProofs = listOpenClawProfileRuntimeLoadProofs(activationRoot);

  if (loadedProofs.error !== null) {
    throw new Error(`runtime load proof file ${loadedProofs.path} is unreadable: ${loadedProofs.error}`);
  }

  const nextRecord = {
    openclawHome: canonicalizeFilesystemPath(openclawHome),
    profileId: inspection.profileId,
    profileSource: inspection.profileSource,
    loadedAt,
    extensionEntryPath,
    packageJsonPath: packageIdentity.packageJsonPath,
    packageName: packageIdentity.packageName,
    packageVersion: packageIdentity.packageVersion,
    packageIdentity: packageIdentity.packageIdentity,
  };
  const nextProofs =
    loadedProofs.proofs === null
      ? buildEmptyRuntimeLoadProofs(activationRoot, loadedAt)
      : {
          ...loadedProofs.proofs,
          updatedAt: loadedAt,
          profiles: [...loadedProofs.proofs.profiles],
        };

  nextProofs.profiles = nextProofs.profiles
    .filter((record) => canonicalizeFilesystemPath(record.openclawHome) !== nextRecord.openclawHome)
    .concat(nextRecord)
    .sort((left, right) => left.openclawHome.localeCompare(right.openclawHome));

  writeRuntimeLoadProofs(loadedProofs.path, nextProofs);
  return nextRecord;
}

export function clearOpenClawProfileRuntimeLoadProof(input) {
  const activationRoot = path.resolve(input.activationRoot);
  const clearedAt = normalizeIsoTimestamp(input.clearedAt, "clearedAt", new Date().toISOString());
  const openclawHome = canonicalizeFilesystemPath(input.openclawHome);
  const loadedProofs = listOpenClawProfileRuntimeLoadProofs(activationRoot);

  if (loadedProofs.error !== null) {
    throw new Error(`runtime load proof file ${loadedProofs.path} is unreadable: ${loadedProofs.error}`);
  }

  if (loadedProofs.proofs === null) {
    return false;
  }

  const filteredProfiles = loadedProofs.proofs.profiles.filter(
    (record) => canonicalizeFilesystemPath(record.openclawHome) !== openclawHome,
  );

  if (filteredProfiles.length === loadedProofs.proofs.profiles.length) {
    return false;
  }

  writeRuntimeLoadProofs(loadedProofs.path, {
    ...loadedProofs.proofs,
    updatedAt: clearedAt,
    profiles: filteredProfiles,
  });

  return true;
}
