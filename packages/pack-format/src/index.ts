import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";

import {
  CONTRACT_IDS,
  type ActivationPointerRecordV1,
  type ActivationPointerSlot,
  type ActivationPointersV1,
  canonicalJson,
  type ArtifactManifestV1,
  checksumJsonPayload,
  type PackGraphPayloadV1,
  type PackVectorsPayloadV1,
  type RuntimeCompileTargetV1,
  type RouterArtifactV1,
  validateActivationPointers,
  validateArtifactManifest,
  validatePackGraphPayload,
  validatePackVectorsPayload,
  validateRouterArtifact
} from "@openclawbrain/contracts";

export const PACK_LAYOUT = {
  graph: "graph.json",
  manifest: "manifest.json",
  router: "router/model.json",
  vectors: "vectors.json"
} as const;

export const ACTIVATION_LAYOUT = {
  pointers: "activation-pointers.json"
} as const;

export interface PackDescriptor {
  rootDir: string;
  manifestPath: string;
  graphPath: string;
  vectorPath: string;
  routerPath: string | null;
  manifest: ArtifactManifestV1;
  graph: PackGraphPayloadV1;
  vectors: PackVectorsPayloadV1;
  router: RouterArtifactV1 | null;
}

export interface ActivationStateDescriptor {
  rootDir: string;
  pointerPath: string;
  pointers: ActivationPointersV1;
}

export interface ActivationSlotInspection {
  slot: ActivationPointerSlot;
  packId: string;
  routePolicy: ArtifactManifestV1["routePolicy"];
  routerIdentity: string | null;
  workspaceSnapshot: string;
  workspaceRevision: string | null;
  eventRange: ActivationPointerRecordV1["eventRange"];
  eventExportDigest: string | null;
  builtAt: string;
  activationReady: boolean;
  findings: string[];
}

export interface ActivationOperationPreview {
  allowed: boolean;
  findings: string[];
  nextPointers: ActivationPointersV1 | null;
}

export interface ActivationInspection {
  rootDir: string;
  pointerPath: string;
  pointers: ActivationPointersV1;
  active: ActivationSlotInspection | null;
  candidate: ActivationSlotInspection | null;
  previous: ActivationSlotInspection | null;
  promotion: ActivationOperationPreview;
  rollback: ActivationOperationPreview;
}

function compareIsoDates(left: string, right: string): number {
  return Date.parse(left) - Date.parse(right);
}

function sha256File(filePath: string): string {
  return `sha256-${createHash("sha256").update(readFileSync(filePath)).digest("hex")}`;
}

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function pushFileError(errors: string[], filePath: string, label: string): void {
  if (!existsSync(filePath)) {
    errors.push(`${label} not found: ${filePath}`);
  }
}

function validatePackAssetPath(assetPath: string, label: string): string[] {
  const errors: string[] = [];

  if (path.isAbsolute(assetPath)) {
    errors.push(`${label} must be relative to the pack root`);
  }

  const segments = assetPath.split(/[\\/]+/u);
  if (segments.includes("..")) {
    errors.push(`${label} must not escape the pack root`);
  }

  return errors;
}

function resolvePackAssetPath(rootDir: string, assetPath: string, label: string): string {
  const resolvedRootDir = path.resolve(rootDir);
  const resolvedAssetPath = path.resolve(resolvedRootDir, assetPath);
  const relativeAssetPath = path.relative(resolvedRootDir, resolvedAssetPath);

  if (relativeAssetPath.startsWith("..") || path.isAbsolute(relativeAssetPath)) {
    throw new Error(`Invalid pack descriptor: ${label} escapes pack root: ${assetPath}`);
  }

  return resolvedAssetPath;
}

export function validatePackDescriptor(manifest: ArtifactManifestV1): string[] {
  const errors = validateArtifactManifest(manifest);

  errors.push(...validatePackAssetPath(manifest.runtimeAssets.graphPath, "graphPath"));
  errors.push(...validatePackAssetPath(manifest.runtimeAssets.vectorPath, "vectorPath"));

  if (manifest.runtimeAssets.router.artifactPath !== null) {
    errors.push(...validatePackAssetPath(manifest.runtimeAssets.router.artifactPath, "router artifactPath"));
  }

  if (!manifest.runtimeAssets.graphPath.endsWith(".json")) {
    errors.push("graph payload must be json-addressable in the initial layout");
  }

  if (!manifest.runtimeAssets.vectorPath.endsWith(".json")) {
    errors.push("vector payload must be json-addressable in the initial layout");
  }

  if (manifest.runtimeAssets.router.artifactPath !== null && !manifest.runtimeAssets.router.artifactPath.endsWith(".json")) {
    errors.push("router payload must be json-addressable in the initial layout");
  }

  if (manifest.routePolicy === "requires_learned_routing" && manifest.runtimeAssets.router.artifactPath === null) {
    errors.push("learned-routing packs require a router artifact path");
  }

  return errors;
}

export function validatePackActivationReadiness(packOrRootDir: PackDescriptor | string): string[] {
  let pack: PackDescriptor;
  if (typeof packOrRootDir === "string") {
    try {
      pack = loadPack(packOrRootDir);
    } catch (error) {
      return [error instanceof Error ? error.message : String(error)];
    }
  } else {
    pack = packOrRootDir;
  }

  const errors: string[] = [];
  const { manifest, router } = pack;

  if (manifest.routePolicy !== "requires_learned_routing") {
    return errors;
  }

  if (manifest.runtimeAssets.router.kind !== "artifact") {
    errors.push("learned-routing packs require runtimeAssets.router.kind=artifact for activation");
  }
  if (manifest.runtimeAssets.router.identity === null) {
    errors.push("learned-routing packs require runtimeAssets.router.identity for activation");
  }
  if (manifest.runtimeAssets.router.artifactPath === null) {
    errors.push("learned-routing packs require runtimeAssets.router.artifactPath for activation");
  }
  if (manifest.payloadChecksums.router === null) {
    errors.push("learned-routing packs require router checksum metadata for activation");
  }
  if (router === null) {
    errors.push("learned-routing packs require a router artifact for activation");
    return errors;
  }
  if (router.requiresLearnedRouting !== true) {
    errors.push("learned-routing packs require router.requiresLearnedRouting=true for activation");
  }
  if (manifest.runtimeAssets.router.identity !== router.routerIdentity) {
    errors.push(
      `learned-routing packs require router identity ${manifest.runtimeAssets.router.identity ?? "null"} but found ${router.routerIdentity}`
    );
  }

  return errors;
}

export function computePayloadChecksum(value: unknown): string {
  return checksumJsonPayload(value);
}

export function writePackFile(rootDir: string, relativePath: string, payload: unknown): string {
  const filePath = path.join(rootDir, relativePath);
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, canonicalJson(payload), "utf8");
  return filePath;
}

function emptyActivationPointers(): ActivationPointersV1 {
  return {
    contract: CONTRACT_IDS.activationPointers,
    active: null,
    candidate: null,
    previous: null
  };
}

function buildActivationPointerRecord(
  slot: ActivationPointerSlot,
  pack: PackDescriptor,
  updatedAt: string
): ActivationPointerRecordV1 {
  return {
    slot,
    packId: pack.manifest.packId,
    packRootDir: path.resolve(pack.rootDir),
    manifestPath: path.resolve(pack.manifestPath),
    manifestDigest: sha256File(pack.manifestPath),
    routePolicy: pack.manifest.routePolicy,
    routerIdentity: pack.manifest.runtimeAssets.router.identity,
    workspaceSnapshot: pack.manifest.provenance.workspaceSnapshot,
    workspaceRevision: pack.manifest.provenance.workspace.revision,
    eventRange: {
      start: pack.manifest.provenance.eventRange.start,
      end: pack.manifest.provenance.eventRange.end,
      count: pack.manifest.provenance.eventRange.count
    },
    eventExportDigest: pack.manifest.provenance.eventExports?.exportDigest ?? null,
    builtAt: pack.manifest.provenance.builtAt,
    updatedAt
  };
}

function buildCompileTargetFromPack(pack: PackDescriptor): RuntimeCompileTargetV1 {
  return {
    packId: pack.manifest.packId,
    routePolicy: pack.manifest.routePolicy,
    routerIdentity: pack.manifest.runtimeAssets.router.identity,
    workspaceSnapshot: pack.manifest.provenance.workspaceSnapshot,
    workspaceRevision: pack.manifest.provenance.workspace.revision,
    eventRange: {
      start: pack.manifest.provenance.eventRange.start,
      end: pack.manifest.provenance.eventRange.end,
      count: pack.manifest.provenance.eventRange.count
    },
    eventExportDigest: pack.manifest.provenance.eventExports?.exportDigest ?? null,
    builtAt: pack.manifest.provenance.builtAt
  };
}

function pointerPackIdentityFindings(
  slot: ActivationPointerSlot,
  record: ActivationPointerRecordV1,
  pack: PackDescriptor
): string[] {
  const expected = buildActivationPointerRecord(slot, pack, record.updatedAt);
  const errors: string[] = [];

  if (path.resolve(record.packRootDir) !== path.resolve(expected.packRootDir)) {
    errors.push(`pointer packRootDir ${record.packRootDir} does not match pack root ${expected.packRootDir}`);
  }
  if (path.resolve(record.manifestPath) !== path.resolve(expected.manifestPath)) {
    errors.push(`pointer manifestPath ${record.manifestPath} does not match pack manifest ${expected.manifestPath}`);
  }
  if (record.manifestDigest !== expected.manifestDigest) {
    errors.push(`pointer manifestDigest ${record.manifestDigest} does not match pack manifest digest ${expected.manifestDigest}`);
  }
  if (record.routePolicy !== expected.routePolicy) {
    errors.push(`pointer routePolicy ${record.routePolicy} does not match pack routePolicy ${expected.routePolicy}`);
  }
  if (record.routerIdentity !== expected.routerIdentity) {
    errors.push(`pointer routerIdentity ${record.routerIdentity ?? "null"} does not match pack router identity ${expected.routerIdentity ?? "null"}`);
  }
  if (record.workspaceSnapshot !== expected.workspaceSnapshot) {
    errors.push(`pointer workspaceSnapshot ${record.workspaceSnapshot} does not match pack workspaceSnapshot ${expected.workspaceSnapshot}`);
  }
  if ((record.workspaceRevision ?? null) !== (expected.workspaceRevision ?? null)) {
    errors.push(
      `pointer workspaceRevision ${record.workspaceRevision ?? "null"} does not match pack workspace revision ${expected.workspaceRevision ?? "null"}`
    );
  }
  if (record.eventRange.start !== expected.eventRange.start) {
    errors.push(`pointer eventRange.start ${record.eventRange.start} does not match pack eventRange.start ${expected.eventRange.start}`);
  }
  if (record.eventRange.end !== expected.eventRange.end) {
    errors.push(`pointer eventRange.end ${record.eventRange.end} does not match pack eventRange.end ${expected.eventRange.end}`);
  }
  if (record.eventRange.count !== expected.eventRange.count) {
    errors.push(`pointer eventRange.count ${record.eventRange.count} does not match pack eventRange.count ${expected.eventRange.count}`);
  }
  if ((record.eventExportDigest ?? null) !== (expected.eventExportDigest ?? null)) {
    errors.push(
      `pointer eventExportDigest ${record.eventExportDigest ?? "null"} does not match pack event export digest ${expected.eventExportDigest ?? "null"}`
    );
  }
  if (record.builtAt !== expected.builtAt) {
    errors.push(`pointer builtAt ${record.builtAt} does not match pack builtAt ${expected.builtAt}`);
  }

  return errors;
}

function assertPointerPinnedToPack(slot: ActivationPointerSlot, record: ActivationPointerRecordV1 | null, pack: PackDescriptor): void {
  if (record === null || record.packId !== pack.manifest.packId) {
    return;
  }

  const errors = pointerPackIdentityFindings(slot, record, pack);
  if (errors.length > 0) {
    throw new Error(`${slot} pointer for packId ${record.packId} is already pinned to a different manifest: ${errors.join("; ")}`);
  }
}

function assertRetainedPointerMatchesManifest(
  slot: ActivationPointerSlot,
  record: ActivationPointerRecordV1 | null,
  options: {
    requireActivationReady?: boolean;
  } = {}
): void {
  if (record === null) {
    return;
  }

  try {
    ensurePackRecordMatchesManifest(record, {
      requireActivationReady: options.requireActivationReady === true
    });
  } catch (error) {
    throw new Error(`${slot} pointer cannot be retained: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function ensurePackRecordMatchesManifest(
  record: ActivationPointerRecordV1,
  options: {
    requireActivationReady?: boolean;
  } = {}
): PackDescriptor {
  const pack = loadPack(path.resolve(record.packRootDir));
  const errors: string[] = [];

  if (path.resolve(pack.manifestPath) !== path.resolve(record.manifestPath)) {
    errors.push(`pointer manifestPath ${record.manifestPath} does not match pack manifest ${pack.manifestPath}`);
  }
  const manifestDigest = sha256File(pack.manifestPath);
  if (manifestDigest !== record.manifestDigest) {
    errors.push(`pointer manifestDigest ${record.manifestDigest} does not match pack manifest digest ${manifestDigest}`);
  }
  if (pack.manifest.packId !== record.packId) {
    errors.push(`pointer packId ${record.packId} does not match manifest packId ${pack.manifest.packId}`);
  }
  if (pack.manifest.routePolicy !== record.routePolicy) {
    errors.push(`pointer routePolicy ${record.routePolicy} does not match manifest routePolicy ${pack.manifest.routePolicy}`);
  }
  if (pack.manifest.runtimeAssets.router.identity !== record.routerIdentity) {
    errors.push(
      `pointer routerIdentity ${record.routerIdentity ?? "null"} does not match manifest router identity ${pack.manifest.runtimeAssets.router.identity ?? "null"}`
    );
  }
  if (pack.manifest.provenance.workspaceSnapshot !== record.workspaceSnapshot) {
    errors.push(
      `pointer workspaceSnapshot ${record.workspaceSnapshot} does not match manifest workspaceSnapshot ${pack.manifest.provenance.workspaceSnapshot}`
    );
  }
  if ((pack.manifest.provenance.workspace.revision ?? null) !== record.workspaceRevision) {
    errors.push(
      `pointer workspaceRevision ${record.workspaceRevision ?? "null"} does not match manifest workspace revision ${pack.manifest.provenance.workspace.revision ?? "null"}`
    );
  }
  if (pack.manifest.provenance.eventRange.start !== record.eventRange.start) {
    errors.push(
      `pointer eventRange.start ${record.eventRange.start} does not match manifest eventRange.start ${pack.manifest.provenance.eventRange.start}`
    );
  }
  if (pack.manifest.provenance.eventRange.end !== record.eventRange.end) {
    errors.push(`pointer eventRange.end ${record.eventRange.end} does not match manifest eventRange.end ${pack.manifest.provenance.eventRange.end}`);
  }
  if (pack.manifest.provenance.eventRange.count !== record.eventRange.count) {
    errors.push(
      `pointer eventRange.count ${record.eventRange.count} does not match manifest eventRange.count ${pack.manifest.provenance.eventRange.count}`
    );
  }
  if ((pack.manifest.provenance.eventExports?.exportDigest ?? null) !== record.eventExportDigest) {
    errors.push(
      `pointer eventExportDigest ${record.eventExportDigest ?? "null"} does not match manifest event export digest ${pack.manifest.provenance.eventExports?.exportDigest ?? "null"}`
    );
  }
  if (pack.manifest.provenance.builtAt !== record.builtAt) {
    errors.push(`pointer builtAt ${record.builtAt} does not match manifest builtAt ${pack.manifest.provenance.builtAt}`);
  }

  if (options.requireActivationReady === true) {
    errors.push(...validatePackActivationReadiness(pack));
  }

  if (errors.length > 0) {
    throw new Error(`Invalid activation pointer: ${errors.join("; ")}`);
  }

  return pack;
}

function writeActivationPointers(rootDir: string, pointers: ActivationPointersV1): ActivationStateDescriptor {
  const errors = validateActivationPointers(pointers);
  if (errors.length > 0) {
    throw new Error(`Invalid activation pointers: ${errors.join("; ")}`);
  }

  const resolvedRootDir = path.resolve(rootDir);
  const pointerPath = path.join(resolvedRootDir, ACTIVATION_LAYOUT.pointers);
  mkdirSync(path.dirname(pointerPath), { recursive: true });
  writeFileSync(pointerPath, canonicalJson(pointers), "utf8");

  return {
    rootDir: resolvedRootDir,
    pointerPath,
    pointers
  };
}

function inspectPointerRecord(
  slot: ActivationPointerSlot,
  record: ActivationPointerRecordV1 | null
): ActivationSlotInspection | null {
  if (record === null) {
    return null;
  }

  const findings: string[] = [];
  try {
    const pack = ensurePackRecordMatchesManifest(record, { requireActivationReady: true });
    findings.push(...validatePackActivationReadiness(pack));
  } catch (error) {
    findings.push(error instanceof Error ? error.message : String(error));
  }

  return {
    slot,
    packId: record.packId,
    routePolicy: record.routePolicy,
    routerIdentity: record.routerIdentity,
    workspaceSnapshot: record.workspaceSnapshot,
    workspaceRevision: record.workspaceRevision,
    eventRange: record.eventRange,
    eventExportDigest: record.eventExportDigest,
    builtAt: record.builtAt,
    activationReady: findings.length === 0,
    findings
  };
}

function promotionCoherenceFindings(active: RuntimeCompileTargetV1, candidate: RuntimeCompileTargetV1): string[] {
  const findings: string[] = [];

  if (compareIsoDates(candidate.builtAt, active.builtAt) < 0) {
    findings.push("candidate pack builtAt must not precede active pack builtAt during promotion");
  }
  if (candidate.eventRange.end < active.eventRange.end) {
    findings.push("candidate eventRange.end must be >= active eventRange.end during promotion");
  }

  return findings;
}

function rollbackCoherenceFindings(active: RuntimeCompileTargetV1, previous: RuntimeCompileTargetV1): string[] {
  const findings: string[] = [];

  if (compareIsoDates(previous.builtAt, active.builtAt) > 0) {
    findings.push("previous pack builtAt must not follow active pack builtAt during rollback");
  }
  if (previous.eventRange.end > active.eventRange.end) {
    findings.push("previous eventRange.end must be <= active eventRange.end during rollback");
  }

  return findings;
}

function duplicatePackIdError(slot: ActivationPointerSlot, record: ActivationPointerRecordV1): string {
  return `${slot} pointer cannot reuse packId ${record.packId}`;
}

function previewPromotionPointers(current: ActivationPointersV1, updatedAt: string): ActivationOperationPreview {
  const findings: string[] = [];

  if (current.candidate === null) {
    findings.push("candidate pointer is required for promotion");
  }

  let candidatePack: PackDescriptor | null = null;
  if (current.candidate !== null) {
    try {
      candidatePack = ensurePackRecordMatchesManifest(current.candidate, { requireActivationReady: true });
    } catch (error) {
      findings.push(error instanceof Error ? error.message : String(error));
    }
  }

  let activePack: PackDescriptor | null = null;
  if (current.active !== null) {
    try {
      activePack = ensurePackRecordMatchesManifest(current.active, { requireActivationReady: true });
    } catch (error) {
      findings.push(error instanceof Error ? error.message : String(error));
    }
  }

  if (findings.length > 0 || candidatePack === null) {
    return {
      allowed: false,
      findings,
      nextPointers: null
    };
  }

  if (activePack !== null) {
    findings.push(...promotionCoherenceFindings(buildCompileTargetFromPack(activePack), buildCompileTargetFromPack(candidatePack)));
  }

  if (findings.length > 0) {
    return {
      allowed: false,
      findings,
      nextPointers: null
    };
  }

  return {
    allowed: true,
    findings: [],
    nextPointers: {
      contract: CONTRACT_IDS.activationPointers,
      active: buildActivationPointerRecord("active", candidatePack, updatedAt),
      candidate: null,
      previous: activePack === null ? null : buildActivationPointerRecord("previous", activePack, updatedAt)
    }
  };
}

function previewRollbackPointers(current: ActivationPointersV1, updatedAt: string): ActivationOperationPreview {
  const findings: string[] = [];

  if (current.active === null) {
    findings.push("active pointer is required for rollback");
  }
  if (current.previous === null) {
    findings.push("previous pointer is required for rollback");
  }
  if (current.candidate !== null) {
    findings.push("rollback requires an empty candidate pointer");
  }

  let previousPack: PackDescriptor | null = null;
  if (current.previous !== null) {
    try {
      previousPack = ensurePackRecordMatchesManifest(current.previous, { requireActivationReady: true });
    } catch (error) {
      findings.push(error instanceof Error ? error.message : String(error));
    }
  }

  let activePack: PackDescriptor | null = null;
  if (current.active !== null) {
    try {
      activePack = ensurePackRecordMatchesManifest(current.active, { requireActivationReady: true });
    } catch (error) {
      findings.push(error instanceof Error ? error.message : String(error));
    }
  }

  if (findings.length > 0 || previousPack === null) {
    return {
      allowed: false,
      findings,
      nextPointers: null
    };
  }

  if (activePack !== null) {
    findings.push(...rollbackCoherenceFindings(buildCompileTargetFromPack(activePack), buildCompileTargetFromPack(previousPack)));
  }

  if (findings.length > 0) {
    return {
      allowed: false,
      findings,
      nextPointers: null
    };
  }

  return {
    allowed: true,
    findings: [],
    nextPointers: {
      contract: CONTRACT_IDS.activationPointers,
      active: buildActivationPointerRecord("active", previousPack, updatedAt),
      candidate: activePack === null ? null : buildActivationPointerRecord("candidate", activePack, updatedAt),
      previous: null
    }
  };
}

function assertPackIdAvailable(
  current: ActivationPointersV1,
  slot: ActivationPointerSlot,
  packId: string
): void {
  const duplicates = (["active", "candidate", "previous"] as const).filter((key) => {
    if (key === slot) {
      return false;
    }
    return current[key]?.packId === packId;
  });

  const duplicateSlot = duplicates[0];
  if (duplicateSlot !== undefined) {
    throw new Error(duplicatePackIdError(duplicateSlot, current[duplicateSlot] as ActivationPointerRecordV1));
  }
}

export function describePackCompileTarget(packOrRootDir: PackDescriptor | string): RuntimeCompileTargetV1 {
  const pack = typeof packOrRootDir === "string" ? loadPack(packOrRootDir) : packOrRootDir;
  return buildCompileTargetFromPack(pack);
}

export function loadPackFromActivation(
  rootDir: string,
  slot: ActivationPointerSlot = "active",
  options: {
    requireActivationReady?: boolean;
  } = {}
): PackDescriptor | null {
  const record = loadActivationPointers(rootDir).pointers[slot];
  if (record === null) {
    return null;
  }

  return ensurePackRecordMatchesManifest(record, {
    requireActivationReady: options.requireActivationReady === true
  });
}

export function describeActivationTarget(
  rootDir: string,
  slot: ActivationPointerSlot = "active",
  options: {
    requireActivationReady?: boolean;
  } = {}
): RuntimeCompileTargetV1 | null {
  const pack = loadPackFromActivation(rootDir, slot, options);
  return pack === null ? null : buildCompileTargetFromPack(pack);
}

export function loadActivationPointers(rootDir: string): ActivationStateDescriptor {
  const resolvedRootDir = path.resolve(rootDir);
  const pointerPath = path.join(resolvedRootDir, ACTIVATION_LAYOUT.pointers);
  if (!existsSync(pointerPath)) {
    return {
      rootDir: resolvedRootDir,
      pointerPath,
      pointers: emptyActivationPointers()
    };
  }

  const pointers = readJsonFile<ActivationPointersV1>(pointerPath);
  const errors = validateActivationPointers(pointers);
  if (errors.length > 0) {
    throw new Error(`Invalid activation pointers: ${errors.join("; ")}`);
  }

  return {
    rootDir: resolvedRootDir,
    pointerPath,
    pointers
  };
}

export function inspectActivationState(rootDir: string, updatedAt = "2026-03-06T00:00:00.000Z"): ActivationInspection {
  const state = loadActivationPointers(rootDir);

  return {
    ...state,
    active: inspectPointerRecord("active", state.pointers.active),
    candidate: inspectPointerRecord("candidate", state.pointers.candidate),
    previous: inspectPointerRecord("previous", state.pointers.previous),
    promotion: previewPromotionPointers(state.pointers, updatedAt),
    rollback: previewRollbackPointers(state.pointers, updatedAt)
  };
}

export function activatePack(rootDir: string, packRootDir: string, updatedAt = "2026-03-06T00:00:00.000Z"): ActivationStateDescriptor {
  const current = loadActivationPointers(rootDir).pointers;
  const pack = loadPack(path.resolve(packRootDir));
  const activationErrors = validatePackActivationReadiness(pack);
  if (activationErrors.length > 0) {
    throw new Error(`Pack is not activation-ready: ${activationErrors.join("; ")}`);
  }

  assertPointerPinnedToPack("active", current.active, pack);

  if (current.active?.packId !== pack.manifest.packId) {
    assertPackIdAvailable(current, "active", pack.manifest.packId);
  }

  let previous: ActivationPointerRecordV1 | null = null;
  if (current.active !== null && current.active.packId !== pack.manifest.packId) {
    const activePack = ensurePackRecordMatchesManifest(current.active, { requireActivationReady: true });
    previous = buildActivationPointerRecord("previous", activePack, updatedAt);
  }

  return writeActivationPointers(rootDir, {
    contract: CONTRACT_IDS.activationPointers,
    active: buildActivationPointerRecord("active", pack, updatedAt),
    candidate: null,
    previous
  });
}

export function stageCandidatePack(rootDir: string, packRootDir: string, updatedAt = "2026-03-06T00:00:00.000Z"): ActivationStateDescriptor {
  const current = loadActivationPointers(rootDir).pointers;
  const pack = loadPack(path.resolve(packRootDir));

  assertPointerPinnedToPack("candidate", current.candidate, pack);
  assertRetainedPointerMatchesManifest("active", current.active, { requireActivationReady: true });
  assertRetainedPointerMatchesManifest("previous", current.previous, { requireActivationReady: true });
  assertPackIdAvailable(current, "candidate", pack.manifest.packId);

  return writeActivationPointers(rootDir, {
    contract: CONTRACT_IDS.activationPointers,
    active: current.active,
    candidate: buildActivationPointerRecord("candidate", pack, updatedAt),
    previous: current.previous
  });
}

export function promoteCandidatePack(rootDir: string, updatedAt = "2026-03-06T00:00:00.000Z"): ActivationStateDescriptor {
  const preview = previewPromotionPointers(loadActivationPointers(rootDir).pointers, updatedAt);
  if (!preview.allowed || preview.nextPointers === null) {
    throw new Error(`Promotion blocked: ${preview.findings.join("; ")}`);
  }

  return writeActivationPointers(rootDir, preview.nextPointers);
}

export function rollbackActivePack(rootDir: string, updatedAt = "2026-03-06T00:00:00.000Z"): ActivationStateDescriptor {
  const preview = previewRollbackPointers(loadActivationPointers(rootDir).pointers, updatedAt);
  if (!preview.allowed || preview.nextPointers === null) {
    throw new Error(`Rollback blocked: ${preview.findings.join("; ")}`);
  }

  return writeActivationPointers(rootDir, preview.nextPointers);
}

export function loadPack(rootDir: string): PackDescriptor {
  const manifestPath = path.join(rootDir, PACK_LAYOUT.manifest);
  if (!existsSync(manifestPath)) {
    throw new Error(`pack manifest not found: ${manifestPath}`);
  }

  const manifest = readJsonFile<ArtifactManifestV1>(manifestPath);
  const manifestErrors = validatePackDescriptor(manifest);
  if (manifestErrors.length > 0) {
    throw new Error(`Invalid pack descriptor: ${manifestErrors.join("; ")}`);
  }

  const graphPath = resolvePackAssetPath(rootDir, manifest.runtimeAssets.graphPath, "graph payload");
  const vectorPath = resolvePackAssetPath(rootDir, manifest.runtimeAssets.vectorPath, "vector payload");
  const routerPath =
    manifest.runtimeAssets.router.artifactPath === null
      ? null
      : resolvePackAssetPath(rootDir, manifest.runtimeAssets.router.artifactPath, "router payload");

  const fileErrors: string[] = [];
  pushFileError(fileErrors, graphPath, "graph payload");
  pushFileError(fileErrors, vectorPath, "vector payload");
  if (routerPath !== null && manifest.runtimeAssets.router.kind !== "none") {
    pushFileError(fileErrors, routerPath, "router payload");
  }
  if (fileErrors.length > 0) {
    throw new Error(`Invalid pack descriptor: ${fileErrors.join("; ")}`);
  }

  const graph = readJsonFile<PackGraphPayloadV1>(graphPath);
  const vectors = readJsonFile<PackVectorsPayloadV1>(vectorPath);
  const router = routerPath === null ? null : readJsonFile<RouterArtifactV1>(routerPath);

  const payloadErrors = [
    ...validatePackGraphPayload(graph, manifest.packId),
    ...validatePackVectorsPayload(vectors, graph),
    ...(router === null ? [] : validateRouterArtifact(router, manifest))
  ];

  if (manifest.payloadChecksums.graph !== sha256File(graphPath)) {
    payloadErrors.push("graph checksum does not match manifest");
  }
  if (manifest.payloadChecksums.vector !== sha256File(vectorPath)) {
    payloadErrors.push("vector checksum does not match manifest");
  }

  if (routerPath === null) {
    if (manifest.payloadChecksums.router !== null) {
      payloadErrors.push("router checksum must be null when router artifact is absent");
    }
  } else {
    const routerChecksum = sha256File(routerPath);
    if (manifest.payloadChecksums.router !== routerChecksum) {
      payloadErrors.push("router checksum does not match manifest");
    }
  }

  if (payloadErrors.length > 0) {
    throw new Error(`Invalid pack descriptor: ${payloadErrors.join("; ")}`);
  }

  return {
    rootDir,
    manifestPath,
    graphPath,
    vectorPath,
    routerPath,
    manifest,
    graph,
    vectors,
    router
  };
}
