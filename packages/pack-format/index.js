import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync } from "node:fs";
import path from "node:path";
import { canonicalJson, checksumJsonPayload } from "../contracts/index.js";

export const PACK_LAYOUT = {
  manifest: "manifest.json",
  graph: "graph.json",
  vectors: "vectors.json",
  router: "router.json",
};

export const LEARNING_SPINE_LOG_LAYOUT = {
  serveTimeRouteDecisions: "learning-spine/serve-time-route-decisions.jsonl",
  promotionActivationEvents: "learning-spine/promotion-activation-events.jsonl",
  supervisionLabelBindings: "learning-spine/supervision-label-bindings.jsonl",
  pgRouteUpdates: "learning-spine/pg-route-updates.jsonl",
};

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function pointerPath(activationRoot, slot) {
  return path.join(path.resolve(activationRoot), "pointers", `${slot}.json`);
}

function readJsonIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function stableId(prefix, value) {
  return `${prefix}-${createHash("sha256").update(canonicalJson(value)).digest("hex").slice(0, 12)}`;
}

export function computePayloadChecksum(value) {
  return checksumJsonPayload(value);
}

export function writePackFile(rootDir, relativePath, value) {
  const filePath = path.join(path.resolve(rootDir), relativePath);
  ensureDir(path.dirname(filePath));
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  return filePath;
}

export function loadPack(rootDir) {
  const resolvedRoot = path.resolve(rootDir);
  const manifest = readJsonIfExists(path.join(resolvedRoot, PACK_LAYOUT.manifest));
  const graph = readJsonIfExists(path.join(resolvedRoot, PACK_LAYOUT.graph));
  const vectors = readJsonIfExists(path.join(resolvedRoot, PACK_LAYOUT.vectors));
  const router = readJsonIfExists(path.join(resolvedRoot, PACK_LAYOUT.router));
  if (manifest === null && graph === null && vectors === null && router === null) {
    return null;
  }
  return { rootDir: resolvedRoot, manifest, graph, vectors, router };
}

export function loadActivationPointers(activationRoot) {
  const root = path.resolve(activationRoot);
  return {
    rootDir: root,
    pointers: {
      active: readJsonIfExists(pointerPath(root, "active")),
      candidate: readJsonIfExists(pointerPath(root, "candidate")),
      previous: readJsonIfExists(pointerPath(root, "previous")),
    },
  };
}

export function loadPackFromActivation(activationRoot, slot, options = {}) {
  const pointers = loadActivationPointers(activationRoot).pointers;
  const pointer = pointers?.[slot] ?? null;
  if (pointer === null) {
    if (options.requireActivationReady) {
      throw new Error(`activation slot ${slot} is empty`);
    }
    return null;
  }
  const pack = loadPack(pointer.packRootDir);
  if (pack === null) {
    if (options.requireActivationReady) {
      throw new Error(`pack at ${pointer.packRootDir} is unavailable`);
    }
    return null;
  }
  return pack;
}

export function inspectActivationState(activationRoot) {
  const { pointers } = loadActivationPointers(activationRoot);
  const buildSlot = (slot) => {
    const pointer = pointers[slot];
    if (pointer === null) {
      return null;
    }
    return {
      slot,
      packId: pointer.packId ?? null,
      packRootDir: pointer.packRootDir,
      activationReady: true,
    };
  };
  return {
    activationRoot: path.resolve(activationRoot),
    active: buildSlot("active"),
    candidate: buildSlot("candidate"),
    previous: buildSlot("previous"),
  };
}

export function describeActivationTarget(activationRoot, slot) {
  const inspection = inspectActivationState(activationRoot);
  return inspection?.[slot] ?? null;
}

export function describeActivationObservability(activationRoot, slot) {
  const target = describeActivationTarget(activationRoot, slot);
  return {
    activationRoot: path.resolve(activationRoot),
    slot,
    present: target !== null,
    activationReady: target?.activationReady ?? false,
    packId: target?.packId ?? null,
  };
}

export function describePackCompileTarget(target) {
  return target === null
    ? { available: false, packId: null }
    : { available: true, packId: target.packId ?? null, packRootDir: target.packRootDir };
}

export function stageCandidatePack(activationRoot, candidatePackRoot, options = {}) {
  const pointer = {
    slot: "candidate",
    packId: options.packId ?? path.basename(path.resolve(candidatePackRoot)),
    packRootDir: path.resolve(candidatePackRoot),
    stagedAt: options.stagedAt ?? new Date().toISOString(),
  };
  writeJson(pointerPath(activationRoot, "candidate"), pointer);
  return pointer;
}

export function promoteCandidatePack(activationRoot, options = {}) {
  const root = path.resolve(activationRoot);
  const pointers = loadActivationPointers(root).pointers;
  if (pointers.active !== null) {
    writeJson(pointerPath(root, "previous"), pointers.active);
  }
  const candidate = pointers.candidate;
  if (candidate === null) {
    throw new Error("candidate pack is not staged");
  }
  const active = {
    ...candidate,
    slot: "active",
    promotedAt: options.promotedAt ?? new Date().toISOString(),
  };
  writeJson(pointerPath(root, "active"), active);
  return active;
}

export function rollbackActivePack(activationRoot) {
  const root = path.resolve(activationRoot);
  const previous = readJsonIfExists(pointerPath(root, "previous"));
  if (previous === null) {
    return null;
  }
  const active = {
    ...previous,
    slot: "active",
    rolledBackAt: new Date().toISOString(),
  };
  writeJson(pointerPath(root, "active"), active);
  return active;
}

export function activatePack(activationRoot, candidatePackRoot, options = {}) {
  stageCandidatePack(activationRoot, candidatePackRoot, options);
  return promoteCandidatePack(activationRoot, options);
}

export function resolveLearningSpineLogPath(activationRoot, kind) {
  const relative = LEARNING_SPINE_LOG_LAYOUT[kind] ?? `learning-spine/${kind}.jsonl`;
  return path.join(path.resolve(activationRoot), relative);
}

export function appendLearningSpineLogEntry(activationRoot, kind, entry) {
  const filePath = resolveLearningSpineLogPath(activationRoot, kind);
  ensureDir(path.dirname(filePath));
  appendFileSync(filePath, `${JSON.stringify(entry)}\n`, "utf8");
  return filePath;
}

export function readLearningSpineLogEntries(activationRoot, kind) {
  const filePath = resolveLearningSpineLogPath(activationRoot, kind);
  if (!existsSync(filePath)) {
    return [];
  }
  return readFileSync(filePath, "utf8")
    .split(/\n/u)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line));
}

export function buildLearningSpineLogId(prefix, value) {
  return stableId(prefix, value);
}

export function summarizeStructuralGraphEvolution(graph) {
  return {
    blockCount: Array.isArray(graph?.blocks) ? graph.blocks.length : 0,
  };
}
