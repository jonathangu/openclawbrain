#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { canonicalJson } from "./openclawbrain-contracts.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultRepoRoot = path.resolve(__dirname, "../../../..");
const defaultWorkspaceRoot = path.resolve(defaultRepoRoot, "..");
const defaultGraphifyRoot = path.join(defaultRepoRoot, "artifacts");
const defaultOcbRoot = defaultRepoRoot;
const defaultOutputRoot = path.join(defaultWorkspaceRoot, "artifacts", "graphify-maintenance-diff");

export const GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1 = {
  maintenanceDiff: "maintenance-diff.json",
  summary: "summary.md",
  proposalSuggestion: "proposal-suggestion.json",
  verdict: "verdict.json",
};

function stableJson(value) {
  return canonicalJson(value);
}

function sha256Text(text) {
  return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  writeFileSync(filePath, `${stableJson(value)}\n`, "utf8");
}

function writeText(filePath, value) {
  writeFileSync(filePath, `${value}\n`, "utf8");
}

function readJsonIfExists(filePath) {
  if (!existsSync(filePath)) {
    return null;
  }
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function readTextIfExists(filePath) {
  return existsSync(filePath) ? readFileSync(filePath, "utf8") : null;
}

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function slugify(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/gu, "-")
    .replace(/^-+|-+$/gu, "")
    .replace(/-{2,}/gu, "-") || "bundle";
}

function timestampToken(value = new Date().toISOString()) {
  return String(value).replace(/[:]/g, "-");
}

function relativeWorkspacePath(absPath, workspaceRoot) {
  const resolvedPath = path.resolve(absPath);
  const relative = path.relative(workspaceRoot, resolvedPath);
  return relative.startsWith("..") ? resolvedPath : relative.replace(/\\/g, "/");
}

function uniqueBy(items, keyFn) {
  const seen = new Set();
  const result = [];
  for (const item of items) {
    const key = keyFn(item);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(item);
  }
  return result;
}

function tryReadDir(dirPath) {
  try {
    return readdirSync(dirPath, { withFileTypes: true });
  } catch {
    return [];
  }
}

function containsBundleMarkers(root) {
  if (!existsSync(root)) {
    return false;
  }
  const markers = [
    "pack.manifest.json",
    "import-slice.json",
    "candidate-pack-input.json",
    "graph.json",
    "graphify-summary.json",
    "surface-map.json",
    "proposal-report.json",
    "verdict.json",
  ];
  return markers.some((marker) => existsSync(path.join(root, marker)));
}

function discoverBundleRootsFromContainer(containerRoot) {
  const roots = [];
  if (!existsSync(containerRoot)) {
    return roots;
  }
  for (const entry of tryReadDir(containerRoot)) {
    if (!entry.isDirectory()) {
      continue;
    }
    const childRoot = path.join(containerRoot, entry.name);
    if (containsBundleMarkers(childRoot)) {
      roots.push(childRoot);
    }
  }
  return roots;
}

function discoverCurrentBundleRoots(graphifyRoot) {
  const resolvedRoot = path.resolve(graphifyRoot);
  const roots = [];
  if (containsBundleMarkers(resolvedRoot)) {
    roots.push(resolvedRoot);
  }
  const directChildren = [
    "compiled",
    "import",
    "run",
    "teacher-v3-proof",
  ];
  for (const child of directChildren) {
    const childRoot = path.join(resolvedRoot, child);
    if (containsBundleMarkers(childRoot)) {
      roots.push(childRoot);
    }
  }
  roots.push(...discoverBundleRootsFromContainer(path.join(resolvedRoot, "graphify-runs")));
  roots.push(...discoverBundleRootsFromContainer(path.join(resolvedRoot, "graphify-imports")));
  roots.push(...discoverBundleRootsFromContainer(path.join(resolvedRoot, "graphify-source-bundles")));
  return uniqueBy(roots, (value) => value);
}

function discoverOcbBundleRoots(ocbRoot) {
  const resolvedRoot = path.resolve(ocbRoot);
  const roots = [];
  if (containsBundleMarkers(resolvedRoot)) {
    roots.push(resolvedRoot);
  }
  const directChildren = [
    "candidate",
    "compiled",
    "promoted",
    "teacher-v3-shadow-examples",
    "teacher-v3-promotable-examples",
    "teacher-v3-proof",
  ];
  for (const child of directChildren) {
    const childRoot = path.join(resolvedRoot, child);
    if (containsBundleMarkers(childRoot)) {
      roots.push(childRoot);
    }
  }
  const explicitFixture = path.join(resolvedRoot, "artifacts", "fixtures", "compiled-artifacts", "target-state-scaffold");
  if (containsBundleMarkers(explicitFixture)) {
    roots.push(explicitFixture);
  }
  return uniqueBy(roots, (value) => value);
}

function loadArtifactMeta(bundleRoot, artifactSummary) {
  const metaPath = typeof artifactSummary.metaPath === "string" && artifactSummary.metaPath.trim().length > 0
    ? path.resolve(bundleRoot, artifactSummary.metaPath)
    : path.join(bundleRoot, "artifacts", artifactSummary.artifactId, "artifact.meta.json");
  const markdownPath = typeof artifactSummary.markdownPath === "string" && artifactSummary.markdownPath.trim().length > 0
    ? path.resolve(bundleRoot, artifactSummary.markdownPath)
    : path.join(bundleRoot, "artifacts", artifactSummary.artifactId, "artifact.md");
  const meta = readJsonIfExists(metaPath) ?? {};
  const markdownText = readTextIfExists(markdownPath);
  return {
    ...meta,
    artifactId: artifactSummary.artifactId,
    kind: meta.kind ?? artifactSummary.kind ?? "unknown",
    title: meta.title ?? artifactSummary.title ?? artifactSummary.artifactId,
    markdownPath,
    metaPath,
    markdownText,
    bundleRoot,
    artifactSummary,
  };
}

function loadBundleSnapshot(bundleRoot, role) {
  const resolvedRoot = path.resolve(bundleRoot);
  const packManifest = readJsonIfExists(path.join(resolvedRoot, "pack.manifest.json"));
  const importSlice = readJsonIfExists(path.join(resolvedRoot, "import-slice.json"));
  const candidatePackInput = readJsonIfExists(path.join(resolvedRoot, "candidate-pack-input.json"));
  const graph = readJsonIfExists(path.join(resolvedRoot, "graph.json"));
  const graphSummary = readJsonIfExists(path.join(resolvedRoot, "graphify-summary.json"));
  const surfaceMap = readJsonIfExists(path.join(resolvedRoot, "surface-map.json"));
  const proposalReport = readJsonIfExists(path.join(resolvedRoot, "proposal-report.json"));
  const verdict = readJsonIfExists(path.join(resolvedRoot, "verdict.json"));
  const labels = readJsonIfExists(path.join(resolvedRoot, "labels.json"));
  const evidencePointers = readJsonIfExists(path.join(resolvedRoot, "evidence-pointers.json"));
  const rationalePointers = readJsonIfExists(path.join(resolvedRoot, "rationale-pointers.json"));

  const artifactRecords = [];
  const artifactSummaries = Array.isArray(packManifest?.artifacts) ? packManifest.artifacts : [];
  for (const summary of artifactSummaries) {
    artifactRecords.push(loadArtifactMeta(resolvedRoot, summary));
  }

  const sourceBundle = graphSummary?.sourceBundle ?? packManifest?.graphifyRun?.sourceBundleId ?? null;
  return {
    role,
    bundleRoot: resolvedRoot,
    packManifest,
    importSlice,
    candidatePackInput,
    graph,
    graphSummary,
    surfaceMap,
    proposalReport,
    verdict,
    labels,
    evidencePointers,
    rationalePointers,
    artifacts: artifactRecords,
    sourceBundle,
  };
}

function toEvidenceRef(sourceId, excerpt) {
  return {
    sourceKind: "file",
    sourceId,
    authority: "raw_source",
    derivation: "teacher_lint",
    excerpt,
  };
}

function baseEvidenceRefs(currentGraphifyRoots, ocbRoots) {
  const refs = [
    toEvidenceRef("docs/architecture/graphify-bridge.md", "Graphify stays off the serve path and below stronger truth layers."),
    toEvidenceRef("docs/architecture/teacher-v3.md", "Teacher v3 is an off-path compiler of graph structure and compiled artifacts, not an arbiter of current truth."),
    toEvidenceRef("docs/architecture/teacher-v3-proof.md", "Teacher v3 surfaces must remain bounded and labeled by truth layer."),
  ];
  for (const root of currentGraphifyRoots.slice(0, 2)) {
    refs.push(toEvidenceRef(relativeWorkspacePath(path.join(root.bundleRoot, "pack.manifest.json"), defaultWorkspaceRoot), `Current Graphify surface root: ${root.role}`));
  }
  for (const root of ocbRoots.slice(0, 2)) {
    refs.push(toEvidenceRef(relativeWorkspacePath(path.join(root.bundleRoot, "pack.manifest.json"), defaultWorkspaceRoot), `OCB inspectable surface root: ${root.role}`));
  }
  return refs;
}

function surfaceKey(record) {
  return `${record.surfaceId}::${record.kind}`;
}

function rolePriority(role) {
  if (typeof role !== "string") {
    return 0;
  }
  if (role.startsWith("ocb_promoted") || role === "promoted") {
    return 3;
  }
  if (role.startsWith("ocb_compiled") || role === "compiled") {
    return 2;
  }
  if (role.startsWith("ocb_candidate") || role === "candidate") {
    return 1;
  }
  return 0;
}

function choosePreferredRecordMap(records) {
  const map = new Map();
  for (const record of records) {
    const existing = map.get(record.surfaceId) ?? null;
    if (existing === null || rolePriority(record.bundleRole) >= rolePriority(existing.bundleRole)) {
      map.set(record.surfaceId, record);
    }
  }
  return map;
}

function buildCurrentSurfaceIndex(snapshot) {
  const current = [];

  for (const artifact of snapshot.artifacts) {
    current.push({
      origin: "current",
      bundleRole: snapshot.role,
      bundleRoot: snapshot.bundleRoot,
      surfaceId: artifact.artifactId,
      kind: artifact.kind,
      state: artifact.status ?? "current",
      title: artifact.title,
      subjectIds: Array.isArray(artifact.subjectIds) ? [...artifact.subjectIds] : [],
      evidence: Array.isArray(artifact.evidence) ? [...artifact.evidence] : [],
      claims: Array.isArray(artifact.claims) ? [...artifact.claims] : [],
      sourceRoots: Array.isArray(artifact.provenance?.sourceRoots) ? [...artifact.provenance.sourceRoots] : [],
      sourceBundleId: artifact.provenance?.sourceBundleId ?? null,
      sourceBundleHash: artifact.provenance?.sourceBundleHash ?? null,
      graphHash: artifact.provenance?.graphHash ?? null,
      contentHash: artifact.contentHash ?? null,
      markdownPath: artifact.markdownPath,
      metaPath: artifact.metaPath,
      sourceArtifactId: artifact.artifactId,
    });
  }

  if (snapshot.importSlice) {
    const hubPriors = Array.isArray(snapshot.importSlice.hubPriors) ? snapshot.importSlice.hubPriors : [];
    const neighborhoodPriors = Array.isArray(snapshot.importSlice.neighborhoodPriors) ? snapshot.importSlice.neighborhoodPriors : [];
    for (const prior of [...hubPriors, ...neighborhoodPriors]) {
      current.push({
        origin: "current",
        bundleRole: snapshot.role,
        bundleRoot: snapshot.bundleRoot,
        surfaceId: prior.priorId,
        kind: prior.kind ?? "prior",
        state: "current",
        title: prior.title ?? prior.label ?? prior.priorId,
        label: prior.label ?? prior.title ?? prior.priorId,
        subjectIds: Array.isArray(prior.subjectIds) ? [...prior.subjectIds] : [],
        evidencePointerIds: Array.isArray(prior.evidencePointerIds) ? [...prior.evidencePointerIds] : [],
        rationalePointerIds: Array.isArray(prior.rationalePointerIds) ? [...prior.rationalePointerIds] : [],
        sourceRoots: Array.isArray(prior.sourceRoots) ? [...prior.sourceRoots] : [],
        sourceBundleId: prior.sourceBundleId ?? snapshot.importSlice.sourceBundleId ?? null,
        sourceBundleHash: prior.sourceBundleHash ?? snapshot.importSlice.sourceBundleHash ?? null,
        sourceArtifactId: prior.sourceArtifactId ?? null,
        sourceArtifactPath: prior.sourceArtifactPath ?? null,
        sourceMetaPath: prior.sourceMetaPath ?? null,
      });
    }
  }

  if (snapshot.surfaceMap) {
    const controlSurfaces = Array.isArray(snapshot.surfaceMap.controlSurfaces) ? snapshot.surfaceMap.controlSurfaces : [];
    const sourceTruthAnchors = Array.isArray(snapshot.surfaceMap.sourceTruthAnchors) ? snapshot.surfaceMap.sourceTruthAnchors : [];
    for (const surface of controlSurfaces) {
      current.push({
        origin: "current",
        bundleRole: snapshot.role,
        bundleRoot: snapshot.bundleRoot,
        surfaceId: surface.id,
        kind: surface.kind ?? "proposal_truth",
        state: surface.state ?? "current",
        title: surface.note ?? surface.id,
        source: surface.source ?? null,
      });
    }
    for (const anchor of sourceTruthAnchors) {
      current.push({
        origin: "current",
        bundleRole: snapshot.role,
        bundleRoot: snapshot.bundleRoot,
        surfaceId: anchor.id,
        kind: anchor.kind ?? "source_truth_anchor",
        state: anchor.state ?? "current",
        title: anchor.source ?? anchor.id,
        source: anchor.source ?? null,
      });
    }
  }

  return uniqueBy(current, surfaceKey);
}

function buildOcbSurfaceIndex(snapshot) {
  const ocb = [];
  for (const artifact of snapshot.artifacts) {
    ocb.push({
      origin: "ocb",
      bundleRole: snapshot.role,
      bundleRoot: snapshot.bundleRoot,
      surfaceId: artifact.artifactId,
      kind: artifact.kind,
      state: artifact.status ?? "current",
      title: artifact.title,
      subjectIds: Array.isArray(artifact.subjectIds) ? [...artifact.subjectIds] : [],
      evidence: Array.isArray(artifact.evidence) ? [...artifact.evidence] : [],
      claims: Array.isArray(artifact.claims) ? [...artifact.claims] : [],
      sourceRoots: Array.isArray(artifact.provenance?.sourceRoots) ? [...artifact.provenance.sourceRoots] : [],
      sourceBundleId: artifact.provenance?.sourceBundleId ?? null,
      sourceBundleHash: artifact.provenance?.sourceBundleHash ?? null,
      graphHash: artifact.provenance?.graphHash ?? null,
      contentHash: artifact.contentHash ?? null,
      markdownPath: artifact.markdownPath,
      metaPath: artifact.metaPath,
      sourceArtifactId: artifact.artifactId,
    });
  }
  if (snapshot.importSlice) {
    const hubPriors = Array.isArray(snapshot.importSlice.hubPriors) ? snapshot.importSlice.hubPriors : [];
    const neighborhoodPriors = Array.isArray(snapshot.importSlice.neighborhoodPriors) ? snapshot.importSlice.neighborhoodPriors : [];
    for (const prior of [...hubPriors, ...neighborhoodPriors]) {
      ocb.push({
        origin: "ocb",
        bundleRole: snapshot.role,
        bundleRoot: snapshot.bundleRoot,
        surfaceId: prior.priorId,
        kind: prior.kind ?? "prior",
        state: "current",
        title: prior.title ?? prior.label ?? prior.priorId,
        label: prior.label ?? prior.title ?? prior.priorId,
        subjectIds: Array.isArray(prior.subjectIds) ? [...prior.subjectIds] : [],
        evidencePointerIds: Array.isArray(prior.evidencePointerIds) ? [...prior.evidencePointerIds] : [],
        rationalePointerIds: Array.isArray(prior.rationalePointerIds) ? [...prior.rationalePointerIds] : [],
        sourceRoots: Array.isArray(prior.sourceRoots) ? [...prior.sourceRoots] : [],
        sourceBundleId: prior.sourceBundleId ?? snapshot.importSlice.sourceBundleId ?? null,
        sourceBundleHash: prior.sourceBundleHash ?? snapshot.importSlice.sourceBundleHash ?? null,
        sourceArtifactId: prior.sourceArtifactId ?? null,
        sourceArtifactPath: prior.sourceArtifactPath ?? null,
        sourceMetaPath: prior.sourceMetaPath ?? null,
      });
    }
  }
  if (snapshot.surfaceMap) {
    const controlSurfaces = Array.isArray(snapshot.surfaceMap.controlSurfaces) ? snapshot.surfaceMap.controlSurfaces : [];
    const sourceTruthAnchors = Array.isArray(snapshot.surfaceMap.sourceTruthAnchors) ? snapshot.surfaceMap.sourceTruthAnchors : [];
    for (const surface of controlSurfaces) {
      ocb.push({
        origin: "ocb",
        bundleRole: snapshot.role,
        bundleRoot: snapshot.bundleRoot,
        surfaceId: surface.id,
        kind: surface.kind ?? "proposal_truth",
        state: surface.state ?? "current",
        title: surface.note ?? surface.id,
        source: surface.source ?? null,
      });
    }
    for (const anchor of sourceTruthAnchors) {
      ocb.push({
        origin: "ocb",
        bundleRole: snapshot.role,
        bundleRoot: snapshot.bundleRoot,
        surfaceId: anchor.id,
        kind: anchor.kind ?? "source_truth_anchor",
        state: anchor.state ?? "current",
        title: anchor.source ?? anchor.id,
        source: anchor.source ?? null,
      });
    }
  }
  return uniqueBy(ocb, surfaceKey);
}

function buildCurrentGraphRootsSummary(currentRoots) {
  return currentRoots.map((root) => ({
    role: root.role,
    bundleRoot: root.bundleRoot,
    relativePath: relativeWorkspacePath(root.bundleRoot, defaultWorkspaceRoot),
  }));
}

function buildOcbGraphRootsSummary(ocbRoots) {
  return ocbRoots.map((root) => ({
    role: root.role,
    bundleRoot: root.bundleRoot,
    relativePath: relativeWorkspacePath(root.bundleRoot, defaultWorkspaceRoot),
  }));
}

function matchingOcbRecord(currentRecord, ocbById) {
  return ocbById.get(currentRecord.surfaceId) ?? null;
}

function recordEvidenceSupport(record) {
  const evidenceIds = Array.isArray(record.evidence) ? record.evidence.map((item) => item?.evidenceId).filter((value) => typeof value === "string" && value.trim().length > 0) : [];
  const evidenceWithSourceHash = Array.isArray(record.evidence)
    ? record.evidence.filter((item) => item && typeof item === "object" && typeof item.sourceHash === "string" && item.sourceHash.trim().length > 0).length
    : 0;
  return { evidenceIds, evidenceWithSourceHash };
}

function buildMissingFromOcbFindings(currentRecords, ocbById, currentRootSummaries, ocbRootSummaries) {
  const findings = [];
  for (const record of currentRecords) {
    const ocbRecord = matchingOcbRecord(record, ocbById);
    if (ocbRecord !== null) {
      continue;
    }
    findings.push({
      surfaceId: record.surfaceId,
      kind: record.kind,
      title: record.title ?? record.label ?? record.surfaceId,
      reason: `current ${record.kind} surface is not represented in the OCB inspectable surfaces`,
      sourcePaths: [record.markdownPath, record.metaPath, record.sourceArtifactPath, record.sourceMetaPath].filter((value) => typeof value === "string" && value.trim().length > 0),
      sourceBundleId: record.sourceBundleId ?? null,
      sourceBundleHash: record.sourceBundleHash ?? null,
      bundleRole: record.bundleRole,
      currentRoots: currentRootSummaries,
      ocbRoots: ocbRootSummaries,
      evidenceRefs: [
        toEvidenceRef(relativeWorkspacePath(record.metaPath ?? record.markdownPath ?? record.sourceMetaPath ?? record.sourceArtifactPath ?? `${record.surfaceId}`, defaultWorkspaceRoot), `Current surface ${record.surfaceId}`),
        toEvidenceRef("docs/architecture/graphify-bridge.md", "Graphify outputs are derived and review-only."),
      ],
    });
  }
  return findings;
}

function buildStaleInOcbFindings(currentRecords, ocbRecords, currentById, ocbRootSummaries, currentRootSummaries) {
  const findings = [];
  for (const ocbRecord of ocbRecords) {
    const currentRecord = currentById.get(ocbRecord.surfaceId) ?? null;
    if (currentRecord === null) {
      findings.push({
        surfaceId: ocbRecord.surfaceId,
        kind: ocbRecord.kind,
        title: ocbRecord.title ?? ocbRecord.label ?? ocbRecord.surfaceId,
        reason: `OCB surface ${ocbRecord.surfaceId} is absent from the current Graphify surfaces`,
        sourcePaths: [ocbRecord.markdownPath, ocbRecord.metaPath, ocbRecord.sourceArtifactPath, ocbRecord.sourceMetaPath].filter((value) => typeof value === "string" && value.trim().length > 0),
        sourceBundleId: ocbRecord.sourceBundleId ?? null,
        sourceBundleHash: ocbRecord.sourceBundleHash ?? null,
        bundleRole: ocbRecord.bundleRole,
        ocbRoots: ocbRootSummaries,
        currentRoots: currentRootSummaries,
        evidenceRefs: [
          toEvidenceRef(relativeWorkspacePath(ocbRecord.metaPath ?? ocbRecord.markdownPath ?? ocbRecord.sourceMetaPath ?? ocbRecord.sourceArtifactPath ?? `${ocbRecord.surfaceId}`, defaultWorkspaceRoot), `OCB surface ${ocbRecord.surfaceId}`),
          toEvidenceRef("docs/architecture/teacher-v3.md", "OCB inspectable surfaces remain below runtime/proof/docs truth."),
        ],
      });
      continue;
    }
    const hashMismatch = normalizeText(currentRecord.contentHash) !== normalizeText(ocbRecord.contentHash)
      || normalizeText(currentRecord.sourceBundleHash) !== normalizeText(ocbRecord.sourceBundleHash)
      || normalizeText(currentRecord.graphHash) !== normalizeText(ocbRecord.graphHash)
      || JSON.stringify(currentRecord.subjectIds ?? []) !== JSON.stringify(ocbRecord.subjectIds ?? []);
    if (hashMismatch) {
      findings.push({
        surfaceId: ocbRecord.surfaceId,
        kind: ocbRecord.kind,
        title: ocbRecord.title ?? ocbRecord.label ?? ocbRecord.surfaceId,
        reason: `OCB surface ${ocbRecord.surfaceId} is stale relative to current Graphify metadata or content`,
        sourcePaths: [ocbRecord.metaPath, ocbRecord.markdownPath, currentRecord.metaPath, currentRecord.markdownPath].filter((value) => typeof value === "string" && value.trim().length > 0),
        currentContentHash: currentRecord.contentHash ?? null,
        ocbContentHash: ocbRecord.contentHash ?? null,
        currentSourceBundleHash: currentRecord.sourceBundleHash ?? null,
        ocbSourceBundleHash: ocbRecord.sourceBundleHash ?? null,
        bundleRole: ocbRecord.bundleRole,
        ocbRoots: ocbRootSummaries,
        currentRoots: currentRootSummaries,
        evidenceRefs: [
          toEvidenceRef(relativeWorkspacePath(ocbRecord.metaPath ?? ocbRecord.markdownPath ?? `${ocbRecord.surfaceId}`, defaultWorkspaceRoot), `OCB stale surface ${ocbRecord.surfaceId}`),
          toEvidenceRef(relativeWorkspacePath(currentRecord.metaPath ?? currentRecord.markdownPath ?? `${currentRecord.surfaceId}`, defaultWorkspaceRoot), `Current Graphify surface ${currentRecord.surfaceId}`),
        ],
      });
    }
  }
  return findings;
}

function buildCandidateOnlyEdgeFindings(currentRecords) {
  const findings = [];
  for (const record of currentRecords) {
    if (!Array.isArray(record.claims) || record.claims.length === 0) {
      continue;
    }
    for (const claim of record.claims) {
      const claimEvidenceIds = Array.isArray(claim?.evidenceIds)
        ? claim.evidenceIds.filter((value) => typeof value === "string" && value.trim().length > 0)
        : [];
      if (claimEvidenceIds.length === 0) {
        findings.push({
          edgeId: `${record.surfaceId}:${claim?.claimId ?? "claim"}`,
          edgeKind: "claim_to_evidence",
          sourceSurfaceId: record.surfaceId,
          targetSurfaceId: claim?.claimId ?? null,
          title: claim?.text ?? claim?.claimId ?? record.surfaceId,
          reason: `claim ${claim?.claimId ?? "unknown"} carries no evidence support`,
          sourcePaths: [record.metaPath, record.markdownPath].filter((value) => typeof value === "string" && value.trim().length > 0),
          evidenceRefs: [
            toEvidenceRef(relativeWorkspacePath(record.metaPath ?? `${record.surfaceId}`, defaultWorkspaceRoot), `Claim without evidence support on ${record.surfaceId}`),
            toEvidenceRef("docs/architecture/compiled-artifacts.md", "Compiled artifacts must keep evidence refs explicit."),
          ],
        });
        continue;
      }
      const evidenceById = new Map(Array.isArray(record.evidence) ? record.evidence.map((item) => [item?.evidenceId ?? item?.sourceId, item]) : []);
      for (const evidenceId of claimEvidenceIds) {
        const evidence = evidenceById.get(evidenceId) ?? null;
        if (evidence === null || !normalizeText(evidence.sourceHash)) {
          findings.push({
            edgeId: `${record.surfaceId}:${claim?.claimId ?? "claim"}:${evidenceId}`,
            edgeKind: "claim_to_evidence",
            sourceSurfaceId: record.surfaceId,
            targetSurfaceId: evidenceId,
            title: claim?.text ?? claim?.claimId ?? record.surfaceId,
            reason: `claim ${claim?.claimId ?? "unknown"} points at unsupported evidence ${evidenceId}`,
            sourcePaths: [record.metaPath, record.markdownPath].filter((value) => typeof value === "string" && value.trim().length > 0),
            evidenceRefs: [
              toEvidenceRef(relativeWorkspacePath(record.metaPath ?? `${record.surfaceId}`, defaultWorkspaceRoot), `Unsupported claim edge on ${record.surfaceId}`),
              toEvidenceRef("docs/architecture/teacher-v3.md", "Teacher v3 keeps derivation below authority and requires bounded evidence."),
            ],
          });
        }
      }
    }

    if (Array.isArray(record.evidence)) {
      for (const evidence of record.evidence) {
        const evidenceId = evidence?.evidenceId ?? evidence?.sourceId ?? null;
        if (evidenceId === null || !normalizeText(evidence.sourceHash) || !normalizeText(evidence.sourceId)) {
          findings.push({
            edgeId: `${record.surfaceId}:${evidenceId ?? "evidence"}`,
            edgeKind: "artifact_to_evidence",
            sourceSurfaceId: record.surfaceId,
            targetSurfaceId: evidenceId,
            title: record.title ?? record.surfaceId,
            reason: `evidence ref on ${record.surfaceId} is missing source support`,
            sourcePaths: [record.metaPath, record.markdownPath].filter((value) => typeof value === "string" && value.trim().length > 0),
            evidenceRefs: [
              toEvidenceRef(relativeWorkspacePath(record.metaPath ?? `${record.surfaceId}`, defaultWorkspaceRoot), `Unsupported evidence on ${record.surfaceId}`),
              toEvidenceRef("docs/architecture/graphify-bridge.md", "Graphify outputs stay derived and provenance-first."),
            ],
          });
        }
      }
    }

  }
  return findings;
}

function overlapRatio(left, right) {
  const leftSet = new Set((left ?? []).filter((value) => typeof value === "string" && value.trim().length > 0));
  const rightSet = new Set((right ?? []).filter((value) => typeof value === "string" && value.trim().length > 0));
  if (leftSet.size === 0 || rightSet.size === 0) {
    return 0;
  }
  let overlap = 0;
  for (const value of leftSet) {
    if (rightSet.has(value)) {
      overlap += 1;
    }
  }
  return overlap / Math.min(leftSet.size, rightSet.size);
}

function buildCurrentSourceHubFindings(currentRecords, ocbRecords, currentRootSummaries, ocbRootSummaries) {
  const ocbSurfaceIds = new Set(ocbRecords.map((record) => record.surfaceId));
  const ocbSubjects = new Set(ocbRecords.flatMap((record) => Array.isArray(record.subjectIds) ? record.subjectIds : []));
  const findings = [];
  for (const record of currentRecords) {
    const isHubLike = ["hub_prior", "map_of_territory", "concept_page"].includes(record.kind);
    if (!isHubLike) {
      continue;
    }
    const subjectIds = Array.isArray(record.subjectIds) ? record.subjectIds : [];
    const isNewSubjectSpace = subjectIds.some((subjectId) => !ocbSubjects.has(subjectId));
    if (isNewSubjectSpace || !ocbSurfaceIds.has(record.surfaceId)) {
      findings.push({
        surfaceId: record.surfaceId,
        kind: record.kind,
        title: record.title ?? record.label ?? record.surfaceId,
        reason: `current source hub ${record.surfaceId} is new relative to OCB subject/surface coverage`,
        subjectIds,
        sourcePaths: [record.metaPath, record.markdownPath, record.sourceArtifactPath, record.sourceMetaPath].filter((value) => typeof value === "string" && value.trim().length > 0),
        currentRoots: currentRootSummaries,
        ocbRoots: ocbRootSummaries,
        evidenceRefs: [
          toEvidenceRef(relativeWorkspacePath(record.metaPath ?? record.sourceMetaPath ?? record.markdownPath ?? `${record.surfaceId}`, defaultWorkspaceRoot), `Current source hub ${record.surfaceId}`),
          toEvidenceRef("docs/architecture/teacher-v3.md", "Teacher v3 keeps derived hubs subordinate to stronger truth layers."),
        ],
      });
    }
  }
  return findings;
}

function buildProvenanceGapFindings(currentRecords, ocbRecords) {
  const findings = [];
  for (const record of [...currentRecords, ...ocbRecords]) {
    const provenanceMissing = ["map_of_territory", "concept_page", "neighborhood_summary", "provenance_gap_report", "hub_prior", "neighborhood_prior", "candidate_pack_input"].includes(record.kind)
      && (!normalizeText(record.sourceBundleHash) || (!normalizeText(record.sourceBundleId) && !normalizeText(record.sourceArtifactId)));
    const evidenceMissingHash = Array.isArray(record.evidence) && record.evidence.some((evidence) => !normalizeText(evidence?.sourceHash) || !normalizeText(evidence?.sourceId));
    const pointerMissingHash = Array.isArray(record.evidencePointerIds) && record.evidencePointerIds.length > 0 && !normalizeText(record.sourceBundleHash);
    if (!(provenanceMissing || evidenceMissingHash || pointerMissingHash)) {
      continue;
    }
    findings.push({
      surfaceId: record.surfaceId,
      kind: record.kind,
      title: record.title ?? record.label ?? record.surfaceId,
      reason: provenanceMissing
        ? `surface ${record.surfaceId} is missing provenance fields`
        : `surface ${record.surfaceId} has evidence or pointer entries without source hashes`,
      sourcePaths: [record.metaPath, record.markdownPath, record.sourceMetaPath, record.sourceArtifactPath].filter((value) => typeof value === "string" && value.trim().length > 0),
      sourceBundleId: record.sourceBundleId ?? null,
      sourceBundleHash: record.sourceBundleHash ?? null,
      evidenceRefs: [
        toEvidenceRef(relativeWorkspacePath(record.metaPath ?? record.sourceMetaPath ?? record.markdownPath ?? `${record.surfaceId}`, defaultWorkspaceRoot), `Provenance gap candidate ${record.surfaceId}`),
        toEvidenceRef("docs/architecture/compiled-artifacts.md", "Compiled artifacts require explicit provenance metadata."),
      ],
    });
  }
  return findings;
}

function buildMergeSplitHints(currentRecords, ocbRecords) {
  const hints = [];
  const hubLikeCurrent = currentRecords.filter((record) => ["hub_prior", "map_of_territory", "concept_page"].includes(record.kind));
  for (let leftIndex = 0; leftIndex < hubLikeCurrent.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < hubLikeCurrent.length; rightIndex += 1) {
      const left = hubLikeCurrent[leftIndex];
      const right = hubLikeCurrent[rightIndex];
      const ratio = overlapRatio(left.subjectIds, right.subjectIds);
      const titleMatch = normalizeText(left.title) && normalizeText(right.title)
        ? left.title.toLowerCase().split(/\s+/u)[0] === right.title.toLowerCase().split(/\s+/u)[0]
        : false;
      if (ratio >= 0.5 || (ratio > 0 && titleMatch)) {
        hints.push({
          hintId: `merge:${left.surfaceId}:${right.surfaceId}`,
          hintKind: "merge",
          leftSurfaceId: left.surfaceId,
          rightSurfaceId: right.surfaceId,
          summary: `Consider merging ${left.surfaceId} and ${right.surfaceId} because their subject coverage overlaps`,
          overlapRatio: Number(ratio.toFixed(3)),
          sourcePaths: [left.metaPath, right.metaPath, left.markdownPath, right.markdownPath].filter((value) => typeof value === "string" && value.trim().length > 0),
          evidenceRefs: [
            toEvidenceRef(relativeWorkspacePath(left.metaPath ?? `${left.surfaceId}`, defaultWorkspaceRoot), `Merge hint left ${left.surfaceId}`),
            toEvidenceRef(relativeWorkspacePath(right.metaPath ?? `${right.surfaceId}`, defaultWorkspaceRoot), `Merge hint right ${right.surfaceId}`),
            toEvidenceRef("docs/architecture/graphify-bridge.md", "Graphify review surfaces can propose merges and splits without mutating live state."),
          ],
        });
        break;
      }
    }
  }

  const ocbHubLike = ocbRecords.filter((record) => ["hub_prior", "map_of_territory", "concept_page"].includes(record.kind));
  for (const ocbRecord of ocbHubLike) {
    const overlaps = hubLikeCurrent.filter((current) => overlapRatio(current.subjectIds, ocbRecord.subjectIds) > 0.25);
    if (overlaps.length >= 2) {
      hints.push({
        hintId: `split:${ocbRecord.surfaceId}`,
        hintKind: "split",
        surfaceId: ocbRecord.surfaceId,
        summary: `Consider splitting ${ocbRecord.surfaceId} because multiple current hubs overlap its subject space`,
        overlapSurfaceIds: overlaps.map((record) => record.surfaceId),
        sourcePaths: [ocbRecord.metaPath, ocbRecord.markdownPath].filter((value) => typeof value === "string" && value.trim().length > 0),
        evidenceRefs: [
          toEvidenceRef(relativeWorkspacePath(ocbRecord.metaPath ?? `${ocbRecord.surfaceId}`, defaultWorkspaceRoot), `Split hint on ${ocbRecord.surfaceId}`),
          toEvidenceRef("docs/architecture/teacher-v3.md", "Structural proposals can review merge and split candidates off-path."),
        ],
      });
    }
  }
  return hints;
}

function capFindingList(list, limit = 8) {
  return {
    items: list.slice(0, limit),
    truncated: list.length > limit,
    total: list.length,
  };
}

function buildSummaryMarkdown(report) {
  const lines = [
    "# Graphify × OCB maintenance diff",
    "",
    `- diff id: \`${report.diffId}\``,
    `- proposal id: \`${report.proposalId}\``,
    `- graphify root: \`${report.graphifyRoot}\``,
    `- OCB root: \`${report.ocbRoot}\``,
    `- current roots: ${report.currentBundleRoots.length}`,
    `- OCB roots: ${report.ocbBundleRoots.length}`,
    `- verdict: **${report.verdict.verdict}** (${report.verdict.severity})`,
    `- current surfaces: ${report.counts.currentSurfaceCount}`,
    `- OCB surfaces: ${report.counts.ocbSurfaceCount}`,
    "",
    "## Output classes",
    `- missing_from_ocb: ${report.counts.missing_from_ocb}`,
    `- stale_in_ocb: ${report.counts.stale_in_ocb}`,
    `- candidate_only_edges_without_source_support: ${report.counts.candidate_only_edges_without_source_support}`,
    `- new_current_source_hubs: ${report.counts.new_current_source_hubs}`,
    `- provenance_gap_candidates: ${report.counts.provenance_gap_candidates}`,
    `- possible merge/split review hints: ${report.counts.possible_merge_split_review_hints}`,
    "",
  ];

  const classSections = [
    ["missing_from_ocb", report.findings.missing_from_ocb.items],
    ["stale_in_ocb", report.findings.stale_in_ocb.items],
    ["candidate_only_edges_without_source_support", report.findings.candidate_only_edges_without_source_support.items],
    ["new_current_source_hubs", report.findings.new_current_source_hubs.items],
    ["provenance_gap_candidates", report.findings.provenance_gap_candidates.items],
    ["possible merge/split review hints", report.findings.possible_merge_split_review_hints.items],
  ];
  for (const [label, items] of classSections) {
    lines.push(`## ${label}`);
    if (items.length === 0) {
      lines.push("- none");
    } else {
      for (const item of items.slice(0, 6)) {
        const id = item.surfaceId ?? item.edgeId ?? item.hintId ?? "unknown";
        const title = item.title ?? item.summary ?? item.reason ?? id;
        lines.push(`- \`${id}\` — ${title}`);
      }
      if (items.length > 6) {
        lines.push(`- … ${items.length - 6} more`);
      }
    }
    lines.push("");
  }

  lines.push("## Proposal-suggestion posture");
  lines.push("This lane only emits bounded diagnostics and proposal suggestions. It does not mutate live or candidate graph state.");
  lines.push("");
  return lines.join("\n");
}

function buildProposalSuggestion(report) {
  const suggestions = [];
  const suggestionSpecs = [
    ["missing_from_ocb", "Add or explicitly exclude missing current surfaces from OCB inspectable bundles."],
    ["stale_in_ocb", "Refresh OCB inspectable bundles or correct the stale promotion surface."],
    ["candidate_only_edges_without_source_support", "Bind the candidate edge to source evidence or downgrade it to review-only."],
    ["new_current_source_hubs", "Review new current source hubs for promotion or merge into an existing hub."],
    ["provenance_gap_candidates", "Fill provenance fields before any later promotion or import claim."],
    ["possible_merge_split_review_hints", "Review overlapping hubs for merge/split before widening the surface."],
  ];
  for (const [code, summary] of suggestionSpecs) {
    const items = report.findings[code]?.items ?? [];
    if (items.length === 0) {
      continue;
    }
    const sample = items.slice(0, 3).map((item) => item.surfaceId ?? item.edgeId ?? item.hintId ?? "unknown");
    suggestions.push({
      suggestionId: `suggest:${code}:${slugify(report.diffId)}`,
      code,
      summary,
      rationale: items[0].reason ?? items[0].summary ?? summary,
      sampleIds: sample,
      confidence: code === "provenance_gap_candidates" ? 0.96 : 0.9,
      rollbackKey: report.rollbackKey,
      reviewMode: "proposal_only",
      targetStateOnly: true,
      evidenceRefs: items[0].evidenceRefs ?? [],
    });
  }

  return {
    contract: "graphify_ocb_maintenance_diff_proposal_suggestion.v1",
    diffId: report.diffId,
    proposalId: report.proposalId,
    rollbackKey: report.rollbackKey,
    reviewMode: "proposal_only",
    status: Object.values(report.counts).some((value) => Number(value ?? 0) > 0) ? "needs_review" : "clear",
    summary: suggestions.length === 0
      ? "No maintenance suggestions were generated; current Graphify and OCB inspectable surfaces are aligned within the bounded checks used here."
      : `${suggestions.length} proposal suggestions generated from bounded maintenance diff findings.`,
    suggestionCount: suggestions.length,
    suggestions,
    counts: report.counts,
    currentBundleRoots: report.currentBundleRoots,
    ocbBundleRoots: report.ocbBundleRoots,
    createdAt: report.createdAt,
    updatedAt: report.updatedAt,
  };
}

function buildVerdict(report) {
  const findingCount = Object.values(report.counts).reduce((total, value) => total + Number(value ?? 0), 0);
  return {
    contract: "graphify_ocb_maintenance_diff_verdict.v1",
    diffId: report.diffId,
    proposalId: report.proposalId,
    verdict: findingCount > 0 ? "needs_review" : "clear",
    severity: findingCount > 0 ? "warn" : "info",
    findingCount,
    proposalSuggestionCount: report.proposalSuggestion.suggestionCount,
    currentSurfaceCount: report.counts.currentSurfaceCount,
    ocbSurfaceCount: report.counts.ocbSurfaceCount,
    why: findingCount > 0
      ? "bounded maintenance diagnostics identified current-vs-OCB surface drift"
      : "bounded maintenance diagnostics found no surface drift requiring operator attention",
    reviewMode: "proposal_only",
    targetStateOnly: true,
    rollbackKey: report.rollbackKey,
    createdAt: report.createdAt,
    updatedAt: report.updatedAt,
  };
}

function buildBundleDigest(files) {
  const entries = Object.entries(files).sort(([left], [right]) => left.localeCompare(right));
  const digest = createHash("sha256");
  const fileHashes = {};
  for (const [name, text] of entries) {
    const hash = sha256Text(text);
    digest.update(`${name}\0${hash}\n`);
    fileHashes[name] = hash;
  }
  return {
    bundleHash: `sha256:${digest.digest("hex")}`,
    fileCount: entries.length,
    files: fileHashes,
  };
}

function buildReportPayload(options) {
  const repoRoot = path.resolve(options.repoRoot ?? defaultRepoRoot);
  const workspaceRoot = path.resolve(options.workspaceRoot ?? defaultWorkspaceRoot);
  const graphifyRoot = path.resolve(options.graphifyRoot ?? defaultGraphifyRoot);
  const ocbRoot = path.resolve(options.ocbRoot ?? defaultOcbRoot);

  const currentBundleRoots = discoverCurrentBundleRoots(graphifyRoot).map((root, index) => loadBundleSnapshot(root, `current_${index + 1}`));
  const ocbBundleRoots = discoverOcbBundleRoots(ocbRoot).map((root, index) => loadBundleSnapshot(root, `ocb_${index + 1}`));

  if (currentBundleRoots.length === 0) {
    throw new Error(`graphify maintenance diff found no recognizable current Graphify bundle roots under ${graphifyRoot}`);
  }
  if (ocbBundleRoots.length === 0) {
    throw new Error(`graphify maintenance diff found no recognizable OCB inspectable bundle roots under ${ocbRoot}`);
  }

  const currentRecords = uniqueBy(currentBundleRoots.flatMap((snapshot) => buildCurrentSurfaceIndex(snapshot)), surfaceKey);
  const ocbRecords = ocbBundleRoots.flatMap((snapshot) => buildOcbSurfaceIndex(snapshot));

  const currentById = choosePreferredRecordMap(currentRecords);
  const ocbById = choosePreferredRecordMap(ocbRecords);

  const currentRootSummaries = buildCurrentGraphRootsSummary(currentBundleRoots);
  const ocbRootSummaries = buildOcbGraphRootsSummary(ocbBundleRoots);

  const missingFromOcb = buildMissingFromOcbFindings(currentRecords, ocbById, currentRootSummaries, ocbRootSummaries);
  const staleInOcb = buildStaleInOcbFindings(currentRecords, ocbRecords, currentById, ocbRootSummaries, currentRootSummaries);
  const candidateOnlyEdgesWithoutSourceSupport = buildCandidateOnlyEdgeFindings(currentRecords);
  const newCurrentSourceHubs = buildCurrentSourceHubFindings(currentRecords, ocbRecords, currentRootSummaries, ocbRootSummaries);
  const provenanceGapCandidates = buildProvenanceGapFindings(currentRecords, ocbRecords);
  const possibleMergeSplitReviewHints = buildMergeSplitHints(currentRecords, ocbRecords);

  const report = {
    contract: "graphify_ocb_maintenance_diff.v1",
    diffId: normalizeText(options.diffId) ?? `graphify-maintenance-diff-${slugify(options.runId ?? timestampToken(new Date().toISOString()))}`,
    proposalId: normalizeText(options.proposalId) ?? `prop_graphify_maintenance_diff_${slugify(options.runId ?? timestampToken(new Date().toISOString()))}`,
    rollbackKey: normalizeText(options.rollbackKey) ?? `rollback:graphify-maintenance-diff:${slugify(options.runId ?? timestampToken(new Date().toISOString()))}`,
    graphifyRoot: relativeWorkspacePath(graphifyRoot, workspaceRoot),
    ocbRoot: relativeWorkspacePath(ocbRoot, workspaceRoot),
    repoRoot: relativeWorkspacePath(repoRoot, workspaceRoot),
    workspaceRoot: relativeWorkspacePath(workspaceRoot, workspaceRoot),
    currentBundleRoots: currentRootSummaries,
    ocbBundleRoots: ocbRootSummaries,
    counts: {
      currentSurfaceCount: currentRecords.length,
      ocbSurfaceCount: ocbRecords.length,
      missing_from_ocb: missingFromOcb.length,
      stale_in_ocb: staleInOcb.length,
      candidate_only_edges_without_source_support: candidateOnlyEdgesWithoutSourceSupport.length,
      new_current_source_hubs: newCurrentSourceHubs.length,
      provenance_gap_candidates: provenanceGapCandidates.length,
      possible_merge_split_review_hints: possibleMergeSplitReviewHints.length,
    },
    findings: {
      missing_from_ocb: capFindingList(missingFromOcb),
      stale_in_ocb: capFindingList(staleInOcb),
      candidate_only_edges_without_source_support: capFindingList(candidateOnlyEdgesWithoutSourceSupport),
      new_current_source_hubs: capFindingList(newCurrentSourceHubs),
      provenance_gap_candidates: capFindingList(provenanceGapCandidates),
      possible_merge_split_review_hints: capFindingList(possibleMergeSplitReviewHints),
    },
    evidenceRefs: baseEvidenceRefs(currentBundleRoots, ocbBundleRoots),
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    sourceUniverse: {
      currentSurfaceIds: currentRecords.slice(0, 40).map((record) => record.surfaceId),
      ocbSurfaceIds: ocbRecords.slice(0, 40).map((record) => record.surfaceId),
    },
  };

  const proposalSuggestion = buildProposalSuggestion(report);
  const verdict = buildVerdict({ ...report, proposalSuggestion });
  const finalReport = {
    ...report,
    proposalSuggestion,
    verdict,
  };
  finalReport.summary = buildSummaryMarkdown(finalReport);
  return {
    report: finalReport,
    proposalSuggestion,
    verdict,
    currentRecords,
    ocbRecords,
    currentBundleRoots,
    ocbBundleRoots,
    currentRootSummaries,
    ocbRootSummaries,
  };
}

function buildFilesFromPayload(payload) {
  const summary = buildSummaryMarkdown(payload.report);
  payload.report.summary = summary;
  return {
    [GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.maintenanceDiff]: stableJson(payload.report),
    [GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.summary]: summary,
    [GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.proposalSuggestion]: stableJson(payload.proposalSuggestion),
    [GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.verdict]: stableJson(payload.verdict),
  };
}

export function buildGraphifyMaintenanceDiffBundle(options = {}) {
  const repoRoot = path.resolve(options.repoRoot ?? defaultRepoRoot);
  const workspaceRoot = path.resolve(options.workspaceRoot ?? defaultWorkspaceRoot);
  const graphifyRoot = path.resolve(options.graphifyRoot ?? defaultGraphifyRoot);
  const ocbRoot = path.resolve(options.ocbRoot ?? defaultOcbRoot);
  const outputRoot = path.resolve(options.outputRoot ?? defaultOutputRoot);
  const runId = normalizeText(options.runId) ?? `graphify-maintenance-diff-${timestampToken(new Date().toISOString())}`;
  const outputDir = path.join(outputRoot, runId);
  const payload = buildReportPayload({
    ...options,
    repoRoot,
    workspaceRoot,
    graphifyRoot,
    ocbRoot,
    runId,
  });
  const files = buildFilesFromPayload(payload);
  const digest = buildBundleDigest(files);
  payload.report.bundleHash = digest.bundleHash;
  payload.proposalSuggestion.bundleHash = digest.bundleHash;
  payload.verdict.bundleHash = digest.bundleHash;
  const finalFiles = buildFilesFromPayload(payload);
  const finalDigest = buildBundleDigest(finalFiles);
  const paths = {
    maintenanceDiff: path.join(outputDir, GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.maintenanceDiff),
    summary: path.join(outputDir, GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.summary),
    proposalSuggestion: path.join(outputDir, GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.proposalSuggestion),
    verdict: path.join(outputDir, GRAPHIFY_MAINTENANCE_DIFF_LAYOUT_V1.verdict),
  };
  return {
    ok: true,
    runId,
    diffId: payload.report.diffId,
    proposalId: payload.report.proposalId,
    rollbackKey: payload.report.rollbackKey,
    repoRoot,
    workspaceRoot,
    graphifyRoot,
    ocbRoot,
    outputRoot,
    outputDir,
    report: payload.report,
    proposalSuggestion: payload.proposalSuggestion,
    verdict: payload.verdict,
    summary: payload.report.summary,
    files: finalFiles,
    paths,
    digest: finalDigest,
    currentRecords: payload.currentRecords,
    ocbRecords: payload.ocbRecords,
    currentBundleRoots: payload.currentRootSummaries,
    ocbBundleRoots: payload.ocbRootSummaries,
  };
}

export function writeGraphifyMaintenanceDiffBundle(outputDir, bundle) {
  rmSync(outputDir, { recursive: true, force: true });
  ensureDir(outputDir);
  const writtenFiles = [];
  for (const [name, text] of Object.entries(bundle.files)) {
    const filePath = path.join(outputDir, name);
    writeText(filePath, text);
    writtenFiles.push(filePath);
  }
  return {
    writtenFiles,
    fileCount: writtenFiles.length,
  };
}

export function parseGraphifyMaintenanceDiffCliArgs(argv) {
  let graphifyRoot = defaultGraphifyRoot;
  let ocbRoot = defaultOcbRoot;
  let repoRoot = defaultRepoRoot;
  let workspaceRoot = defaultWorkspaceRoot;
  let outputRoot = null;
  let runId = null;
  let help = false;
  let json = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      help = true;
      continue;
    }
    if (arg === "--json") {
      json = true;
      continue;
    }
    if (arg === "--graphify-root" || arg === "--bundle-root") {
      const next = argv[index + 1];
      if (next === undefined) {
        throw new Error(`${arg} requires a value`);
      }
      graphifyRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--ocb-root") {
      const next = argv[index + 1];
      if (next === undefined) {
        throw new Error("--ocb-root requires a value");
      }
      ocbRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--repo-root") {
      const next = argv[index + 1];
      if (next === undefined) {
        throw new Error("--repo-root requires a value");
      }
      repoRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--workspace-root") {
      const next = argv[index + 1];
      if (next === undefined) {
        throw new Error("--workspace-root requires a value");
      }
      workspaceRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--output-root") {
      const next = argv[index + 1];
      if (next === undefined) {
        throw new Error("--output-root requires a value");
      }
      outputRoot = next;
      index += 1;
      continue;
    }
    if (arg === "--run-id") {
      const next = argv[index + 1];
      if (next === undefined) {
        throw new Error("--run-id requires a value");
      }
      runId = next;
      index += 1;
      continue;
    }
    throw new Error(`unknown argument for graphify-maintenance-diff: ${arg}`);
  }
  return {
    command: "graphify-maintenance-diff",
    graphifyRoot: path.resolve(graphifyRoot),
    ocbRoot: path.resolve(ocbRoot),
    repoRoot: path.resolve(repoRoot),
    workspaceRoot: path.resolve(workspaceRoot),
    outputRoot: outputRoot === null ? null : path.resolve(outputRoot),
    runId,
    json,
    help,
  };
}

export function formatGraphifyMaintenanceDiffSummary(result) {
  const lines = [
    "GRAPHIFY MAINTENANCE DIFF ok",
    `  Diff id:             ${result.report.diffId}`,
    `  Proposal:            ${result.report.proposalId}`,
    `  Graphify root:       ${result.report.graphifyRoot}`,
    `  OCB root:            ${result.report.ocbRoot}`,
    `  Current roots:       ${result.currentBundleRoots.length}`,
    `  OCB roots:           ${result.ocbBundleRoots.length}`,
    `  Verdict:             ${result.verdict.verdict} (${result.verdict.severity})`,
    `  Current surfaces:    ${result.report.counts.currentSurfaceCount}`,
    `  OCB surfaces:        ${result.report.counts.ocbSurfaceCount}`,
    `  Output root:         ${result.outputRoot}`,
    `  Report:              ${result.paths.maintenanceDiff}`,
    `  Proposal suggestion: ${result.paths.proposalSuggestion}`,
    `  Verdict file:        ${result.paths.verdict}`,
    "  Findings:",
  ];
  for (const [code, count] of Object.entries(result.report.counts)) {
    if (code === "currentSurfaceCount" || code === "ocbSurfaceCount") {
      continue;
    }
    lines.push(`    - ${code}: ${count}`);
  }
  const topFindingLines = [
    ["missing_from_ocb", result.report.findings.missing_from_ocb.items],
    ["stale_in_ocb", result.report.findings.stale_in_ocb.items],
    ["candidate_only_edges_without_source_support", result.report.findings.candidate_only_edges_without_source_support.items],
    ["new_current_source_hubs", result.report.findings.new_current_source_hubs.items],
    ["provenance_gap_candidates", result.report.findings.provenance_gap_candidates.items],
    ["possible merge/split review hints", result.report.findings.possible_merge_split_review_hints.items],
  ];
  for (const [label, items] of topFindingLines) {
    if (items.length === 0) {
      continue;
    }
    lines.push(`  ${label}: ${items[0].surfaceId ?? items[0].edgeId ?? items[0].hintId ?? "unknown"}`);
  }
  return `${lines.join("\n")}\n`;
}

export function runGraphifyMaintenanceDiff(argvOrOptions = {}) {
  const parsed = Array.isArray(argvOrOptions)
    ? parseGraphifyMaintenanceDiffCliArgs(argvOrOptions)
    : { command: "graphify-maintenance-diff", json: false, help: false, ...argvOrOptions };
  if (parsed.help) {
    return {
      ok: true,
      help: true,
      summary: "",
      report: null,
      proposalSuggestion: null,
      verdict: null,
      paths: null,
      outputRoot: null,
      outputDir: null,
      graphifyRoot: null,
      ocbRoot: null,
      repoRoot: null,
      workspaceRoot: null,
      runId: null,
      diffId: null,
      proposalId: null,
    };
  }
  const result = buildGraphifyMaintenanceDiffBundle({
    graphifyRoot: parsed.graphifyRoot,
    ocbRoot: parsed.ocbRoot,
    repoRoot: parsed.repoRoot,
    workspaceRoot: parsed.workspaceRoot,
    outputRoot: parsed.outputRoot ?? undefined,
    runId: parsed.runId ?? undefined,
  });
  writeGraphifyMaintenanceDiffBundle(result.outputDir, result);
  return {
    ...result,
    json: Boolean(parsed.json),
    summary: formatGraphifyMaintenanceDiffSummary(result),
  };
}

function main() {
  const result = runGraphifyMaintenanceDiff(process.argv.slice(2));
  if (result.help) {
    process.stdout.write([
      "Usage:",
      "  node scripts/graphify-maintenance-diff.mjs --graphify-root <path> --ocb-root <path> [--repo-root <path>] [--workspace-root <path>] [--output-root <path>] [--run-id <id>] [--json]",
      "",
      "This maintenance diff lane emits bounded operator diagnostics and proposal suggestions only.",
    ].join("\n") + "\n");
    return;
  }
  if (result.json) {
    process.stdout.write(`${stableJson({ ok: result.ok, report: result.report, proposalSuggestion: result.proposalSuggestion, verdict: result.verdict, paths: result.paths })}\n`);
    return;
  }
  process.stdout.write(result.summary);
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
