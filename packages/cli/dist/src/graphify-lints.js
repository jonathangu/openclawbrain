#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultRepoRoot = path.resolve(__dirname, "..", "..", "..");
const defaultWorkspaceRoot = path.resolve(defaultRepoRoot, "..");
const defaultOutputRoot = path.join(defaultWorkspaceRoot, "artifacts", "graphify-lints");

export const GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT = {
  deterministicLints: "deterministic-lints.json",
  summary: "summary.md",
  proposalEnvelope: "proposal-envelope.json",
  verdict: "verdict.json",
};

const RELEVANT_RELEASE_SURFACES = new Set([
  "README.md",
  "docs/README.md",
  "docs/END_STATE.md",
  "CHANGELOG.md",
]);

const SHADOW_ONLY_CLASSES = new Set(["mutation", "forgetting", "correction"]);
const PROMOTABLE_CLASSES = new Set(["compiler", "lint"]);

function stableJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function sha256Text(text) {
  return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}

function ensureDir(dirPath) {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

function readTextIfExists(filePath) {
  return existsSync(filePath) ? readFileSync(filePath, "utf8") : null;
}

function readJsonIfExists(filePath) {
  const text = readTextIfExists(filePath);
  return text === null ? null : JSON.parse(text);
}

function writeJson(filePath, value) {
  writeFileSync(filePath, stableJson(value), "utf8");
}

function writeText(filePath, value) {
  writeFileSync(filePath, `${value}\n`, "utf8");
}

function toPosix(value) {
  return String(value).replace(/\\/g, "/");
}

function relativeWorkspacePath(absPath, workspaceRoot) {
  const relative = path.relative(workspaceRoot, absPath);
  return relative.startsWith("..") ? toPosix(absPath) : toPosix(relative);
}

function stripFragment(sourceId) {
  return sourceId.split("#")[0] ?? sourceId;
}

function isLikelyPathRef(value) {
  if (typeof value !== "string") {
    return false;
  }
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return false;
  }
  if (
    trimmed === "README.md" ||
    trimmed === "CHANGELOG.md" ||
    trimmed === "manifest.json" ||
    trimmed === "pack.manifest.json" ||
    trimmed === "summary.md" ||
    trimmed === "status.json" ||
    trimmed === "surface-map.json" ||
    trimmed === "proposal-report.json" ||
    trimmed === "verdict.json" ||
    trimmed === "docs/README.md" ||
    trimmed === "docs/END_STATE.md"
  ) {
    return true;
  }
  const allowedPrefixes = ["docs/", "artifacts/", "scripts/", "openclawbrain/", "./", "../"];
  if (allowedPrefixes.some((prefix) => trimmed.startsWith(prefix)) && /\.[a-z]+$/i.test(trimmed)) {
    return true;
  }
  return false;
}

function candidatePathsForRef(ref, bundleRoot, repoRoot, workspaceRoot) {
  const cleaned = stripFragment(ref);
  const candidates = [];
  if (path.isAbsolute(cleaned)) {
    candidates.push(cleaned);
    return candidates;
  }
  if (cleaned.startsWith("openclawbrain/")) {
    candidates.push(path.join(repoRoot, cleaned.slice("openclawbrain/".length)));
    candidates.push(path.join(workspaceRoot, cleaned.slice("openclawbrain/".length)));
  }
  if (cleaned.startsWith("./")) {
    candidates.push(path.join(bundleRoot, cleaned.slice(2)));
  }
  candidates.push(path.join(bundleRoot, cleaned));
  candidates.push(path.join(repoRoot, cleaned));
  candidates.push(path.join(workspaceRoot, cleaned));
  return [...new Set(candidates)];
}

function resolveRef(ref, bundleRoot, repoRoot, workspaceRoot) {
  const candidates = candidatePathsForRef(ref, bundleRoot, repoRoot, workspaceRoot);
  const resolvedPath = candidates.find((candidate) => existsSync(candidate)) ?? candidates[0] ?? null;
  return {
    sourceRef: ref,
    resolvedPath,
    exists: resolvedPath !== null && existsSync(resolvedPath),
  };
}

function hashFile(filePath) {
  return sha256Text(readFileSync(filePath, "utf8"));
}

function collectPathLikeRefs(value, refs = new Set()) {
  if (value === null || value === undefined) {
    return refs;
  }
  if (typeof value === "string") {
    if (isLikelyPathRef(value.trim())) {
      refs.add(value.trim());
    }
    return refs;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectPathLikeRefs(item, refs);
    }
    return refs;
  }
  if (typeof value === "object") {
    for (const nestedValue of Object.values(value)) {
      collectPathLikeRefs(nestedValue, refs);
    }
  }
  return refs;
}

function collectEvidenceRefsFromObjects(objects, bundleRoot, repoRoot, workspaceRoot) {
  const refs = [];
  const seen = new Set();
  const sourceRefs = new Set();
  for (const object of objects) {
    collectPathLikeRefs(object, sourceRefs);
  }
  for (const sourceRef of sourceRefs) {
    const resolved = resolveRef(sourceRef, bundleRoot, repoRoot, workspaceRoot);
    const key = `${sourceRef}::${resolved.resolvedPath ?? "missing"}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    refs.push({
      evidenceId: `evi_${sha256Text(sourceRef).slice(7, 19)}`,
      sourceKind: sourceRef.includes("/") || /\.[a-z]+$/i.test(sourceRef) ? "file" : "summary",
      sourceId: sourceRef,
      authority: "raw_source",
      derivation: "teacher_lint",
      sourceHash: resolved.exists && resolved.resolvedPath !== null ? hashFile(resolved.resolvedPath) : null,
    });
  }
  return refs;
}

function detectCurrentReleaseVersion(changelogText) {
  for (const line of changelogText.split(/\r?\n/)) {
    const match = /^##\s+((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\s*$/.exec(line);
    if (match) {
      return match[1];
    }
  }
  return null;
}

function readReleaseState(repoRoot) {
  const changelogPath = path.join(repoRoot, "CHANGELOG.md");
  const readmePath = path.join(repoRoot, "README.md");
  const docsReadmePath = path.join(repoRoot, "docs", "README.md");
  const endStatePath = path.join(repoRoot, "docs", "END_STATE.md");
  const readmeText = readTextIfExists(readmePath);
  const docsReadmeText = readTextIfExists(docsReadmePath);
  const endStateText = readTextIfExists(endStatePath);
  const changelogText = readTextIfExists(changelogPath);
  const currentVersion = changelogText === null ? null : detectCurrentReleaseVersion(changelogText);
  const readmeVersion = readmeText === null ? null : (readmeText.match(/Current version:\s+\*\*((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\*\*/m)?.[1] ?? null);
  const docsIndexMatch = docsReadmeText?.match(/-\s+\[Current release notes \(((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\)\]\(release-notes-((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\.md\)/m);
  const docsIndexVersion = docsIndexMatch?.[1] ?? null;
  const docsIndexTargetVersion = docsIndexMatch?.[2] ?? null;
  const endStateMatches = [...(endStateText?.matchAll(/split packages `@openclawbrain\/openclaw@((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))` and `@openclawbrain\/cli@((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))` are published/g) ?? [])].map((match) => [match[1], match[2]]);
  return {
    currentVersion,
    readmeVersion,
    docsIndexVersion,
    docsIndexTargetVersion,
    endStateVersions: endStateMatches,
    releaseNotesPath: currentVersion === null ? null : path.join(repoRoot, "docs", `release-notes-${currentVersion}.md`),
  };
}

function addFinding(findings, finding) {
  findings.push({
    code: finding.code,
    severity: finding.severity ?? "error",
    summary: finding.summary,
    detail: finding.detail ?? null,
    evidenceRefs: finding.evidenceRefs ?? [],
    paths: finding.paths ?? [],
    truthLayer: finding.truthLayer ?? null,
  });
}

function uniqueByKey(items, keyFn) {
  const seen = new Set();
  const unique = [];
  for (const item of items) {
    const key = keyFn(item);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(item);
  }
  return unique;
}

function normalizeBundleObjects(bundleRoot) {
  const summaryPath = path.join(bundleRoot, GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT.summary);
  const statusPath = path.join(bundleRoot, GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT.summary.replace("summary.md", "status.json"));
  const surfaceMapPath = path.join(bundleRoot, "surface-map.json");
  const proposalReportPath = path.join(bundleRoot, "proposal-report.json");
  const verdictPath = path.join(bundleRoot, GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT.verdict);
  const manifestPath = path.join(bundleRoot, "manifest.json");
  const packManifestPath = path.join(bundleRoot, "pack.manifest.json");

  return {
    summaryText: readTextIfExists(summaryPath),
    status: readJsonIfExists(statusPath),
    surfaceMap: readJsonIfExists(surfaceMapPath),
    proposalReport: readJsonIfExists(proposalReportPath),
    verdict: readJsonIfExists(verdictPath),
    manifest: readJsonIfExists(manifestPath),
    packManifest: readJsonIfExists(packManifestPath),
    summaryPath,
    statusPath,
    surfaceMapPath,
    proposalReportPath,
    verdictPath,
    manifestPath,
    packManifestPath,
  };
}

function extractVersionCandidates(objects) {
  const candidates = new Set();
  for (const object of objects) {
    const strings = [];
    const stack = [object];
    while (stack.length > 0) {
      const current = stack.pop();
      if (current === null || current === undefined) {
        continue;
      }
      if (typeof current === "string") {
        strings.push(current);
        continue;
      }
      if (Array.isArray(current)) {
        for (const item of current) {
          stack.push(item);
        }
        continue;
      }
      if (typeof current === "object") {
        for (const [key, value] of Object.entries(current)) {
          if (typeof value === "string" && (key === "version" || key === "currentVersion" || key === "releaseVersion" || key === "docsVersion")) {
            const semverMatch = /((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))/u.exec(value);
            if (semverMatch) {
              candidates.add(semverMatch[1]);
            }
          }
          stack.push(value);
        }
      }
    }
    for (const text of strings) {
      const releaseNotesMatch = /^docs\/release-notes-((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\.md$/.exec(text.trim());
      if (releaseNotesMatch) {
        candidates.add(releaseNotesMatch[1]);
      }
      const subjectMatch = /^release:((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))$/.exec(text.trim());
      if (subjectMatch) {
        candidates.add(subjectMatch[1]);
      }
    }
  }
  return [...candidates];
}

function buildBundleFindings(bundleRoot, repoRoot, workspaceRoot, bundle) {
  const findings = [];
  const bundleObjects = [bundle.manifest, bundle.packManifest, bundle.status, bundle.surfaceMap, bundle.proposalReport, bundle.verdict].filter(Boolean);
  const evidenceRefs = collectEvidenceRefsFromObjects(bundleObjects, bundleRoot, repoRoot, workspaceRoot);
  const bundleId = bundle.proposalReport?.bundleId ?? bundle.status?.bundleId ?? bundle.surfaceMap?.bundleId ?? bundle.manifest?.bundleId ?? bundle.packManifest?.packId ?? path.basename(bundleRoot);
  const proposal = bundle.proposalReport?.proposal ?? bundle.status?.proposal ?? null;
  const proposalId = proposal?.proposalId ?? bundle.proposalReport?.proposalId ?? bundle.status?.proposalId ?? bundle.verdict?.proposalId ?? bundleId;
  const proposalClass = proposal?.proposalClass ?? bundle.proposalReport?.proposalClass ?? bundle.status?.proposalClass ?? bundle.status?.proposalLane ?? bundle.manifest?.proposalClass ?? "lint";
  const reviewMode = proposal?.reviewMode ?? bundle.proposalReport?.reviewMode ?? bundle.status?.reviewMode ?? bundle.verdict?.reviewMode ?? "reviewable";
  const proposalStatus = proposal?.status ?? bundle.proposalReport?.status ?? bundle.status?.proposalStatus ?? bundle.verdict?.verdict ?? "reviewable";

  const mandatoryFiles = [
    { label: "summary.md", path: bundle.summaryPath },
    { label: "status.json", path: bundle.statusPath },
    { label: "surface-map.json", path: bundle.surfaceMapPath },
    { label: "proposal-report.json", path: bundle.proposalReportPath },
    { label: "verdict.json", path: bundle.verdictPath },
  ];
  const missingMandatory = mandatoryFiles.filter((item) => !existsSync(item.path));
  if (missingMandatory.length > 0) {
    addFinding(findings, {
      code: "missing_source_files",
      summary: `${missingMandatory.length} required bundle files are missing`,
      detail: missingMandatory.map((item) => item.label).join(", "),
      evidenceRefs: [
        { sourceId: "docs/architecture/teacher-v3-lints.md", evidenceId: "evi_teacher_v3_lints", authority: "raw_source", derivation: "teacher_lint" },
        { sourceId: "docs/architecture/teacher-v3-proof.md", evidenceId: "evi_teacher_v3_proof", authority: "raw_source", derivation: "teacher_lint" },
      ],
      paths: missingMandatory.map((item) => relativeWorkspacePath(item.path, workspaceRoot)),
      truthLayer: "bundle_surface",
    });
  }

  const sourceRefs = uniqueByKey([...collectPathLikeRefs(bundle.manifest), ...collectPathLikeRefs(bundle.packManifest), ...collectPathLikeRefs(bundle.status), ...collectPathLikeRefs(bundle.surfaceMap), ...collectPathLikeRefs(bundle.proposalReport), ...collectPathLikeRefs(bundle.verdict)], (value) => value);
  const resolvedRefs = sourceRefs.map((sourceRef) => resolveRef(sourceRef, bundleRoot, repoRoot, workspaceRoot));
  const missingRefs = resolvedRefs.filter((ref) => !ref.exists);
  if (missingRefs.length > 0) {
    addFinding(findings, {
      code: "missing_source_files",
      summary: `${missingRefs.length} referenced source files are missing`,
      detail: missingRefs.slice(0, 8).map((ref) => ref.sourceRef).join(", "),
      evidenceRefs: evidenceRefs.slice(0, 4),
      paths: missingRefs.map((ref) => relativeWorkspacePath(ref.resolvedPath ?? ref.sourceRef, workspaceRoot)),
      truthLayer: "bundle_manifest",
    });
  }

  const hashDrifts = [];
  for (const manifest of [bundle.manifest, bundle.packManifest].filter(Boolean)) {
    const artifactEntries = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
    for (const artifact of artifactEntries) {
      const markdownPath = typeof artifact.markdownPath === "string" ? resolveRef(artifact.markdownPath, bundleRoot, repoRoot, workspaceRoot) : null;
      const metaPath = typeof artifact.metaPath === "string" ? resolveRef(artifact.metaPath, bundleRoot, repoRoot, workspaceRoot) : null;
      if (markdownPath?.exists && typeof artifact.contentHash === "string") {
        const actualHash = hashFile(markdownPath.resolvedPath);
        if (actualHash !== artifact.contentHash) {
          hashDrifts.push({
            label: artifact.artifactId ?? artifact.kind ?? artifact.markdownPath,
            path: markdownPath.resolvedPath,
            expected: artifact.contentHash,
            actual: actualHash,
          });
        }
      }
      if (metaPath?.exists) {
        const meta = readJsonIfExists(metaPath.resolvedPath);
        if (meta && typeof meta.contentHash === "string" && markdownPath?.exists) {
          const actualHash = hashFile(markdownPath.resolvedPath);
          if (meta.contentHash !== actualHash) {
            hashDrifts.push({
              label: artifact.artifactId ?? artifact.kind ?? artifact.markdownPath,
              path: markdownPath.resolvedPath,
              expected: meta.contentHash,
              actual: actualHash,
            });
          }
        }
        if (meta && typeof artifact.contentHash === "string" && typeof meta.contentHash === "string" && meta.contentHash !== artifact.contentHash) {
          hashDrifts.push({
            label: artifact.artifactId ?? artifact.kind ?? artifact.markdownPath,
            path: metaPath.resolvedPath,
            expected: artifact.contentHash,
            actual: meta.contentHash,
          });
        }
      }
    }
    if (typeof manifest.contentHash === "string") {
      const manifestPath = manifest === bundle.manifest ? bundle.manifestPath : bundle.packManifestPath;
      if (existsSync(manifestPath)) {
        const actualHash = hashFile(manifestPath);
        if (manifest.contentHash !== actualHash) {
          hashDrifts.push({
            label: path.basename(manifestPath),
            path: manifestPath,
            expected: manifest.contentHash,
            actual: actualHash,
          });
        }
      }
    }
  }
  if (hashDrifts.length > 0) {
    addFinding(findings, {
      code: "manifest_hash_drift",
      summary: `${hashDrifts.length} manifest or artifact hashes drifted from the written content`,
      detail: hashDrifts.slice(0, 5).map((item) => `${relativeWorkspacePath(item.path, workspaceRoot)} expected=${item.expected} actual=${item.actual}`).join("; "),
      evidenceRefs: [
        { sourceId: "docs/architecture/compiled-artifacts.md", evidenceId: "evi_compiled_artifacts", authority: "raw_source", derivation: "teacher_lint" },
        ...evidenceRefs.slice(0, 2),
      ],
      paths: hashDrifts.map((item) => relativeWorkspacePath(item.path, workspaceRoot)),
      truthLayer: "manifest",
    });
  }

  const promotableOnly = PROMOTABLE_CLASSES.has(proposalClass);
  const shadowOnly = SHADOW_ONLY_CLASSES.has(proposalClass);
  const promotionSignals = [proposalStatus, reviewMode, bundle.status?.proposalStatus, bundle.verdict?.verdict].filter((value) => typeof value === "string");
  if (shadowOnly && promotionSignals.some((value) => ["promotable", "promoted"].includes(value))) {
    addFinding(findings, {
      code: "illegal_trust_class_promotion",
      summary: `shadow-only proposal class ${proposalClass} is surfaced as promotable`,
      detail: `status=${proposalStatus}; reviewMode=${reviewMode}`,
      evidenceRefs: [
        { sourceId: "docs/architecture/teacher-v3.md", evidenceId: "evi_teacher_v3_truth", authority: "raw_source", derivation: "teacher_lint" },
        { sourceId: "docs/architecture/teacher-v3-proposals.md", evidenceId: "evi_teacher_v3_proposals", authority: "raw_source", derivation: "teacher_lint" },
      ],
      truthLayer: "proposal_truth",
    });
  }
  if (!promotableOnly && !shadowOnly && ["promotable", "promoted"].includes(proposalStatus)) {
    addFinding(findings, {
      code: "illegal_trust_class_promotion",
      summary: `unknown proposal class ${proposalClass} is being promoted`,
      detail: `status=${proposalStatus}; reviewMode=${reviewMode}`,
      evidenceRefs: [
        { sourceId: "docs/architecture/teacher-v3.md", evidenceId: "evi_teacher_v3_truth", authority: "raw_source", derivation: "teacher_lint" },
      ],
      truthLayer: "proposal_truth",
    });
  }

  const evidenceList = Array.isArray(proposal?.evidence) ? proposal.evidence : Array.isArray(bundle.proposalReport?.evidence) ? bundle.proposalReport.evidence : [];
  const counterevidenceList = Array.isArray(proposal?.counterevidence) ? proposal.counterevidence : Array.isArray(bundle.proposalReport?.counterevidence) ? bundle.proposalReport.counterevidence : [];
  const claims = Array.isArray(proposal?.claims) ? proposal.claims : Array.isArray(bundle.proposalReport?.claims) ? bundle.proposalReport.claims : [];
  const allEvidenceIds = new Set(evidenceList.map((item) => item?.evidenceId).filter(Boolean));
  const evidenceProblems = [];
  for (const evidence of evidenceList) {
    if (!evidence || typeof evidence !== "object" || typeof evidence.evidenceId !== "string" || evidence.evidenceId.trim().length === 0 || typeof evidence.sourceId !== "string" || evidence.sourceId.trim().length === 0 || typeof evidence.authority !== "string" || evidence.authority.trim().length === 0) {
      evidenceProblems.push("invalid evidence ref");
    }
  }
  for (const evidence of counterevidenceList) {
    if (!evidence || typeof evidence !== "object" || typeof evidence.evidenceId !== "string" || evidence.evidenceId.trim().length === 0 || typeof evidence.sourceId !== "string" || evidence.sourceId.trim().length === 0 || typeof evidence.authority !== "string" || evidence.authority.trim().length === 0) {
      evidenceProblems.push("invalid counterevidence ref");
    }
  }
  for (const claim of claims) {
    const claimRefs = Array.isArray(claim?.evidenceIds) ? claim.evidenceIds.filter((item) => typeof item === "string" && item.trim().length > 0) : [];
    if (claimRefs.length === 0) {
      evidenceProblems.push(`claim ${claim?.claimId ?? "unknown"} is missing evidence refs`);
      continue;
    }
    for (const claimRef of claimRefs) {
      if (!allEvidenceIds.has(claimRef)) {
        evidenceProblems.push(`claim ${claim?.claimId ?? "unknown"} references missing evidence ${claimRef}`);
      }
    }
  }
  if (evidenceProblems.length > 0) {
    addFinding(findings, {
      code: "missing_evidence_refs",
      summary: `${evidenceProblems.length} evidence coverage issues were found`,
      detail: evidenceProblems.slice(0, 6).join("; "),
      evidenceRefs: [
        { sourceId: "docs/architecture/compiled-artifacts.md", evidenceId: "evi_compiled_artifacts", authority: "raw_source", derivation: "teacher_lint" },
        { sourceId: "docs/architecture/teacher-v3-proof.md", evidenceId: "evi_teacher_v3_proof", authority: "raw_source", derivation: "teacher_lint" },
      ],
      truthLayer: "proposal_truth",
    });
  }

  const joinProblems = [];
  const bundleIds = [bundle.manifest?.bundleId, bundle.packManifest?.bundleId, bundle.status?.bundleId, bundle.surfaceMap?.bundleId, bundle.proposalReport?.bundleId, bundle.verdict?.bundleId].filter((value) => typeof value === "string");
  if (new Set(bundleIds).size > 1) {
    joinProblems.push(`bundleId mismatch: ${bundleIds.join(", ")}`);
  }
  const proposalIds = [bundle.status?.proposalId, proposal?.proposalId, bundle.proposalReport?.proposalId, bundle.verdict?.proposalId].filter((value) => typeof value === "string");
  if (new Set(proposalIds).size > 1) {
    joinProblems.push(`proposalId mismatch: ${proposalIds.join(", ")}`);
  }
  const proposalClasses = [bundle.status?.proposalClass, bundle.status?.proposalLane, proposal?.proposalClass, bundle.proposalReport?.proposalClass, bundle.verdict?.proposalClass].filter((value) => typeof value === "string");
  if (new Set(proposalClasses).size > 1) {
    joinProblems.push(`proposal class mismatch: ${proposalClasses.join(", ")}`);
  }
  const reviewModes = [bundle.status?.reviewMode, proposal?.reviewMode, bundle.proposalReport?.reviewMode, bundle.verdict?.reviewMode].filter((value) => typeof value === "string");
  if (new Set(reviewModes).size > 1) {
    joinProblems.push(`reviewMode mismatch: ${reviewModes.join(", ")}`);
  }
  const surfaceMapCounts = bundle.surfaceMap?.counts;
  if (surfaceMapCounts && typeof surfaceMapCounts === "object") {
    const observedLength = Array.isArray(bundle.surfaceMap?.observedSurfaces) ? bundle.surfaceMap.observedSurfaces.length : null;
    const bundleArtifactLength = Array.isArray(bundle.surfaceMap?.bundleArtifacts) ? bundle.surfaceMap.bundleArtifacts.length : null;
    const totalLength = (observedLength ?? 0) + (bundleArtifactLength ?? 0);
    if (
      surfaceMapCounts.observedSurfaceCount !== observedLength ||
      surfaceMapCounts.totalSurfaceCount !== totalLength ||
      surfaceMapCounts.targetSurfaceCount !== bundleArtifactLength
    ) {
      joinProblems.push(
        `surface counts mismatch: observed=${surfaceMapCounts.observedSurfaceCount}/${observedLength} target=${surfaceMapCounts.targetSurfaceCount}/${bundleArtifactLength} total=${surfaceMapCounts.totalSurfaceCount}/${totalLength}`,
      );
    }
  }
  const publicationSafeArtifacts = Array.isArray(bundle.proposalReport?.publicationSafeArtifacts) ? bundle.proposalReport.publicationSafeArtifacts : [];
  const publicationSafeProblems = publicationSafeArtifacts.filter((artifact) => {
    if (!artifact || typeof artifact.path !== "string") {
      return true;
    }
    const resolved = resolveRef(artifact.path, bundleRoot, repoRoot, workspaceRoot);
    return !resolved.exists;
  });
  if (publicationSafeProblems.length > 0) {
    joinProblems.push(`publication-safe artifacts are missing: ${publicationSafeProblems.slice(0, 5).map((item) => item?.path ?? "unknown").join(", ")}`);
  }
  const releaseCandidates = extractVersionCandidates(bundleObjects);
  const referencesPublicReleaseSurfaces = [...sourceRefs].some((ref) => {
    const cleaned = stripFragment(ref);
    if (RELEVANT_RELEASE_SURFACES.has(cleaned)) {
      return true;
    }
    return /^docs\/release-notes-((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\.md$/.test(cleaned);
  });
  const releaseState = readReleaseState(repoRoot);
  const repoReleaseIssues = [];
  if (releaseState.currentVersion === null) {
    repoReleaseIssues.push("CHANGELOG.md does not expose a current semver heading");
  }
  if (releaseState.currentVersion !== null) {
    if (releaseState.readmeVersion === null) {
      repoReleaseIssues.push("README.md does not advertise the current version banner");
    } else if (releaseState.readmeVersion !== releaseState.currentVersion) {
      repoReleaseIssues.push(`README.md advertises ${releaseState.readmeVersion} instead of ${releaseState.currentVersion}`);
    }
    if (releaseState.docsIndexVersion === null || releaseState.docsIndexTargetVersion === null) {
      repoReleaseIssues.push("docs/README.md does not point at the current release notes link");
    } else if (releaseState.docsIndexVersion !== releaseState.currentVersion || releaseState.docsIndexTargetVersion !== releaseState.currentVersion) {
      repoReleaseIssues.push(`docs/README.md points at ${releaseState.docsIndexVersion}/${releaseState.docsIndexTargetVersion} instead of ${releaseState.currentVersion}`);
    }
    if (releaseState.endStateVersions.length > 0 && releaseState.endStateVersions.some(([openclawVersion, cliVersion]) => openclawVersion !== releaseState.currentVersion || cliVersion !== releaseState.currentVersion)) {
      repoReleaseIssues.push(`docs/END_STATE.md still mixes split-package versions instead of ${releaseState.currentVersion}`);
    }
    if (releaseState.releaseNotesPath !== null && !existsSync(releaseState.releaseNotesPath)) {
      repoReleaseIssues.push(`docs/release-notes-${releaseState.currentVersion}.md is missing`);
    }
  }
  if (repoReleaseIssues.length > 0 && referencesPublicReleaseSurfaces) {
    addFinding(findings, {
      code: "release_docs_version_drift",
      summary: `${repoReleaseIssues.length} public release-truth checks failed in the repository`,
      detail: repoReleaseIssues.join("; "),
      evidenceRefs: [
        { sourceId: "scripts/verify-release-docs-drift.mjs", evidenceId: "evi_release_drift_script", authority: "raw_source", derivation: "teacher_lint" },
        { sourceId: "docs/architecture/teacher-v3-lints.md", evidenceId: "evi_teacher_v3_lints", authority: "raw_source", derivation: "teacher_lint" },
      ],
      truthLayer: "docs_truth",
    });
  }
  if (releaseCandidates.length > 0 && releaseState.currentVersion !== null && releaseCandidates.some((version) => version !== releaseState.currentVersion)) {
    addFinding(findings, {
      code: "release_docs_version_drift",
      summary: `bundle release references point at ${releaseCandidates.join(", ")} instead of ${releaseState.currentVersion}`,
      detail: `bundle versions=${releaseCandidates.join(", ")}; currentVersion=${releaseState.currentVersion}`,
      evidenceRefs: [
        { sourceId: "scripts/verify-release-docs-drift.mjs", evidenceId: "evi_release_drift_script", authority: "raw_source", derivation: "teacher_lint" },
        ...evidenceRefs.slice(0, 2),
      ],
      truthLayer: "docs_truth",
    });
  }

  if (bundle.status?.proposalStatus !== undefined && bundle.verdict?.verdict !== undefined && bundle.status.proposalStatus !== bundle.verdict.verdict) {
    joinProblems.push(`verdict/status mismatch: status=${bundle.status.proposalStatus} verdict=${bundle.verdict.verdict}`);
  }
  if (joinProblems.length > 0) {
    addFinding(findings, {
      code: "broken_bundle_joins",
      summary: `${joinProblems.length} bundle join problems were found`,
      detail: joinProblems.slice(0, 6).join("; "),
      evidenceRefs: evidenceRefs.slice(0, 4),
      truthLayer: "bundle_join",
    });
  }

  return {
    ok: findings.length === 0,
    bundleId,
    proposalId,
    proposalClass,
    reviewMode,
    proposalStatus,
    evidenceRefs: evidenceRefs.slice(0, 12),
    sourceRefs: resolvedRefs.map((ref) => ({
      sourceRef: ref.sourceRef,
      resolvedPath: ref.resolvedPath === null ? null : relativeWorkspacePath(ref.resolvedPath, workspaceRoot),
      exists: ref.exists,
      hash: ref.exists && ref.resolvedPath !== null ? hashFile(ref.resolvedPath) : null,
    })),
    releaseState: referencesPublicReleaseSurfaces ? releaseState : null,
    findings,
  };
}

function chooseSeverity(findings) {
  if (findings.some((finding) => finding.severity === "error")) {
    return "error";
  }
  if (findings.some((finding) => finding.severity === "warn")) {
    return "warn";
  }
  return "info";
}

function buildSummaryMarkdown(report) {
  const findingLines = report.findings.length > 0
    ? report.findings.map((finding) => `- **${finding.code}** — ${finding.summary}${finding.detail ? ` (${finding.detail})` : ""}`)
    : ["- none"];
  return [
    "# Graphify deterministic pre-lint",
    "",
    `- bundle: \`${report.bundleId}\``,
    `- proposal: \`${report.proposalId}\` (${report.proposalClass}, ${report.reviewMode})`,
    `- verdict: **${report.verdict}**`,
    `- severity: **${report.severity}**`,
    `- blockers: ${report.blockerCount}`, 
    `- warnings: ${report.warningCount}`,
    `- checked bundle files: summary.md, status.json, surface-map.json, proposal-report.json, verdict.json`,
    `- source refs checked: ${report.sourceRefCount}`,
    `- evidence refs attached: ${report.evidenceRefCount}`,
    "",
    "## Findings",
    ...findingLines,
    "",
    "## Guardrail",
    "This bundle is review/proposal-only. It does not mutate the graph or serve path.",
  ].join("\n") + "\n";
}

function buildProposalEnvelope(report, bundlePaths) {
  return {
    contract: "graphify_deterministic_lint_proposal_envelope.v1",
    bundleId: report.bundleId,
    proposalId: report.proposalId,
    lane: "lint",
    status: report.ok ? "reviewable" : "rejected",
    reviewMode: "deterministic",
    proposalClass: report.proposalClass,
    trustClass: report.proposalClass,
    bundleRoot: report.bundleRoot,
    repoRoot: report.repoRoot,
    lineage: {
      producer: "graphify-deterministic-lint",
      producerVersion: "0.1.0",
      scope: "bundle-prelint",
      idempotencyKey: sha256Text([report.bundleRoot, report.bundleId, report.findings.map((finding) => finding.code).join("|")].join("::")),
      sourceBundleId: report.bundleId,
      sourceManifestHash: report.bundleManifestHash ?? null,
    },
    subjectIds: uniqueByKey([report.bundleId, report.proposalId, ...(report.releaseState?.currentVersion ? [`release:${report.releaseState.currentVersion}`] : [])], (value) => value),
    evidence: report.evidenceRefs,
    counterevidence: [],
    findings: report.findings,
    recommendations: report.ok
      ? ["Proceed to semantic-lint or proposal review only after the deterministic bundle stays clean."]
      : ["Fix the deterministic blockers before any semantic lint or graph mutation work.", "Keep the bundle review-only until hash, join, evidence, and truth surfaces align."],
    bundlePaths,
    counts: {
      findings: report.findings.length,
      blockers: report.blockerCount,
      warnings: report.warningCount,
    },
  };
}

function buildVerdict(report, bundlePaths) {
  return {
    contract: "graphify_deterministic_lint_verdict.v1",
    bundleId: report.bundleId,
    proposalId: report.proposalId,
    verdict: report.ok ? "reviewable" : "rejected",
    severity: report.severity,
    reviewMode: "deterministic",
    why: report.ok
      ? "deterministic pre-lint found no blockers"
      : `${report.blockerCount} blocker(s) and ${report.warningCount} warning(s) were found`,
    blockerCount: report.blockerCount,
    warningCount: report.warningCount,
    findingCount: report.findings.length,
    bundlePaths,
    outputRoot: report.outputRoot,
  };
}

function buildNormalizedReport(bundleRoot, repoRoot, workspaceRoot) {
  const bundle = normalizeBundleObjects(bundleRoot);
  const bundleId = bundle.proposalReport?.bundleId ?? bundle.status?.bundleId ?? bundle.surfaceMap?.bundleId ?? bundle.manifest?.bundleId ?? bundle.packManifest?.packId ?? path.basename(bundleRoot);
  const proposal = bundle.proposalReport?.proposal ?? bundle.status?.proposal ?? null;
  const proposalId = proposal?.proposalId ?? bundle.proposalReport?.proposalId ?? bundle.status?.proposalId ?? bundle.verdict?.proposalId ?? bundleId;
  const bundleAnalysis = buildBundleFindings(bundleRoot, repoRoot, workspaceRoot, bundle);
  const findings = bundleAnalysis.findings;
  const sourceRefObjects = bundleAnalysis.sourceRefs;
  const releaseState = bundleAnalysis.releaseState;
  const evidenceRefs = bundleAnalysis.evidenceRefs;
  const severity = chooseSeverity(findings);
  const blockerCount = findings.filter((finding) => finding.severity === "error").length;
  const warningCount = findings.filter((finding) => finding.severity === "warn").length;
  return {
    contract: "graphify_deterministic_lint_bundle.v1",
    bundleId,
    proposalId,
    proposalClass: proposal?.proposalClass ?? bundle.proposalReport?.proposalClass ?? bundle.status?.proposalClass ?? bundle.status?.proposalLane ?? "lint",
    reviewMode: proposal?.reviewMode ?? bundle.proposalReport?.reviewMode ?? bundle.status?.reviewMode ?? "reviewable",
    proposalStatus: proposal?.status ?? bundle.proposalReport?.status ?? bundle.status?.proposalStatus ?? bundle.verdict?.verdict ?? "reviewable",
    bundleRoot: relativeWorkspacePath(bundleRoot, workspaceRoot),
    repoRoot: relativeWorkspacePath(repoRoot, workspaceRoot),
    outputRoot: null,
    inspectedAt: new Date().toISOString(),
    bundlePaths: {
      summary: relativeWorkspacePath(bundle.summaryPath, workspaceRoot),
      status: relativeWorkspacePath(bundle.statusPath, workspaceRoot),
      surfaceMap: relativeWorkspacePath(bundle.surfaceMapPath, workspaceRoot),
      proposalReport: relativeWorkspacePath(bundle.proposalReportPath, workspaceRoot),
      verdict: relativeWorkspacePath(bundle.verdictPath, workspaceRoot),
      manifest: existsSync(bundle.manifestPath) ? relativeWorkspacePath(bundle.manifestPath, workspaceRoot) : null,
      packManifest: existsSync(bundle.packManifestPath) ? relativeWorkspacePath(bundle.packManifestPath, workspaceRoot) : null,
    },
    bundleManifestHash: existsSync(bundle.packManifestPath) ? hashFile(bundle.packManifestPath) : (existsSync(bundle.manifestPath) ? hashFile(bundle.manifestPath) : null),
    findings,
    severity,
    verdict: findings.length === 0 ? "reviewable" : "rejected",
    ok: findings.length === 0,
    blockerCount,
    warningCount,
    sourceRefCount: sourceRefObjects.length,
    evidenceRefCount: evidenceRefs.length,
    sourceRefs: sourceRefObjects,
    evidenceRefs,
    releaseState,
  };
}

export function buildGraphifyDeterministicLintBundle(options = {}) {
  const repoRoot = path.resolve(options.repoRoot ?? defaultRepoRoot);
  const workspaceRoot = path.resolve(options.workspaceRoot ?? path.resolve(repoRoot, ".."));
  const bundleRoot = path.resolve(options.bundleRoot ?? options.inputBundleRoot ?? options.bundleDir ?? options.bundlePath ?? "");
  if (!existsSync(bundleRoot)) {
    throw new Error(`bundle root does not exist: ${bundleRoot}`);
  }
  const report = buildNormalizedReport(bundleRoot, repoRoot, workspaceRoot);
  const runId = options.runId ?? `graphify-lint-${Date.now()}`;
  const outputRoot = path.resolve(options.outputRoot ?? path.join(defaultOutputRoot, runId));
  ensureDir(outputRoot);
  const bundlePaths = {
    deterministicLints: path.join(outputRoot, GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT.deterministicLints),
    summary: path.join(outputRoot, GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT.summary),
    proposalEnvelope: path.join(outputRoot, GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT.proposalEnvelope),
    verdict: path.join(outputRoot, GRAPHIFY_DETERMINISTIC_LINT_BUNDLE_LAYOUT.verdict),
  };
  report.outputRoot = relativeWorkspacePath(outputRoot, workspaceRoot);
  const proposalEnvelope = buildProposalEnvelope(report, Object.fromEntries(Object.entries(bundlePaths).map(([key, value]) => [key, relativeWorkspacePath(value, workspaceRoot)])));
  const verdict = buildVerdict(report, Object.fromEntries(Object.entries(bundlePaths).map(([key, value]) => [key, relativeWorkspacePath(value, workspaceRoot)])));
  const summary = buildSummaryMarkdown(report);
  const deterministicLints = {
    ...report,
    outputRoot: relativeWorkspacePath(outputRoot, workspaceRoot),
    bundlePaths: Object.fromEntries(Object.entries(bundlePaths).map(([key, value]) => [key, relativeWorkspacePath(value, workspaceRoot)])),
  };
  writeJson(bundlePaths.deterministicLints, deterministicLints);
  writeText(bundlePaths.summary, summary);
  writeJson(bundlePaths.proposalEnvelope, proposalEnvelope);
  writeJson(bundlePaths.verdict, verdict);
  return {
    ok: report.ok,
    runId,
    outputRoot,
    bundleRoot,
    repoRoot,
    workspaceRoot,
    report: deterministicLints,
    proposalEnvelope,
    verdict,
    paths: bundlePaths,
    summary,
  };
}

export function parseGraphifyDeterministicLintCliArgs(argv) {
  let bundleRoot = null;
  let repoRoot = defaultRepoRoot;
  let workspaceRoot = defaultWorkspaceRoot;
  let outputRoot = null;
  let runId = null;
  let json = false;
  let help = false;
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
    if (arg === "--bundle-root") {
      const next = argv[index + 1];
      if (next === undefined) {
        throw new Error("--bundle-root requires a value");
      }
      bundleRoot = next;
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
    throw new Error(`unknown argument for graphify-lints: ${arg}`);
  }
  return {
    command: "graphify-lints",
    bundleRoot: bundleRoot === null ? null : path.resolve(bundleRoot),
    repoRoot: path.resolve(repoRoot),
    workspaceRoot: path.resolve(workspaceRoot),
    outputRoot: outputRoot === null ? null : path.resolve(outputRoot),
    runId,
    json,
    help,
  };
}

export function formatGraphifyDeterministicLintSummary(result) {
  const lines = [
    "GRAPHIFY deterministic pre-lint",
    `bundle: ${result.report.bundleId}`,
    `proposal: ${result.report.proposalId} (${result.report.proposalClass}, ${result.report.reviewMode})`,
    `verdict: ${result.verdict.verdict} (${result.verdict.severity})`,
    `findings: ${result.report.findings.length} (blockers=${result.report.blockerCount}, warnings=${result.report.warningCount})`,
    `output: ${result.paths.summary}`,
    `proposal-envelope: ${result.paths.proposalEnvelope}`,
    `deterministic-lints: ${result.paths.deterministicLints}`,
    `verdict-json: ${result.paths.verdict}`,
  ];
  if (result.report.findings.length > 0) {
    lines.push("findings:");
    for (const finding of result.report.findings.slice(0, 8)) {
      lines.push(`  - ${finding.code}: ${finding.summary}`);
    }
  }
  return `${lines.join("\n")}\n`;
}

export function runGraphifyDeterministicLints(argvOrOptions = {}) {
  const parsed = Array.isArray(argvOrOptions)
    ? parseGraphifyDeterministicLintCliArgs(argvOrOptions)
    : { command: "graphify-lints", json: false, help: false, ...argvOrOptions };
  if (parsed.help) {
    return {
      ok: true,
      help: true,
      summary: "",
      report: null,
      verdict: null,
      proposalEnvelope: null,
      paths: null,
      outputRoot: null,
      bundleRoot: null,
      repoRoot: null,
      workspaceRoot: null,
      runId: null,
    };
  }
  const bundleRoot = parsed.bundleRoot ?? parsed.inputBundleRoot ?? parsed.bundlePath ?? parsed.bundleDir;
  if (bundleRoot === null || bundleRoot === undefined) {
    throw new Error("graphify-lints requires --bundle-root <path>");
  }
  const result = buildGraphifyDeterministicLintBundle({
    bundleRoot,
    repoRoot: parsed.repoRoot,
    workspaceRoot: parsed.workspaceRoot,
    outputRoot: parsed.outputRoot,
    runId: parsed.runId,
  });
  return {
    ...result,
    json: Boolean(parsed.json),
    summary: formatGraphifyDeterministicLintSummary(result),
  };
}

function main() {
  const result = runGraphifyDeterministicLints(process.argv.slice(2));
  if (result.help) {
    process.stdout.write([
      "Usage:",
      "  node scripts/graphify-lints.mjs --bundle-root <path> [--repo-root <path>] [--workspace-root <path>] [--output-root <path>] [--run-id <id>] [--json]",
      "",
      "This deterministic pre-lint writes bounded review/proposal surfaces only.",
    ].join("\n") + "\n");
    return;
  }
  if (result.json) {
    process.stdout.write(`${stableJson({ ok: result.ok, report: result.report, verdict: result.verdict, proposalEnvelope: result.proposalEnvelope, paths: result.paths })}`);
    return;
  }
  process.stdout.write(result.summary);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
