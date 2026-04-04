#!/usr/bin/env node

const TEACHER_V3_REPLAY_OUTCOME_REVIEW_MODE_BY_CLASS = {
  compiler: "promotable",
  lint: "promotable",
  mutation: "shadow_only",
  forgetting: "shadow_only",
  correction: "shadow_only",
};

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeStringArray(values) {
  const seen = new Set();
  const normalized = [];
  for (const value of Array.isArray(values) ? values : []) {
    const text = normalizeText(value);
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    normalized.push(text);
  }
  return normalized;
}

function normalizeProofLink(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const refId = normalizeText(entry.refId);
  const kind = normalizeText(entry.kind);
  const path = normalizeText(entry.path);
  if (!refId || !kind || !path) {
    return null;
  }
  return { refId, kind, path };
}

function normalizeProofLinkArray(values) {
  const normalized = [];
  for (const value of Array.isArray(values) ? values : []) {
    const entry = normalizeProofLink(value);
    if (entry) {
      normalized.push(entry);
    }
  }
  return normalized;
}

function normalizeReviewMode(value, proposalClass) {
  const text = normalizeText(value);
  if (text === "promotable" || text === "shadow_only") {
    return text;
  }
  return TEACHER_V3_REPLAY_OUTCOME_REVIEW_MODE_BY_CLASS[proposalClass] ?? "shadow_only";
}

function normalizeResultFromVerdict(verdict) {
  const verdictText = normalizeText(verdict?.verdict);
  const severityText = normalizeText(verdict?.severity);

  if (verdictText === "reviewable" || verdictText === "success_and_proven") {
    return "pass";
  }
  if (verdictText === "success_but_proof_incomplete") {
    return "warn";
  }
  if (verdictText === "degraded_or_failed_proof" || verdictText === "rejected") {
    return "fail";
  }
  if (severityText === "warn" || severityText === "degraded") {
    return "warn";
  }
  if (severityText === "blocking") {
    return "fail";
  }
  return "pass";
}

function normalizeResult(value, verdict) {
  const text = normalizeText(value);
  if (text === "pass" || text === "warn" || text === "fail") {
    return text;
  }
  return normalizeResultFromVerdict(verdict);
}

function normalizeSource(value, fallback = "derived") {
  const text = normalizeText(value);
  if (text === "proposal_record" || text === "proof_bundle" || text === "derived") {
    return text;
  }
  return fallback;
}

function uniqueStrings(values) {
  const seen = new Set();
  const unique = [];
  for (const value of values) {
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    unique.push(value);
  }
  return unique;
}

function normalizeReplayOutcome(entry, fallback, index) {
  const proposalClass = normalizeText(entry?.proposalClass) ?? fallback.proposalClass;
  const replaySuite = normalizeText(entry?.replaySuite)
    ?? fallback.replaySuites[index]
    ?? fallback.replaySuites[0]
    ?? `${proposalClass}-proof-bundle`;
  const reviewMode = normalizeReviewMode(entry?.reviewMode, proposalClass);
  const result = normalizeResult(entry?.result, fallback.proofVerdict);
  const source = normalizeSource(entry?.source, fallback.hasExplicitOutcomes ? "proposal_record" : "proof_bundle");
  const outcomeId = normalizeText(entry?.outcomeId)
    ?? `${fallback.bundleId ?? fallback.proposalId ?? proposalClass}:${replaySuite}:${index + 1}`;
  const summary = normalizeText(entry?.summary)
    ?? (fallback.proofVerdict?.why ? fallback.proofVerdict.why : `Captured ${proposalClass} replay outcome for ${replaySuite}`);
  const capturedAt = normalizeText(entry?.capturedAt) ?? fallback.bundleStartedAt;
  const evidenceLinks = normalizeProofLinkArray(entry?.evidenceLinks);
  const counterevidenceLinks = normalizeProofLinkArray(entry?.counterevidenceLinks);
  const notes = normalizeStringArray(entry?.notes);

  return {
    outcomeId,
    replaySuite,
    proposalClass,
    reviewMode,
    result,
    source,
    summary,
    evidenceLinks: evidenceLinks.length > 0 ? evidenceLinks : undefined,
    counterevidenceLinks: counterevidenceLinks.length > 0 ? counterevidenceLinks : undefined,
    capturedAt,
    notes: notes.length > 0 ? notes : undefined,
  };
}

export function buildTeacherV3ReplayOutcomeSummary(outcomes) {
  const replayOutcomes = Array.isArray(outcomes) ? outcomes : [];
  const resultCounts = {
    pass: 0,
    warn: 0,
    fail: 0,
  };
  const reviewModeCounts = {
    promotable: 0,
    shadow_only: 0,
  };
  const sourceCounts = {
    proposal_record: 0,
    proof_bundle: 0,
    derived: 0,
  };
  const replaySuites = [];

  for (const outcome of replayOutcomes) {
    if (outcome && typeof outcome === "object") {
      if (Object.prototype.hasOwnProperty.call(resultCounts, outcome.result)) {
        resultCounts[outcome.result] += 1;
      }
      if (Object.prototype.hasOwnProperty.call(reviewModeCounts, outcome.reviewMode)) {
        reviewModeCounts[outcome.reviewMode] += 1;
      }
      if (Object.prototype.hasOwnProperty.call(sourceCounts, outcome.source)) {
        sourceCounts[outcome.source] += 1;
      }
      const suite = normalizeText(outcome.replaySuite);
      if (suite) {
        replaySuites.push(suite);
      }
    }
  }

  const uniqueReplaySuites = uniqueStrings(replaySuites);
  const outcomeCount = replayOutcomes.length;
  const summary = outcomeCount === 0
    ? "No replay outcomes captured."
    : `Captured ${outcomeCount} replay outcome${outcomeCount === 1 ? "" : "s"} across ${uniqueReplaySuites.length} suite${uniqueReplaySuites.length === 1 ? "" : "s"} (${uniqueReplaySuites.length > 0 ? uniqueReplaySuites.join(", ") : "none"}); results pass=${resultCounts.pass}, warn=${resultCounts.warn}, fail=${resultCounts.fail}; review modes promotable=${reviewModeCounts.promotable}, shadow_only=${reviewModeCounts.shadow_only}; sources proposal_record=${sourceCounts.proposal_record}, proof_bundle=${sourceCounts.proof_bundle}, derived=${sourceCounts.derived}.`;

  return {
    replayOutcomeCount: outcomeCount,
    replaySuites: uniqueReplaySuites,
    resultCounts,
    reviewModeCounts,
    sourceCounts,
    summary,
  };
}

export function captureTeacherV3ReplayOutcomes(input = {}) {
  const explicitOutcomes = Array.isArray(input.replayOutcomes)
    ? input.replayOutcomes.filter((value) => value !== null && value !== undefined)
    : [];
  const fallback = {
    bundleId: normalizeText(input.bundleId),
    proposalId: normalizeText(input.proposalId),
    proposalClass: normalizeText(input.proposalClass) ?? "lint",
    replaySuites: normalizeStringArray(input.replaySuites),
    proofVerdict: input.proofVerdict && typeof input.proofVerdict === "object" ? input.proofVerdict : null,
    bundleStartedAt: normalizeText(input.bundleStartedAt) ?? new Date().toISOString(),
    hasExplicitOutcomes: explicitOutcomes.length > 0,
  };

  const outcomes = explicitOutcomes.length > 0
    ? explicitOutcomes.map((entry, index) => normalizeReplayOutcome(entry, fallback, index))
    : [{
        outcomeId: `${fallback.bundleId ?? fallback.proposalId ?? fallback.proposalClass}:proof-bundle`,
        replaySuite: fallback.replaySuites[0] ?? "teacher-v3-proof-bundle",
        proposalClass: fallback.proposalClass,
        reviewMode: normalizeReviewMode(input.reviewMode, fallback.proposalClass),
        result: normalizeResult(undefined, fallback.proofVerdict),
        source: "proof_bundle",
        summary: fallback.proofVerdict?.why ?? `Derived replay outcome for ${fallback.proposalClass}`,
        capturedAt: fallback.bundleStartedAt,
        notes: ["derived from proof bundle verdict"],
      }];

  return {
    outcomes,
    summary: buildTeacherV3ReplayOutcomeSummary(outcomes),
  };
}
