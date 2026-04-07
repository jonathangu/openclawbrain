export const ROUTER_MIGRATION_COMPARISON_CONTRACT = "openclawbrain_router_migration_comparison.v1" as const;

export type RouterMigrationVariantV1 = "old_live" | "base_only" | "mixed";

export interface RouterMigrationVariantInputV1 {
  packVersion?: number | null;
  packId?: string | null;
  routerChecksum?: string | null;
  graphHash?: string | null;
  notes?: string[];
}

export interface RouterMigrationReplayCaseV1 {
  caseId: string;
  summary: string;
  highAuthority?: boolean;
  explicitCorrection?: boolean;
  preserved: Record<RouterMigrationVariantV1, boolean>;
  notes?: string[];
}

export interface RouterMigrationComparisonInputV1 {
  migrationId: string;
  proposalId?: string | null;
  rollbackKey?: string | null;
  proofBundleId?: string | null;
  priorLivePackVersion?: number | null;
  priorLivePackId?: string | null;
  priorRouterChecksum?: string | null;
  proofBundleFiles?: string[];
  variants: Record<RouterMigrationVariantV1, RouterMigrationVariantInputV1>;
  replayCases: RouterMigrationReplayCaseV1[];
}

export interface RouterMigrationVariantSummaryV1 extends RouterMigrationVariantInputV1 {
  variant: RouterMigrationVariantV1;
  caseCount: number;
  preservedCaseCount: number;
  highAuthorityCaseCount: number;
  explicitCorrectionCaseCount: number;
  weightedCaseCount: number;
  weightedPreservedCount: number;
  supportRatio: number | null;
  highAuthoritySupportRatio: number | null;
  explicitCorrectionSupportRatio: number | null;
  regressionCaseCount: number;
  highAuthorityRegressionCount: number;
  explicitCorrectionRegressionCount: number;
  notes: string[];
  summary: string;
}

export interface RouterMigrationReplayCaseSummaryV1 extends RouterMigrationReplayCaseV1 {
  preservedByVariant: Record<RouterMigrationVariantV1, boolean>;
  weight: number;
}

export interface RouterMigrationExplicitCorrectionProtectionV1 {
  caseCount: number;
  preservedCount: number;
  regressionCount: number;
  protected: boolean;
  blockers: string[];
  summary: string;
}

export interface RouterMigrationComparisonVerdictV1 {
  winner: RouterMigrationVariantV1;
  decision: "promote" | "hold";
  blocked: boolean;
  allowed: boolean;
  blockers: string[];
  summary: string;
  supportRatios: Record<RouterMigrationVariantV1, number | null>;
  supportOrdering: RouterMigrationVariantV1[];
}

export interface RouterMigrationRollbackSummaryV1 {
  rollbackKey: string | null;
  priorLivePackVersion: number | null;
  priorLivePackId: string | null;
  priorRouterChecksum: string | null;
  proofBundleId: string | null;
  proofBundleFiles: string[];
  available: boolean;
  summary: string;
}

export interface RouterMigrationProofBundleExpectationsV1 {
  expectedFiles: string[];
  exactArtifactRefs: boolean;
  checksumsBound: boolean;
  rollbackBound: boolean;
  proofBundleBound: boolean;
  summary: string;
  blockers: string[];
}

export interface RouterMigrationComparisonSummaryV1 {
  contract: typeof ROUTER_MIGRATION_COMPARISON_CONTRACT;
  state: "target";
  migrationId: string;
  proposalId: string | null;
  rollbackKey: string | null;
  proofBundleId: string | null;
  proofBundleFiles: string[];
  replayCaseCount: number;
  replayCases: RouterMigrationReplayCaseSummaryV1[];
  variants: Record<RouterMigrationVariantV1, RouterMigrationVariantSummaryV1>;
  comparison: RouterMigrationComparisonVerdictV1;
  explicitCorrectionProtection: RouterMigrationExplicitCorrectionProtectionV1;
  rollback: RouterMigrationRollbackSummaryV1;
  proofBundleExpectations: RouterMigrationProofBundleExpectationsV1;
  summary: string;
}

function normalizeText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeNumber(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeBoolean(value: unknown): boolean {
  return value === true;
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter((value) => typeof value === "string" && value.length > 0))];
}

function variantTieBreakOrder(variant: RouterMigrationVariantV1): number {
  switch (variant) {
    case "mixed":
      return 0;
    case "old_live":
      return 1;
    case "base_only":
      return 2;
  }
}

function variantWeight(caseSummary: RouterMigrationReplayCaseV1): number {
  if (caseSummary.explicitCorrection === true) {
    return 3;
  }
  if (caseSummary.highAuthority === true) {
    return 2;
  }
  return 1;
}

function summarizeVariantSupport(
  variant: RouterMigrationVariantV1,
  variantInput: RouterMigrationVariantInputV1,
  replayCases: RouterMigrationReplayCaseSummaryV1[],
): RouterMigrationVariantSummaryV1 {
  const caseCount = replayCases.length;
  const preservedCaseCount = replayCases.filter((replayCase) => replayCase.preservedByVariant[variant]).length;
  const highAuthorityCases = replayCases.filter((replayCase) => replayCase.highAuthority === true);
  const explicitCorrectionCases = replayCases.filter((replayCase) => replayCase.explicitCorrection === true);
  const highAuthorityCaseCount = highAuthorityCases.length;
  const explicitCorrectionCaseCount = explicitCorrectionCases.length;
  const highAuthorityPreservedCount = highAuthorityCases.filter((replayCase) => replayCase.preservedByVariant[variant]).length;
  const explicitCorrectionPreservedCount = explicitCorrectionCases.filter((replayCase) => replayCase.preservedByVariant[variant]).length;
  const weightedCaseCount = replayCases.reduce((total, replayCase) => total + variantWeight(replayCase), 0);
  const weightedPreservedCount = replayCases.reduce(
    (total, replayCase) => total + (replayCase.preservedByVariant[variant] ? variantWeight(replayCase) : 0),
    0,
  );
  const regressionCaseCount = caseCount - preservedCaseCount;
  const highAuthorityRegressionCount = highAuthorityCaseCount - highAuthorityPreservedCount;
  const explicitCorrectionRegressionCount = explicitCorrectionCaseCount - explicitCorrectionPreservedCount;

  const supportRatio = weightedCaseCount > 0 ? weightedPreservedCount / weightedCaseCount : null;
  const highAuthoritySupportRatio = highAuthorityCaseCount > 0 ? highAuthorityPreservedCount / highAuthorityCaseCount : null;
  const explicitCorrectionSupportRatio = explicitCorrectionCaseCount > 0 ? explicitCorrectionPreservedCount / explicitCorrectionCaseCount : null;

  const notes = uniqueStrings([
    ...(variantInput.notes ?? []),
    `${variant} replay preserves ${preservedCaseCount}/${caseCount} case${caseCount === 1 ? "" : "s"}`,
    highAuthorityCaseCount > 0
      ? `${variant} preserves ${highAuthorityPreservedCount}/${highAuthorityCaseCount} high-authority case${highAuthorityCaseCount === 1 ? "" : "s"}`
      : `${variant} has no high-authority cases in the replay set`,
    explicitCorrectionCaseCount > 0
      ? `${variant} preserves ${explicitCorrectionPreservedCount}/${explicitCorrectionCaseCount} explicit-correction case${explicitCorrectionCaseCount === 1 ? "" : "s"}`
      : `${variant} has no explicit-correction cases in the replay set`,
  ]);

  return {
    variant,
    packVersion: normalizeNumber(variantInput.packVersion),
    packId: normalizeText(variantInput.packId),
    routerChecksum: normalizeText(variantInput.routerChecksum),
    graphHash: normalizeText(variantInput.graphHash),
    caseCount,
    preservedCaseCount,
    highAuthorityCaseCount,
    explicitCorrectionCaseCount,
    weightedCaseCount,
    weightedPreservedCount,
    supportRatio,
    highAuthoritySupportRatio,
    explicitCorrectionSupportRatio,
    regressionCaseCount,
    highAuthorityRegressionCount,
    explicitCorrectionRegressionCount,
    notes,
    summary: `${variant} preserves ${preservedCaseCount}/${caseCount} replay case${caseCount === 1 ? "" : "s"}${supportRatio !== null ? ` (weighted support ${supportRatio.toFixed(3)})` : ""}${highAuthorityCaseCount > 0 ? `; high-authority ${highAuthorityPreservedCount}/${highAuthorityCaseCount}` : ""}${explicitCorrectionCaseCount > 0 ? `; explicit corrections ${explicitCorrectionPreservedCount}/${explicitCorrectionCaseCount}` : ""}.`,
  };
}

function summarizeComparisonVerdict(
  variants: Record<RouterMigrationVariantV1, RouterMigrationVariantSummaryV1>,
  explicitCorrectionProtection: RouterMigrationExplicitCorrectionProtectionV1,
  rollback: RouterMigrationRollbackSummaryV1,
  proofBundleExpectations: RouterMigrationProofBundleExpectationsV1,
): RouterMigrationComparisonVerdictV1 {
  const supportRatios: Record<RouterMigrationVariantV1, number | null> = {
    old_live: variants.old_live.supportRatio,
    base_only: variants.base_only.supportRatio,
    mixed: variants.mixed.supportRatio,
  };

  const supportOrdering = (Object.keys(variants) as RouterMigrationVariantV1[])
    .slice()
    .sort((left, right) => {
      const leftRatio = supportRatios[left] ?? -1;
      const rightRatio = supportRatios[right] ?? -1;
      if (leftRatio !== rightRatio) {
        return rightRatio - leftRatio;
      }
      return variantTieBreakOrder(left) - variantTieBreakOrder(right);
    });

  const winner = supportOrdering[0] ?? "old_live";
  const mixedCanPromote = winner === "mixed"
    && explicitCorrectionProtection.protected
    && rollback.available
    && proofBundleExpectations.exactArtifactRefs
    && proofBundleExpectations.checksumsBound
    && proofBundleExpectations.rollbackBound
    && proofBundleExpectations.proofBundleBound;

  const blockers = [
    ...explicitCorrectionProtection.blockers,
    ...rollback.available ? [] : ["rollback binding is incomplete"],
    ...proofBundleExpectations.blockers,
  ];

  const decision: RouterMigrationComparisonVerdictV1["decision"] = mixedCanPromote ? "promote" : "hold";
  const blocked = blockers.length > 0;
  const allowed = decision === "promote" && !blocked;

  return {
    winner,
    decision,
    blocked,
    allowed,
    blockers,
    summary: allowed
      ? `mixed wins the replay set and may promote after preserving explicit corrections, rollback binding, and proof bundle expectations.`
      : blocked
        ? `mixed does not yet clear the migration gate; keep old_live active until the replay set, rollback binding, and proof bundle expectations are all satisfied.`
        : winner === "mixed"
          ? "mixed is the best-supported candidate, but the migration gate remains holdback-only for rollout discipline."
          : `${winner} leads the replay set; keep old_live active and treat mixed as a holdback-only candidate until it overtakes the current live policy.`,
    supportRatios,
    supportOrdering,
  };
}

function summarizeExplicitCorrectionProtection(
  replayCases: RouterMigrationReplayCaseSummaryV1[],
): RouterMigrationExplicitCorrectionProtectionV1 {
  const explicitCases = replayCases.filter((replayCase) => replayCase.explicitCorrection === true);
  const preservedCount = explicitCases.filter((replayCase) => replayCase.preservedByVariant.mixed).length;
  const regressionCount = explicitCases.length - preservedCount;
  const blockers = regressionCount > 0
    ? [`mixed regressed ${regressionCount} explicit-correction case${regressionCount === 1 ? "" : "s"}`]
    : [];

  return {
    caseCount: explicitCases.length,
    preservedCount,
    regressionCount,
    protected: regressionCount === 0,
    blockers,
    summary: explicitCases.length === 0
      ? "no explicit-correction cases were included in the migration replay set"
      : regressionCount === 0
        ? `mixed preserved all ${explicitCases.length} explicit-correction case${explicitCases.length === 1 ? "" : "s"}`
        : `mixed regressed ${regressionCount}/${explicitCases.length} explicit-correction case${explicitCases.length === 1 ? "" : "s"}`,
  };
}

function summarizeRollback(input: RouterMigrationComparisonInputV1): RouterMigrationRollbackSummaryV1 {
  const proofBundleFiles = uniqueStrings(Array.isArray(input.proofBundleFiles) ? input.proofBundleFiles : []);
  const rollbackKey = normalizeText(input.rollbackKey);
  const priorLivePackVersion = normalizeNumber(input.priorLivePackVersion);
  const priorLivePackId = normalizeText(input.priorLivePackId);
  const priorRouterChecksum = normalizeText(input.priorRouterChecksum);
  const proofBundleId = normalizeText(input.proofBundleId);
  const available = rollbackKey !== null
    && (priorLivePackVersion !== null || priorLivePackId !== null)
    && priorRouterChecksum !== null
    && proofBundleId !== null;

  return {
    rollbackKey,
    priorLivePackVersion,
    priorLivePackId,
    priorRouterChecksum,
    proofBundleId,
    proofBundleFiles,
    available,
    summary: available
      ? `rollback stays available via ${rollbackKey}; prior live pack ${priorLivePackId ?? priorLivePackVersion ?? "unbound"} and proof bundle ${proofBundleId} remain bound.`
      : `rollback is incomplete: ${[
        rollbackKey === null ? "missing rollback key" : null,
        priorLivePackVersion === null && priorLivePackId === null ? "missing prior live pack identity" : null,
        priorRouterChecksum === null ? "missing prior router checksum" : null,
        proofBundleId === null ? "missing proof bundle id" : null,
      ].filter(Boolean).join(", ") || "no rollback inputs provided"}`,
  };
}

function summarizeProofBundleExpectations(
  input: RouterMigrationComparisonInputV1,
  rollback: RouterMigrationRollbackSummaryV1,
): RouterMigrationProofBundleExpectationsV1 {
  const expectedFiles = [
    "summary.md",
    "status.json",
    "surface-map.json",
    "proposal-report.json",
    "verdict.json",
  ];
  const proofBundleFiles = uniqueStrings(Array.isArray(input.proofBundleFiles) ? input.proofBundleFiles : []);
  const exactArtifactRefs = expectedFiles.length === proofBundleFiles.length
    && expectedFiles.every((fileName) => proofBundleFiles.includes(fileName));
  const checksumsBound = input.variants.old_live.routerChecksum !== null
    && input.variants.base_only.routerChecksum !== null
    && input.variants.mixed.routerChecksum !== null
    && rollback.priorRouterChecksum !== null;
  const rollbackBound = rollback.available && rollback.rollbackKey !== null;
  const proofBundleBound = normalizeText(input.proofBundleId) !== null;

  const blockers = [
    ...exactArtifactRefs ? [] : ["proof bundle does not contain the bounded five-file layout"],
    ...checksumsBound ? [] : ["router checksum binding is incomplete"],
    ...rollbackBound ? [] : ["rollback binding is incomplete"],
    ...proofBundleBound ? [] : ["proof bundle id is missing"],
  ];

  return {
    expectedFiles,
    exactArtifactRefs,
    checksumsBound,
    rollbackBound,
    proofBundleBound,
    summary: exactArtifactRefs && checksumsBound && rollbackBound && proofBundleBound
      ? "proof bundle expectations are met: bounded five-file layout, rollback binding, and router checksums are all explicit"
      : `proof bundle expectations are incomplete: ${blockers.join(", ")}`,
    blockers,
  };
}

export function summarizeRouterMigrationComparisonV1(
  input: RouterMigrationComparisonInputV1,
): RouterMigrationComparisonSummaryV1 {
  const replayCases: RouterMigrationReplayCaseSummaryV1[] = (input.replayCases ?? []).map((replayCase) => ({
    ...replayCase,
    preservedByVariant: {
      old_live: normalizeBoolean(replayCase.preserved.old_live),
      base_only: normalizeBoolean(replayCase.preserved.base_only),
      mixed: normalizeBoolean(replayCase.preserved.mixed),
    },
    weight: variantWeight(replayCase),
  }));

  const variants: Record<RouterMigrationVariantV1, RouterMigrationVariantSummaryV1> = {
    old_live: summarizeVariantSupport("old_live", input.variants.old_live, replayCases),
    base_only: summarizeVariantSupport("base_only", input.variants.base_only, replayCases),
    mixed: summarizeVariantSupport("mixed", input.variants.mixed, replayCases),
  };

  const explicitCorrectionProtection = summarizeExplicitCorrectionProtection(replayCases);
  const rollback = summarizeRollback(input);
  const proofBundleExpectations = summarizeProofBundleExpectations(input, rollback);
  const comparison = summarizeComparisonVerdict(variants, explicitCorrectionProtection, rollback, proofBundleExpectations);
  const replayCaseCount = replayCases.length;
  const variantSummaryLine = [
    `old_live=${variants.old_live.summary}`,
    `base_only=${variants.base_only.summary}`,
    `mixed=${variants.mixed.summary}`,
  ].join(" ");

  return {
    contract: ROUTER_MIGRATION_COMPARISON_CONTRACT,
    state: "target",
    migrationId: input.migrationId,
    proposalId: normalizeText(input.proposalId),
    rollbackKey: rollback.rollbackKey,
    proofBundleId: rollback.proofBundleId,
    proofBundleFiles: rollback.proofBundleFiles,
    replayCaseCount,
    replayCases,
    variants,
    comparison,
    explicitCorrectionProtection,
    rollback,
    proofBundleExpectations,
    summary: `${comparison.summary} ${variantSummaryLine} explicitCorrections=${explicitCorrectionProtection.summary}; rollback=${rollback.summary}; proofBundle=${proofBundleExpectations.summary}`,
  };
}
