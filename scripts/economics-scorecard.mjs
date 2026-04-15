export const ECONOMICS_SCORECARD_CONTRACT = "openclawbrain_economics_scorecard.v1";
export const ECONOMICS_SCORECARD_JSON_FILE = "economics.json";
export const ECONOMICS_SCORECARD_MARKDOWN_FILE = "economics.md";
export const ECONOMICS_SCORECARD_SECTION_LIMIT = 8;

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeNumber(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function round(value, digits = 6) {
  if (!Number.isFinite(value)) {
    return null;
  }
  const factor = 10 ** digits;
  return Math.round(Number(value) * factor) / factor;
}

function formatScalar(value, unit) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  if (unit === "usd") {
    const rounded = round(Number(value), 6);
    return rounded === null ? "n/a" : `$${rounded.toFixed(6)}`;
  }
  if (unit === "ratio") {
    const rounded = round(Number(value), 4);
    return rounded === null ? "n/a" : String(rounded);
  }
  if (typeof value === "number") {
    const rounded = round(value, 6);
    return rounded === null ? "n/a" : String(rounded);
  }
  return String(value);
}

function buildEntry(kind, metric, label, value, unit, source, note) {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    return null;
  }
  return {
    kind,
    metric,
    label,
    value,
    unit,
    source,
    note: normalizeText(note),
  };
}

function compactEntries(entries) {
  return entries.filter(Boolean).slice(0, ECONOMICS_SCORECARD_SECTION_LIMIT);
}

function sumNumericField(items, field) {
  return (Array.isArray(items) ? items : []).reduce((total, item) => {
    const next = Number(item?.[field]);
    return Number.isFinite(next) ? total + next : total;
  }, 0);
}

function latestBundleByKind(latestBundles, kind) {
  if (Array.isArray(latestBundles)) {
    return latestBundles.find((bundle) => bundle?.kind === kind) ?? null;
  }
  if (latestBundles && typeof latestBundles === "object") {
    const mapping = {
      "operator-proof": latestBundles.operatorProof ?? null,
      "recorded-session-replay": latestBundles.recordedSessionReplay ?? null,
      "host-evidence": latestBundles.hostEvidence ?? null,
    };
    return mapping[kind] ?? null;
  }
  return null;
}

function buildSourcesFromSnapshot(snapshot) {
  const latestOperatorProof = latestBundleByKind(snapshot?.latestBundles ?? null, "operator-proof");
  const latestReplayProof = latestBundleByKind(snapshot?.latestBundles ?? null, "recorded-session-replay");
  const latestHostEvidence = latestBundleByKind(snapshot?.latestBundles ?? null, "host-evidence");
  return {
    surface: "proof-cron",
    runKind: "health",
    statusProbeCommand: normalizeText(snapshot?.probe?.command),
    latestOperatorProofPath: normalizeText(latestOperatorProof?.relativePath),
    latestReplayProofPath: normalizeText(latestReplayProof?.relativePath),
    latestHostEvidencePath: normalizeText(latestHostEvidence?.relativePath),
    replayPricingTableVersion: normalizeText(snapshot?.replayCostProxy?.pricingTableVersion),
    replayPricingTablePath: normalizeText(snapshot?.replayCostProxy?.pricingTablePath),
  };
}

function buildSourcesFromAggregate(aggregate) {
  const latestOperatorProof = aggregate?.latestBundles?.operatorProof ?? null;
  const latestReplayProof = aggregate?.latestBundles?.recordedSessionReplay ?? null;
  const latestReplayLane = aggregate?.latestBundles?.recordedSessionReplayLane ?? null;
  const latestHostEvidence = aggregate?.latestBundles?.hostEvidence ?? null;
  return {
    surface: "proof-cron",
    runKind: "nightly",
    statusProbeCommand: null,
    latestOperatorProofPath: normalizeText(latestOperatorProof?.relativePath),
    latestReplayLanePath: normalizeText(latestReplayLane?.relativePath),
    latestReplayProofPath: normalizeText(latestReplayProof?.relativePath),
    replayFocusManifestId: normalizeText(aggregate?.replayMetrics?.focus?.sourceManifestId),
    latestHostEvidencePath: normalizeText(latestHostEvidence?.relativePath),
    replayPricingTableVersion: normalizeText(aggregate?.replayMetrics?.pricingTableVersion),
    replayPricingTablePath: normalizeText(aggregate?.replayMetrics?.pricingTablePath),
  };
}

function buildLegend() {
  return {
    measured: "Direct measurements from the proof-cron surface or underlying proof bundles.",
    derived: "Aggregates, means, and rollups computed from measured proof-cron values.",
    proxy: "Explicit cost or effort estimates that stand in for economics, not primary truth.",
  };
}

function buildBoundedness(measured, derived, proxy) {
  const total = measured.length + derived.length + proxy.length;
  return {
    sectionLimit: ECONOMICS_SCORECARD_SECTION_LIMIT,
    measuredCount: measured.length,
    derivedCount: derived.length,
    proxyCount: proxy.length,
    totalCount: total,
    truncated: false,
  };
}

function summarizeScorecard(scope, scorecard) {
  return `${scope}: ${scorecard.measured.length} measured, ${scorecard.derived.length} derived, ${scorecard.proxy.length} proxy entries; bounded to ${scorecard.boundedness.sectionLimit} per section`;
}

export function buildEconomicsScorecardFromHealthSnapshot(snapshot) {
  const replaySavings = Array.isArray(snapshot?.replaySavings) ? snapshot.replaySavings : [];
  const replayContextCharsTotal = replaySavings.length > 0
    ? sumNumericField(replaySavings, "selectedContextChars")
    : normalizeNumber(snapshot?.performance?.replayContextCharsTotal);
  const replayEstimatedPromptTokensTotal = replaySavings.length > 0
    ? sumNumericField(replaySavings, "estimatedPromptTokens")
    : normalizeNumber(snapshot?.performance?.replayEstimatedPromptTokensTotal);
  const replayRetrievalToolHopCountTotal = replaySavings.length > 0
    ? sumNumericField(replaySavings, "retrievalToolHopCount")
    : normalizeNumber(snapshot?.performance?.replayRetrievalToolHopCountTotal);

  const measured = compactEntries([
    buildEntry(
      "measured",
      "status_probe_duration_ms",
      "status probe duration",
      normalizeNumber(snapshot?.probe?.durationMs),
      "ms",
      "snapshot.probe.durationMs",
      "Observed from the live status probe.",
    ),
    buildEntry(
      "measured",
      "scan_duration_ms",
      "scan duration",
      normalizeNumber(snapshot?.performance?.scanMs),
      "ms",
      "snapshot.performance.scanMs",
      "Observed while scanning the proof-cron surface.",
    ),
    buildEntry(
      "measured",
      "bundle_count",
      "bundle count",
      normalizeNumber(snapshot?.proofInventory?.bundleCount),
      "count",
      "snapshot.proofInventory.bundleCount",
      "Total proof bundles scanned.",
    ),
    buildEntry(
      "measured",
      "operator_proof_count",
      "operator proof count",
      normalizeNumber(snapshot?.proofInventory?.operatorProofCount),
      "count",
      "snapshot.proofInventory.operatorProofCount",
      "Operator proof bundles visible in the scan.",
    ),
    buildEntry(
      "measured",
      "replay_proof_count",
      "replay proof count",
      normalizeNumber(snapshot?.proofInventory?.replayProofCount),
      "count",
      "snapshot.proofInventory.replayProofCount",
      "Recorded-session replay bundles visible in the scan.",
    ),
    buildEntry(
      "measured",
      "artifact_bytes_scanned",
      "artifact bytes scanned",
      normalizeNumber(snapshot?.costProxy?.artifactBytes),
      "bytes",
      "snapshot.costProxy.artifactBytes",
      "Raw bytes scanned across the proof-cron surface.",
    ),
    buildEntry(
      "measured",
      "host_evidence_count",
      "host evidence count",
      normalizeNumber(snapshot?.proofInventory?.hostEvidenceCount),
      "count",
      "snapshot.proofInventory.hostEvidenceCount",
      "Host evidence bundles visible in the scan.",
    ),
  ]);

  const derived = compactEntries([
    buildEntry(
      "derived",
      "operator_step_ms_total",
      "operator step time total",
      normalizeNumber(snapshot?.performance?.operatorStepMsTotal),
      "ms",
      "snapshot.performance.operatorStepMsTotal",
      "Sum of operator step durations.",
    ),
    buildEntry(
      "derived",
      "operator_step_ms_mean",
      "operator step time mean",
      normalizeNumber(snapshot?.performance?.operatorStepMsMean),
      "ms",
      "snapshot.performance.operatorStepMsMean",
      "Mean operator step duration.",
    ),
    buildEntry(
      "derived",
      "replay_winner_score_mean",
      "replay winner score mean",
      normalizeNumber(snapshot?.performance?.replayWinnerScoreMean),
      "score",
      "snapshot.performance.replayWinnerScoreMean",
      "Mean winner score across replay bundles.",
    ),
    buildEntry(
      "derived",
      "replay_context_chars_total",
      "replay context chars total",
      replayContextCharsTotal,
      "chars",
      "snapshot.replaySavings[].selectedContextChars",
      "Selected replay context chars summed across modes.",
    ),
    buildEntry(
      "derived",
      "replay_estimated_prompt_tokens_total",
      "replay estimated prompt tokens total",
      replayEstimatedPromptTokensTotal,
      "tokens",
      "snapshot.replaySavings[].estimatedPromptTokens",
      "Estimated prompt tokens summed across replay bundles.",
    ),
    buildEntry(
      "derived",
      "replay_retrieval_tool_hop_count_total",
      "replay retrieval/tool-hop count total",
      replayRetrievalToolHopCountTotal,
      "count",
      "snapshot.replaySavings[].retrievalToolHopCount",
      "Retrieval/tool-hop proxy count summed across replay bundles.",
    ),
  ]);

  const proxy = compactEntries([
    buildEntry(
      "proxy",
      "proof_minutes_proxy",
      "proof minutes proxy",
      normalizeNumber(snapshot?.costProxy?.proofMinutes),
      "minutes",
      "snapshot.costProxy.proofMinutes",
      "Scan plus operator time normalized into minutes.",
    ),
    buildEntry(
      "proxy",
      "replay_prompt_cost_usd_proxy",
      "replay prompt cost proxy",
      normalizeNumber(snapshot?.replayCostProxy?.estimatedPromptCostUsd),
      "usd",
      "snapshot.replayCostProxy.estimatedPromptCostUsd",
      "Estimated replay prompt cost from the pricing table.",
    ),
    buildEntry(
      "proxy",
      "replay_completion_cost_usd_proxy",
      "replay completion cost proxy",
      normalizeNumber(snapshot?.replayCostProxy?.estimatedCompletionCostUsd),
      "usd",
      "snapshot.replayCostProxy.estimatedCompletionCostUsd",
      "Estimated replay completion cost from the pricing table.",
    ),
    buildEntry(
      "proxy",
      "replay_total_cost_usd_proxy",
      "replay total cost proxy",
      normalizeNumber(snapshot?.replayCostProxy?.estimatedTotalCostUsd),
      "usd",
      "snapshot.replayCostProxy.estimatedTotalCostUsd",
      "Estimated replay prompt plus completion cost.",
    ),
  ]);

  const scorecard = {
    contract: ECONOMICS_SCORECARD_CONTRACT,
    scope: "health",
    surface: "proof-cron",
    surfaceState: "shipped",
    generatedAt: normalizeText(snapshot?.generatedAt),
    sources: buildSourcesFromSnapshot(snapshot),
    legend: buildLegend(),
    boundedness: buildBoundedness(measured, derived, proxy),
    measured,
    derived,
    proxy,
  };

  return {
    ...scorecard,
    summary: summarizeScorecard("proof-cron health", scorecard),
  };
}

export function buildEconomicsScorecardFromNightlyAggregate(aggregate) {
  const measured = compactEntries([
    buildEntry(
      "measured",
      "bundle_count",
      "bundle count",
      normalizeNumber(Array.isArray(aggregate?.bundles) ? aggregate.bundles.length : null),
      "count",
      "aggregate.bundles.length",
      "Total proof bundles scanned.",
    ),
    buildEntry(
      "measured",
      "operator_proof_count",
      "operator proof count",
      normalizeNumber(aggregate?.bundleTypeCounts?.operatorProof),
      "count",
      "aggregate.bundleTypeCounts.operatorProof",
      "Operator proof bundles visible in the nightly aggregate.",
    ),
    buildEntry(
      "measured",
      "recorded_session_replay_count",
      "recorded-session replay count",
      normalizeNumber(aggregate?.bundleTypeCounts?.recordedSessionReplay),
      "count",
      "aggregate.bundleTypeCounts.recordedSessionReplay",
      "Recorded-session replay bundles visible in the nightly aggregate.",
    ),
    buildEntry(
      "measured",
      "host_evidence_count",
      "host evidence count",
      normalizeNumber(aggregate?.bundleTypeCounts?.hostEvidence),
      "count",
      "aggregate.bundleTypeCounts.hostEvidence",
      "Host evidence bundles visible in the nightly aggregate.",
    ),
    buildEntry(
      "measured",
      "validation_ok_count",
      "validation ok count",
      normalizeNumber(aggregate?.validationCounts?.ok),
      "count",
      "aggregate.validationCounts.ok",
      "Bundles that validated cleanly.",
    ),
    buildEntry(
      "measured",
      "validation_fail_count",
      "validation fail count",
      normalizeNumber(aggregate?.validationCounts?.fail),
      "count",
      "aggregate.validationCounts.fail",
      "Bundles that failed validation.",
    ),
  ]);

  const derived = compactEntries([
    buildEntry(
      "derived",
      "operator_step_ms_total",
      "operator step time total",
      normalizeNumber(aggregate?.operatorMetrics?.stepMsTotal),
      "ms",
      "aggregate.operatorMetrics.stepMsTotal",
      "Sum of operator step durations.",
    ),
    buildEntry(
      "derived",
      "operator_step_ms_mean",
      "operator step time mean",
      normalizeNumber(aggregate?.operatorMetrics?.stepMsMean),
      "ms",
      "aggregate.operatorMetrics.stepMsMean",
      "Mean operator step duration.",
    ),
    buildEntry(
      "derived",
      "replay_focus_trace_count",
      "replay focus trace count",
      normalizeNumber(
        aggregate?.replayMetrics?.focus?.successfulTraceCount
          ?? aggregate?.replayMetrics?.focus?.requestedTraceCount,
      ),
      "count",
      "aggregate.replayMetrics.focus.successfulTraceCount",
      "Trace count for the replay focus surface that reporting should optimize over.",
    ),
    buildEntry(
      "derived",
      "replay_focus_better_count",
      "replay focus better count",
      normalizeNumber(aggregate?.replayMetrics?.focus?.candidateUtilityVsBaselineCounts?.better),
      "count",
      "aggregate.replayMetrics.focus.candidateUtilityVsBaselineCounts.better",
      "How often learned_route beats graph_prior_only on the replay focus surface.",
    ),
    buildEntry(
      "derived",
      "replay_focus_worse_count",
      "replay focus worse count",
      normalizeNumber(aggregate?.replayMetrics?.focus?.candidateUtilityVsBaselineCounts?.worse),
      "count",
      "aggregate.replayMetrics.focus.candidateUtilityVsBaselineCounts.worse",
      "How often learned_route loses to graph_prior_only on the replay focus surface.",
    ),
    buildEntry(
      "derived",
      "replay_focus_tie_or_better_rate",
      "replay focus tie-or-better rate",
      normalizeNumber(aggregate?.replayMetrics?.focus?.tieOrBetterRate),
      "ratio",
      "aggregate.replayMetrics.focus.tieOrBetterRate",
      "Primary optimize-over rate on the replay focus surface.",
    ),
    buildEntry(
      "derived",
      "replay_focus_regression_rate",
      "replay focus regression rate",
      normalizeNumber(aggregate?.replayMetrics?.focus?.regressionRate),
      "ratio",
      "aggregate.replayMetrics.focus.regressionRate",
      "Guardrail rate for learned_route losses on the replay focus surface.",
    ),
    buildEntry(
      "derived",
      "replay_focus_required_context_recall_delta",
      "replay focus required-context recall delta",
      normalizeNumber(aggregate?.replayMetrics?.focus?.requiredContextRecallDelta),
      "ratio",
      "aggregate.replayMetrics.focus.requiredContextRecallDelta",
      "Recall delta between learned_route and graph_prior_only on the replay focus surface.",
    ),
    buildEntry(
      "derived",
      "replay_context_chars_total",
      "replay context chars total",
      normalizeNumber(aggregate?.replayMetrics?.selectedContextCharsTotal),
      "chars",
      "aggregate.replayMetrics.selectedContextCharsTotal",
      "Selected replay context chars summed across replay bundles.",
    ),
    buildEntry(
      "derived",
      "replay_estimated_prompt_tokens_total",
      "replay estimated prompt tokens total",
      normalizeNumber(aggregate?.replayMetrics?.estimatedPromptTokensTotal),
      "tokens",
      "aggregate.replayMetrics.estimatedPromptTokensTotal",
      "Estimated prompt tokens summed across replay bundles.",
    ),
  ]);

  const proxy = compactEntries([
    buildEntry(
      "proxy",
      "proof_minutes_proxy",
      "proof minutes proxy",
      normalizeNumber(aggregate?.costProxy?.proofMinutes),
      "minutes",
      "aggregate.costProxy.proofMinutes",
      "Scan plus operator time normalized into minutes.",
    ),
    buildEntry(
      "proxy",
      "replay_prompt_cost_usd_proxy",
      "replay prompt cost proxy",
      normalizeNumber(aggregate?.replayMetrics?.estimatedPromptCostUsdTotal),
      "usd",
      "aggregate.replayMetrics.estimatedPromptCostUsdTotal",
      "Estimated replay prompt cost from the pricing table.",
    ),
    buildEntry(
      "proxy",
      "replay_completion_cost_usd_proxy",
      "replay completion cost proxy",
      normalizeNumber(aggregate?.replayMetrics?.estimatedCompletionCostUsdTotal),
      "usd",
      "aggregate.replayMetrics.estimatedCompletionCostUsdTotal",
      "Estimated replay completion cost from the pricing table.",
    ),
    buildEntry(
      "proxy",
      "replay_total_cost_usd_proxy",
      "replay total cost proxy",
      normalizeNumber(aggregate?.replayMetrics?.estimatedTotalCostUsdTotal),
      "usd",
      "aggregate.replayMetrics.estimatedTotalCostUsdTotal",
      "Estimated replay prompt plus completion cost.",
    ),
  ]);

  const scorecard = {
    contract: ECONOMICS_SCORECARD_CONTRACT,
    scope: "nightly",
    surface: "proof-cron",
    surfaceState: "shipped",
    generatedAt: normalizeText(aggregate?.generatedAt),
    sources: buildSourcesFromAggregate(aggregate),
    legend: buildLegend(),
    boundedness: buildBoundedness(measured, derived, proxy),
    measured,
    derived,
    proxy,
  };

  return {
    ...scorecard,
    summary: summarizeScorecard("proof-cron nightly", scorecard),
  };
}

function renderSectionRows(entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return ["| none | n/a | n/a | n/a | n/a |"];
  }

  return entries.map((entry) => `| ${entry.metric} | ${formatScalar(entry.value, entry.unit)} | ${entry.unit ?? "n/a"} | ${entry.source ?? "n/a"} | ${entry.note ?? ""} |`);
}

function renderSourcesMarkdown(sources) {
  return [
    `- surface: \`${sources.surface ?? "n/a"}\``,
    `- run kind: \`${sources.runKind ?? "n/a"}\``,
    `- status probe command: ${sources.statusProbeCommand ? `\`${sources.statusProbeCommand}\`` : "n/a"}`,
    `- latest operator proof path: ${sources.latestOperatorProofPath ?? "n/a"}`,
    `- latest replay lane path: ${sources.latestReplayLanePath ?? "n/a"}`,
    `- latest replay proof path: ${sources.latestReplayProofPath ?? "n/a"}`,
    `- replay focus manifest id: ${sources.replayFocusManifestId ?? "n/a"}`,
    `- latest host evidence path: ${sources.latestHostEvidencePath ?? "n/a"}`,
    `- replay pricing table version: ${sources.replayPricingTableVersion ?? "n/a"}`,
    `- replay pricing table path: ${sources.replayPricingTablePath ?? "n/a"}`,
  ];
}

export function buildEconomicsScorecardMarkdown(scorecard) {
  const measured = Array.isArray(scorecard?.measured) ? scorecard.measured : [];
  const derived = Array.isArray(scorecard?.derived) ? scorecard.derived : [];
  const proxy = Array.isArray(scorecard?.proxy) ? scorecard.proxy : [];

  return [
    "# OpenClawBrain economics scorecard",
    "",
    `- contract: \`${scorecard?.contract ?? ECONOMICS_SCORECARD_CONTRACT}\``,
    `- scope: \`${scorecard?.scope ?? "n/a"}\``,
    `- surface: \`${scorecard?.surface ?? "n/a"}\``,
    `- surface state: \`${scorecard?.surfaceState ?? "n/a"}\``,
    `- generated at: ${scorecard?.generatedAt ?? "n/a"}`,
    `- bounded entries per section: ${scorecard?.boundedness?.sectionLimit ?? ECONOMICS_SCORECARD_SECTION_LIMIT}`,
    `- measured entries: ${measured.length}`,
    `- derived entries: ${derived.length}`,
    `- proxy entries: ${proxy.length}`,
    `- labels: measured / derived / proxy`,
    "",
    `## Measured (${measured.length})`,
    "| metric | value | unit | source | note |",
    "| --- | ---: | --- | --- | --- |",
    ...renderSectionRows(measured),
    "",
    `## Derived (${derived.length})`,
    "| metric | value | unit | source | note |",
    "| --- | ---: | --- | --- | --- |",
    ...renderSectionRows(derived),
    "",
    `## Proxy (${proxy.length})`,
    "| metric | value | unit | source | note |",
    "| --- | ---: | --- | --- | --- |",
    ...renderSectionRows(proxy),
    "",
    "## Sources",
    ...renderSourcesMarkdown(scorecard?.sources ?? {}),
    "",
    "## Legend",
    `- measured: ${scorecard?.legend?.measured ?? buildLegend().measured}`,
    `- derived: ${scorecard?.legend?.derived ?? buildLegend().derived}`,
    `- proxy: ${scorecard?.legend?.proxy ?? buildLegend().proxy}`,
    scorecard?.summary ? "" : null,
    scorecard?.summary ? `- summary: ${scorecard.summary}` : null,
    scorecard?.boundedness?.truncated ? "- note: some sections were truncated to keep the scorecard bounded" : null,
  ].filter((line) => line !== null).join("\n") + "\n";
}

export function isEconomicsScorecard(value) {
  return Boolean(value)
    && typeof value === "object"
    && value.contract === ECONOMICS_SCORECARD_CONTRACT
    && Array.isArray(value.measured)
    && Array.isArray(value.derived)
    && Array.isArray(value.proxy)
    && typeof value.summary === "string";
}

export { formatScalar as formatEconomicsScorecardValue };
