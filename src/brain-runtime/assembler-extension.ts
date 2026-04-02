import type { AssembleContextResult, AssembledSummaryMetadata } from "../assembler.js";
import type { ContextEngine, AgentMessage } from "../openclaw-sdk-compat.js";
import type {
  BrainAgentIdentity,
  BrainCompileReportV1,
  BrainDropReason,
  BrainDropStage,
  BrainFitStrategy,
  BrainFittingDropReason,
  BrainInterruptionMetadata,
  BrainInterruptionStage,
  BrainPrefetchDecision,
  DecisionRouteTrace,
  DecisionTrace,
  DecisionTraceInjectedNodeSummary,
  TraversalResult,
} from "../brain-core/types.js";
import { buildBrainCompileReport } from "../brain-core/trace.js";
import type { BrainService } from "./service.js";
import { decideSummaryRouting } from "./summary-routing-policy.js";

export type BrainAssemblyRouteMode =
  | "use_brain"
  | "shadow"
  | "skip_no_query"
  | "skip_short_static_lookup"
  | "skip_no_embedding"
  | "skip_uninitialized"
  | "skip_budget_too_small";

export type BrainAssemblyOutcomeMode =
  | BrainAssemblyRouteMode
  | "skip_query_returned_no_nodes"
  | "skip_deadline_before_query"
  | "skip_deadline_after_query"
  | "skip_deadline_before_injection"
  | "partial_query_interruption"
  | "partial_deadline_after_query"
  | "partial_deadline_before_injection";

export type BrainAssemblyDecision = {
  mode: BrainAssemblyRouteMode;
  queryText: string;
};

export type BrainAssembledContextResult = AssembleContextResult;

const COMPACT_INJECTED_PREVIEW_CHARS = 96;
const INTERRUPTED_SERVE_MAX_CONTEXT_CHARS = 1024;
const INTERRUPTION_REASON_SOFT_COMPILE_DEADLINE = "soft_compile_deadline";

type BudgetedBrainContext = {
  brainContext: string;
  injectedChars: number;
  droppedChars: number;
  contextClipped: boolean;
  fitStrategy?: BrainFitStrategy | null;
  retrievedNodeCount?: number | null;
  fittedNodeCount?: number | null;
  droppedNodeCount?: number | null;
  fittingDropReasons?: Partial<Record<BrainFittingDropReason, number>> | null;
};

type BudgetDecisionDetails = {
  budgetFraction?: number | null;
  maxContextChars?: number | null;
  queryBudgetChars?: number | null;
  injectedChars?: number | null;
  droppedChars?: number | null;
  contextClipped?: boolean;
  fitStrategy?: BrainFitStrategy | null;
  retrievedNodeCount?: number | null;
  fittedNodeCount?: number | null;
  droppedNodeCount?: number | null;
  fittingDropReasons?: Partial<Record<BrainFittingDropReason, number>> | null;
};

type CompileDecisionDetails = {
  interruption?: BrainInterruptionMetadata | null;
  queryInterrupted?: boolean | null;
  interruptionStage?: BrainInterruptionStage | null;
  interruptionReason?: string | null;
  servedPartial?: boolean | null;
  compileElapsedMs?: number | null;
  compileDeadlineMs?: number | null;
  compileDeadlineHit?: boolean | null;
  brainDropReason?: BrainDropReason | null;
  brainDropStage?: BrainDropStage | null;
};

type InterruptionStage = NonNullable<DecisionRouteTrace["selectionMetadata"]["interruptionStage"]>;

type InterruptionDecisionDetails = {
  queryInterrupted?: boolean | null;
  interruptionStage?: InterruptionStage | null;
  interruptionReason?: string | null;
  servedPartial?: boolean | null;
};

type TraceCarryoverDecisionDetails = Partial<Pick<
  DecisionRouteTrace["selectionMetadata"],
  | "chosenStopCount"
  | "forcedStopCount"
  | "branchOutcomeSummary"
  | "droppedProposalCount"
  | "droppedProposalReasons"
  | "interruptionAccounting"
>>;

type CompileCheckpoint = {
  compileElapsedMs: number;
  compileDeadlineMs?: number | null;
  compileDeadlineHit?: boolean | null;
};

type AssemblyDecisionDetails = BudgetDecisionDetails & CompileDecisionDetails & InterruptionDecisionDetails & TraceCarryoverDecisionDetails & {
  prefetch?: BrainPrefetchDecision | null;
};

function decisionFooter(mode: BrainAssemblyOutcomeMode): string {
  switch (mode) {
    case "use_brain":
      return "[brain] used graph retrieval for this turn.";
    case "shadow":
      return "[brain shadow] recorded routing without injecting learned context.";
    case "skip_no_query":
      return "[brain] bypassed: no user query text.";
    case "skip_short_static_lookup":
      return "[brain] bypassed: short static lookup.";
    case "skip_no_embedding":
      return "[brain] bypassed: embeddings unavailable.";
    case "skip_uninitialized":
      return "[brain] bypassed: brain uninitialized or disabled.";
    case "skip_budget_too_small":
      return "[brain] bypassed: token budget too small.";
    case "skip_query_returned_no_nodes":
      return "[brain] bypassed: query returned no nodes.";
    case "skip_deadline_before_query":
      return "[brain] bypassed: soft compile deadline hit before query.";
    case "skip_deadline_after_query":
      return "[brain] bypassed: soft compile deadline hit after query.";
    case "skip_deadline_before_injection":
      return "[brain] bypassed: soft compile deadline hit before injection.";
    case "partial_query_interruption":
      return "[brain] partial serve: query interrupted under budget pressure; injected committed prefix.";
    case "partial_deadline_after_query":
      return "[brain] partial serve: soft compile deadline hit after query; injected committed prefix.";
    case "partial_deadline_before_injection":
      return "[brain] partial serve: soft compile deadline hit before injection; injected committed prefix.";
  }
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function extractText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter((item): item is { type?: unknown; text?: unknown } => !!item && typeof item === "object")
    .map((item) => (item.type === "text" && typeof item.text === "string" ? item.text : ""))
    .join("\n")
    .trim();
}

function buildLegacyBrainContextBlock(result: TraversalResult): string {
  const corrections = result.fired.filter((node) => node.kind === "correction");
  const playbooks = result.fired.filter((node) => node.kind === "workflow" || node.kind === "toolcard");
  const evidence = result.fired.filter((node) => node.kind !== "correction" && node.kind !== "workflow" && node.kind !== "toolcard");

  const sections = [
    "OpenClawBrain retrieved context. Prefer correction cards over conflicting heuristics when directly relevant.",
    "",
    "## Correction Cards",
    corrections.length > 0 ? corrections.map((node) => `- ${node.content}`).join("\n") : "- none",
    "",
    "## Route-Selected Evidence",
    evidence.length > 0 ? evidence.map((node) => `- [${node.kind}] ${node.content}`).join("\n") : "- none",
    "",
    "## Toolcards And Playbooks",
    playbooks.length > 0 ? playbooks.map((node) => `- [${node.kind}] ${node.content}`).join("\n") : "- none",
    "",
    "## Transcript Support",
    "- Use the LCM transcript and summary context below for chronology and grounding.",
    "",
    `Trace: ${result.trace.footer}`,
  ];
  return sections.join("\n");
}

function compactPreview(content: string, maxChars = COMPACT_INJECTED_PREVIEW_CHARS): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "(preview unavailable)";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(1, maxChars - 1))}…`;
}

function formatCountMap(counts: Record<string, number | undefined>): string {
  const entries = Object.entries(counts)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number" && entry[1] > 0);
  if (entries.length === 0) {
    return "none";
  }
  return entries.map(([label, count]) => `${label} ${count}`).join(", ");
}

function formatSourceUri(sourceUri: string | null): string {
  return sourceUri?.trim() || "unknown source";
}

function buildCorrectionSummaryLine(summary: DecisionTraceInjectedNodeSummary): string {
  return `- [${summary.trust}] ${compactPreview(summary.contentPreview)}`;
}

function buildTypedSummaryLine(summary: DecisionTraceInjectedNodeSummary): string {
  return `- [${summary.kind}] ${compactPreview(summary.contentPreview)}`;
}

function buildAuditOverview(routeTrace: DecisionRouteTrace): string {
  const sourceCount = routeTrace.sourceSummary.sourceUris.length;
  const sourceLabel = sourceCount === 1 ? "1 source" : `${sourceCount} sources`;
  return `- Pack ${routeTrace.activePackId ?? "unknown"} · ${routeTrace.sourceSummary.injectedCount} injected nodes · ${sourceLabel}`;
}

function buildStructuredBrainContextBlock(result: TraversalResult, routeTrace: DecisionRouteTrace): string {
  const corrections = routeTrace.injectedNodeSummaries.filter((node) => node.kind === "correction");
  const playbooks = routeTrace.injectedNodeSummaries
    .filter((node) => node.kind === "workflow" || node.kind === "toolcard");
  const evidence = routeTrace.injectedNodeSummaries
    .filter((node) => node.kind !== "correction" && node.kind !== "workflow" && node.kind !== "toolcard");

  const sections = [
    "OpenClawBrain retrieved context. Prefer correction cards over conflicting heuristics when directly relevant.",
    "",
    "## Correction Cards",
    corrections.length > 0 ? corrections.map(buildCorrectionSummaryLine).join("\n") : "- none",
    "",
    "## Route-Selected Summaries",
    evidence.length > 0 ? evidence.map(buildTypedSummaryLine).join("\n") : "- none",
    "",
    "## Toolcards And Playbooks",
    playbooks.length > 0 ? playbooks.map(buildTypedSummaryLine).join("\n") : "- none",
    "",
    "## Provenance And Audit",
    `- ${result.trace.footer}`,
    buildAuditOverview(routeTrace),
    `- Kinds: ${formatCountMap(routeTrace.sourceSummary.kinds)}`,
    `- Trusts: ${formatCountMap(routeTrace.sourceSummary.trusts)}`,
    routeTrace.injectedNodeSummaries
      .map((node) => `- \`${node.nodeId}\` [${node.kind}/${node.trust}] from ${formatSourceUri(node.sourceUri)}`)
      .join("\n"),
    "",
    "## Transcript Support",
    "- Use the LCM transcript and summary context below for chronology and grounding.",
  ];
  return sections.join("\n");
}

function orderedInjectedNodeSummaries(routeTrace: DecisionRouteTrace): DecisionTraceInjectedNodeSummary[] {
  const corrections = routeTrace.injectedNodeSummaries.filter((node) => node.kind === "correction");
  const evidence = routeTrace.injectedNodeSummaries
    .filter((node) => node.kind !== "correction" && node.kind !== "workflow" && node.kind !== "toolcard");
  const playbooks = routeTrace.injectedNodeSummaries
    .filter((node) => node.kind === "workflow" || node.kind === "toolcard");
  return [...corrections, ...evidence, ...playbooks];
}

function buildCompactFittedNodeLine(summary: DecisionTraceInjectedNodeSummary): string {
  return `- [${summary.kind}/${summary.trust}] ${compactPreview(summary.contentPreview)}`;
}

function buildCompactPartialFittedNodeLine(summary: DecisionTraceInjectedNodeSummary): string {
  return `- [${summary.kind}] ${compactPreview(summary.contentPreview, 56)}`;
}

function buildCompactStructuredBrainContext(
  injectedNodeSummaries: DecisionTraceInjectedNodeSummary[],
): string {
  if (injectedNodeSummaries.length === 0) {
    return "";
  }
  return [
    "[brain]",
    ...injectedNodeSummaries.map(buildCompactFittedNodeLine),
  ].join("\n");
}

function buildBrainContextBlock(result: TraversalResult): string {
  const routeTrace = result.trace.routeTrace;
  if (!routeTrace || routeTrace.injectedNodeSummaries.length === 0) {
    return buildLegacyBrainContextBlock(result);
  }
  return buildStructuredBrainContextBlock(result, routeTrace);
}

function buildPartialServeNotice(stage: InterruptionStage): string {
  switch (stage) {
    case "traversal":
      return "[brain partial] Query traversal stopped under deadline pressure. Committed prefix only.";
    case "query":
      return "[brain partial] Soft compile deadline hit after query. Committed prefix only.";
    case "injection":
      return "[brain partial] Soft compile deadline hit before full injection. Committed prefix only.";
    default:
      return "[brain partial] Committed prefix only.";
  }
}

function buildPartialAuditLine(stage: InterruptionStage): string {
  switch (stage) {
    case "traversal":
      return "- Partial serve reason: traversal stopped under deadline pressure.";
    case "query":
      return "- Partial serve reason: soft compile deadline hit after query.";
    case "injection":
      return "- Partial serve reason: soft compile deadline hit before full injection.";
    default:
      return "- Partial serve reason: compile interruption.";
  }
}

function buildLegacyPartialBrainContextBlock(result: TraversalResult, stage: InterruptionStage): string {
  const corrections = result.fired.filter((node) => node.kind === "correction");
  const playbooks = result.fired.filter((node) => node.kind === "workflow" || node.kind === "toolcard");
  const evidence = result.fired.filter((node) => node.kind !== "correction" && node.kind !== "workflow" && node.kind !== "toolcard");

  const sections = [
    buildPartialServeNotice(stage),
    "",
    "## Correction Cards",
    corrections.length > 0 ? corrections.map((node) => `- ${compactPreview(node.content)}`).join("\n") : "- none",
    "",
    "## Route-Selected Evidence",
    evidence.length > 0 ? evidence.map((node) => `- [${node.kind}] ${compactPreview(node.content)}`).join("\n") : "- none",
    "",
    "## Toolcards And Playbooks",
    playbooks.length > 0 ? playbooks.map((node) => `- [${node.kind}] ${compactPreview(node.content)}`).join("\n") : "- none",
    "",
    "## Audit",
    `- ${result.trace.footer}`,
    buildPartialAuditLine(stage),
  ];
  return sections.join("\n");
}

function buildStructuredPartialBrainContextBlock(
  result: TraversalResult,
  routeTrace: DecisionRouteTrace,
  stage: InterruptionStage,
): string {
  const corrections = routeTrace.injectedNodeSummaries.filter((node) => node.kind === "correction");
  const playbooks = routeTrace.injectedNodeSummaries
    .filter((node) => node.kind === "workflow" || node.kind === "toolcard");
  const evidence = routeTrace.injectedNodeSummaries
    .filter((node) => node.kind !== "correction" && node.kind !== "workflow" && node.kind !== "toolcard");

  const sections = [
    buildPartialServeNotice(stage),
    "",
    "## Correction Cards",
    corrections.length > 0 ? corrections.map(buildCorrectionSummaryLine).join("\n") : "- none",
    "",
    "## Route-Selected Summaries",
    evidence.length > 0 ? evidence.map(buildTypedSummaryLine).join("\n") : "- none",
    "",
    "## Toolcards And Playbooks",
    playbooks.length > 0 ? playbooks.map(buildTypedSummaryLine).join("\n") : "- none",
    "",
    "## Audit",
    `- ${result.trace.footer}`,
    buildAuditOverview(routeTrace),
    buildPartialAuditLine(stage),
  ];
  return sections.join("\n");
}

function buildPartialBrainContextBlock(result: TraversalResult, stage: InterruptionStage): string {
  const routeTrace = result.trace.routeTrace;
  if (!routeTrace || routeTrace.injectedNodeSummaries.length === 0) {
    return buildLegacyPartialBrainContextBlock(result, stage);
  }
  return buildStructuredPartialBrainContextBlock(result, routeTrace, stage);
}

function buildCompactStructuredPartialBrainContext(params: {
  routeTrace: DecisionRouteTrace;
  stage: InterruptionStage;
  fittedNodeSummaries: DecisionTraceInjectedNodeSummary[];
}): string {
  const retrievedNodeCount = params.routeTrace.injectedNodeSummaries.length;
  const fittedNodeCount = params.fittedNodeSummaries.length;
  const droppedNodeCount = Math.max(0, retrievedNodeCount - fittedNodeCount);
  const stageDetail = (() => {
    switch (params.stage) {
      case "traversal":
        return "query interrupted";
      case "query":
        return "deadline after query";
      case "injection":
        return "deadline before injection";
      default:
        return "committed prefix";
    }
  })();
  const lines = [
    `[brain partial] ${stageDetail}.`,
    `- committed prefix ${fittedNodeCount}/${retrievedNodeCount} nodes${droppedNodeCount > 0 ? `; omitted ${droppedNodeCount}` : ""}.`,
  ];
  if (droppedNodeCount > 0) {
    lines.push("- omitted context was not completed before the deadline.");
  }
  lines.push(...params.fittedNodeSummaries.map(buildCompactPartialFittedNodeLine));
  return lines.join("\n");
}

function buildSummaryRoutingPrompt(
  mode: ReturnType<typeof decideSummaryRouting>["mode"],
  summaryMetadata?: AssembledSummaryMetadata,
): string | undefined {
  const branchHeavy =
    (summaryMetadata?.branchCount ?? 0) > 1 ||
    (summaryMetadata?.snapshotCount ?? 0) > 0 ||
    (summaryMetadata?.hasNonFreshSummaries ?? false) ||
    (summaryMetadata?.hasTruthConflict ?? false);
  const staleSummaryCount = Object.entries(summaryMetadata?.freshnessStateCounts ?? {})
    .filter(([freshnessState]) => freshnessState !== "fresh")
    .reduce((count, [, value]) => count + (value ?? 0), 0);

  switch (mode) {
    case "summary_suffices":
      return branchHeavy
        ? `This turn looks like a broad recap over branch-heavy compacted history${staleSummaryCount > 0 ? ` with ${staleSummaryCount} stale or superseded summary(s).` : "."} Summary-level context is a reasonable starting point, but expand toward source before making exact claims or resolving current-truth conflicts.`
        : "This turn looks like a broad recap. Summary-level context is a reasonable starting point unless the user asks for exact proof or current-truth conflict resolution.";
    case "prefer_typed_memory":
      return branchHeavy
        ? `This turn looks current-truth or conflict-sensitive${staleSummaryCount > 0 ? `, and ${staleSummaryCount} summary(s) are stale or superseded.` : "."} Prefer explicit correction cards and typed memory over summary recap; if typed memory is missing or the branch history is forked/snapshotted, expand toward source before asserting specifics.`
        : "This turn looks current-truth or conflict-sensitive. Prefer explicit correction cards and typed memory over summary recap; if typed memory is missing, expand toward source before asserting specifics.";
    case "expand_to_source":
      return branchHeavy
        ? `This turn looks precision-sensitive against branch-heavy compacted history${staleSummaryCount > 0 ? ` with ${staleSummaryCount} stale or superseded summary(s).` : "."} Use summaries only to locate the region, then expand toward source material and snapshots before asserting exact details.`
        : "This turn looks precision-sensitive against compacted history. Use summaries only to locate the region, then expand toward source material before asserting exact details.";
    default:
      return undefined;
  }
}

function normalizeBudgetFraction(value: number): number {
  if (!Number.isFinite(value)) {
    return 0.3;
  }
  return Math.min(1, Math.max(0, value));
}

function resolveInterruptedMaxContextChars(maxContextChars?: number): number {
  if (typeof maxContextChars !== "number" || !Number.isFinite(maxContextChars)) {
    return INTERRUPTED_SERVE_MAX_CONTEXT_CHARS;
  }
  return Math.max(0, Math.min(Math.floor(maxContextChars), INTERRUPTED_SERVE_MAX_CONTEXT_CHARS));
}

export function resolveBrainQueryBudgetChars(tokenBudget: number, budgetFraction: number): number {
  // Retrieval budget stays separate from the final injected-block cap so
  // zero/tight caps remain attributable without collapsing the query itself.
  return Math.max(256, Math.floor(tokenBudget * 4 * normalizeBudgetFraction(budgetFraction)));
}

function clipTextToLimit(text: string, limit: number): string {
  if (text.length <= limit) {
    return text;
  }
  if (limit === 0) {
    return "";
  }
  const hardSlice = text.slice(0, limit);
  const lineBoundary = hardSlice.lastIndexOf("\n");
  const clipped = (
    lineBoundary >= Math.floor(limit * 0.6)
      ? hardSlice.slice(0, lineBoundary)
      : hardSlice
  ).trimEnd();
  return clipped.length > 0 ? clipped : hardSlice.trimEnd();
}

function applyLegacyMaxContextChars(text: string, maxContextChars?: number): BudgetedBrainContext {
  if (typeof maxContextChars !== "number" || !Number.isFinite(maxContextChars)) {
    return {
      brainContext: text,
      injectedChars: text.length,
      droppedChars: 0,
      contextClipped: false,
    };
  }

  const limit = Math.max(0, Math.floor(maxContextChars));
  const brainContext = clipTextToLimit(text, limit);
  return {
    brainContext,
    injectedChars: brainContext.length,
    droppedChars: Math.max(0, text.length - brainContext.length),
    contextClipped: brainContext.length < text.length,
    fitStrategy: "legacy_raw_clip",
  };
}

function budgetDecisionDetails(params: {
  budgetFraction?: number | null;
  maxContextChars?: number;
  queryBudgetChars?: number | null;
  budgetedBrainContext?: BudgetedBrainContext;
}): BudgetDecisionDetails {
  const details: BudgetDecisionDetails = {};
  if (typeof params.budgetFraction === "number") {
    details.budgetFraction = params.budgetFraction;
  }
  if (params.maxContextChars !== undefined) {
    details.maxContextChars = params.maxContextChars;
  }
  if (typeof params.queryBudgetChars === "number") {
    details.queryBudgetChars = params.queryBudgetChars;
  }
  if (params.budgetedBrainContext) {
    details.injectedChars = params.budgetedBrainContext.injectedChars;
    details.droppedChars = params.budgetedBrainContext.droppedChars;
    details.contextClipped = params.budgetedBrainContext.contextClipped;
    details.fitStrategy = params.budgetedBrainContext.fitStrategy;
    details.retrievedNodeCount = params.budgetedBrainContext.retrievedNodeCount;
    details.fittedNodeCount = params.budgetedBrainContext.fittedNodeCount;
    details.droppedNodeCount = params.budgetedBrainContext.droppedNodeCount;
    details.fittingDropReasons = params.budgetedBrainContext.fittingDropReasons;
  }
  return details;
}

function resolveCompileDeadlineMs(value: number | null | undefined): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.max(0, Math.floor(value));
}

function captureCompileCheckpoint(startedAt: number, compileDeadlineMs?: number): CompileCheckpoint {
  const compileElapsedMs = Math.max(0, Date.now() - startedAt);
  if (compileDeadlineMs === undefined) {
    return { compileElapsedMs };
  }
  return {
    compileElapsedMs,
    compileDeadlineMs,
    compileDeadlineHit: compileElapsedMs >= compileDeadlineMs,
  };
}

function interruptionDecisionDetails(
  interruption?: BrainInterruptionMetadata | null,
): Pick<
  CompileDecisionDetails,
  "interruption" | "queryInterrupted" | "interruptionStage" | "interruptionReason" | "servedPartial"
> {
  if (!interruption) {
    return {};
  }
  return {
    interruption,
    queryInterrupted: true,
    interruptionStage: interruption.stage,
    interruptionReason: interruption.reason,
    servedPartial: interruption.servedPartial,
  };
}

function traceCarryoverDecisionDetails(
  trace: DecisionTrace | null | undefined,
): TraceCarryoverDecisionDetails {
  const selectionMetadata = trace?.routeTrace?.selectionMetadata;
  if (!selectionMetadata) {
    return {};
  }
  return {
    chosenStopCount: selectionMetadata.chosenStopCount ?? null,
    forcedStopCount: selectionMetadata.forcedStopCount ?? null,
    branchOutcomeSummary: selectionMetadata.branchOutcomeSummary
      ? {
          ...selectionMetadata.branchOutcomeSummary,
          terminationReasons: selectionMetadata.branchOutcomeSummary.terminationReasons
            ? { ...selectionMetadata.branchOutcomeSummary.terminationReasons }
            : null,
        }
      : null,
    droppedProposalCount: selectionMetadata.droppedProposalCount ?? null,
    droppedProposalReasons: selectionMetadata.droppedProposalReasons
      ? { ...selectionMetadata.droppedProposalReasons }
      : null,
    interruptionAccounting: selectionMetadata.interruptionAccounting
      ? {
          ...selectionMetadata.interruptionAccounting,
          droppedFrontierNodeIds: [...selectionMetadata.interruptionAccounting.droppedFrontierNodeIds],
          droppedProposalNodeIds: [...selectionMetadata.interruptionAccounting.droppedProposalNodeIds],
          droppedProposalReasons: { ...selectionMetadata.interruptionAccounting.droppedProposalReasons },
        }
      : null,
  };
}

function createInterruption(params: {
  stage: BrainInterruptionStage;
  reason: string;
  servedPartial?: boolean;
}): BrainInterruptionMetadata {
  return {
    interrupted: true,
    stage: params.stage,
    reason: params.reason,
    servedPartial: params.servedPartial ?? false,
  };
}

function assemblyDecisionDetails(params: {
  checkpoint: CompileCheckpoint;
  brainDropReason: BrainDropReason;
  brainDropStage?: BrainDropStage;
  interruption?: BrainInterruptionMetadata | null;
  budgetFraction?: number | null;
  maxContextChars?: number;
  queryBudgetChars?: number | null;
  budgetedBrainContext?: BudgetedBrainContext;
  queryInterrupted?: boolean | null;
  interruptionStage?: InterruptionStage | null;
  interruptionReason?: string | null;
  servedPartial?: boolean | null;
}): AssemblyDecisionDetails {
  return {
    ...params.checkpoint,
    ...interruptionDecisionDetails(params.interruption),
    brainDropReason: params.brainDropReason,
    ...(params.brainDropStage ? { brainDropStage: params.brainDropStage } : {}),
    ...(params.queryInterrupted !== undefined ? { queryInterrupted: params.queryInterrupted } : {}),
    ...(params.interruptionStage !== undefined ? { interruptionStage: params.interruptionStage } : {}),
    ...(params.interruptionReason !== undefined ? { interruptionReason: params.interruptionReason } : {}),
    ...(params.servedPartial !== undefined ? { servedPartial: params.servedPartial } : {}),
    ...budgetDecisionDetails({
      budgetFraction: params.budgetFraction,
      maxContextChars: params.maxContextChars,
      queryBudgetChars: params.queryBudgetChars,
      budgetedBrainContext: params.budgetedBrainContext,
    }),
  };
}

function withCompileReport(
  trace: DecisionTrace | null | undefined,
  selectionMetadata: AssemblyDecisionDetails,
  mode: BrainAssemblyOutcomeMode | BrainAssemblyRouteMode,
): AssemblyDecisionDetails & { compileReport?: BrainCompileReportV1 | null; compileReportSummary?: string | null; prefetch?: BrainPrefetchDecision | null } {
  const inheritedTraceDetails = traceCarryoverDecisionDetails(trace);
  if (!trace?.routeTrace?.selectionMetadata) {
    return {
      ...inheritedTraceDetails,
      ...selectionMetadata,
    };
  }
  const compileReport = buildBrainCompileReport({
    routeTrace: {
      ...trace.routeTrace,
      selectionMetadata: {
        ...trace.routeTrace.selectionMetadata,
        ...selectionMetadata,
      },
    },
    decision: {
      mode,
      bindingMode: trace.routeTrace.selectionMetadata.compileReport?.bindingMode ?? null,
      traceId: trace.id,
      episodeId: trace.episodeId,
    },
  });
  if (!compileReport) {
    return {
      ...inheritedTraceDetails,
      ...selectionMetadata,
    };
  }
  return {
    ...inheritedTraceDetails,
    ...selectionMetadata,
    compileReport,
    compileReportSummary: compileReport.summary,
  };
}

function applyStructuredNodeBudget(params: {
  routeTrace: DecisionRouteTrace,
  maxContextChars?: number;
  buildFullContext: () => string;
  buildCompactContext: (fittedNodeSummaries: DecisionTraceInjectedNodeSummary[]) => string;
  dropReason: BrainFittingDropReason;
}): BudgetedBrainContext {
  const fullText = params.buildFullContext();
  if (typeof params.maxContextChars !== "number" || !Number.isFinite(params.maxContextChars)) {
    return {
      brainContext: fullText,
      injectedChars: fullText.length,
      droppedChars: 0,
      contextClipped: false,
    };
  }

  const limit = Math.max(0, Math.floor(params.maxContextChars));
  const retrievedNodeCount = params.routeTrace.injectedNodeSummaries.length;
  const baseDetails = {
    fitStrategy: "structured_node_budget" as const,
    retrievedNodeCount,
  };
  if (fullText.length <= limit) {
    return {
      brainContext: fullText,
      injectedChars: fullText.length,
      droppedChars: 0,
      contextClipped: false,
      ...baseDetails,
      fittedNodeCount: retrievedNodeCount,
      droppedNodeCount: 0,
      fittingDropReasons: null,
    };
  }

  if (limit === 0 || retrievedNodeCount === 0) {
    return {
      brainContext: "",
      injectedChars: 0,
      droppedChars: fullText.length,
      contextClipped: true,
      ...baseDetails,
      fittedNodeCount: 0,
      droppedNodeCount: retrievedNodeCount,
      fittingDropReasons: retrievedNodeCount > 0
        ? { [params.dropReason]: retrievedNodeCount }
        : null,
    };
  }

  const orderedNodes = orderedInjectedNodeSummaries(params.routeTrace);
  const fittedNodeSummaries: DecisionTraceInjectedNodeSummary[] = [];
  for (const summary of orderedNodes) {
    const candidateContext = params.buildCompactContext([...fittedNodeSummaries, summary]);
    if (candidateContext.length > limit) {
      break;
    }
    fittedNodeSummaries.push(summary);
  }

  const brainContext = params.buildCompactContext(fittedNodeSummaries);
  const fittedNodeCount = fittedNodeSummaries.length;
  const droppedNodeCount = Math.max(0, retrievedNodeCount - fittedNodeCount);
  return {
    brainContext,
    injectedChars: brainContext.length,
    droppedChars: Math.max(0, fullText.length - brainContext.length),
    contextClipped: brainContext.length < fullText.length,
    ...baseDetails,
    fittedNodeCount,
    droppedNodeCount,
    fittingDropReasons: droppedNodeCount > 0
      ? { [params.dropReason]: droppedNodeCount }
      : null,
  };
}

function applyStructuredMaxContextChars(
  result: TraversalResult,
  routeTrace: DecisionRouteTrace,
  maxContextChars?: number,
): BudgetedBrainContext {
  return applyStructuredNodeBudget({
    routeTrace,
    maxContextChars,
    buildFullContext: () => buildStructuredBrainContextBlock(result, routeTrace),
    buildCompactContext: (fittedNodeSummaries) => buildCompactStructuredBrainContext(fittedNodeSummaries),
    dropReason: "omitted_for_max_context_chars",
  });
}

function buildInterruptedBudgetedBrainContext(
  result: TraversalResult,
  stage: InterruptionStage,
  maxContextChars?: number,
): BudgetedBrainContext {
  const routeTrace = result.trace.routeTrace;
  if (!routeTrace || routeTrace.injectedNodeSummaries.length === 0) {
    return applyLegacyMaxContextChars(buildPartialBrainContextBlock(result, stage), maxContextChars);
  }
  return applyStructuredNodeBudget({
    routeTrace,
    maxContextChars,
    buildFullContext: () => buildStructuredPartialBrainContextBlock(result, routeTrace, stage),
    buildCompactContext: (fittedNodeSummaries) => buildCompactStructuredPartialBrainContext({
      routeTrace,
      stage,
      fittedNodeSummaries,
    }),
    dropReason: "omitted_for_partial_serve",
  });
}

function buildBudgetedBrainContext(
  result: TraversalResult,
  maxContextChars?: number,
): BudgetedBrainContext {
  const routeTrace = result.trace.routeTrace;
  if (routeTrace && routeTrace.injectedNodeSummaries.length > 0) {
    return applyStructuredMaxContextChars(result, routeTrace, maxContextChars);
  }
  return applyLegacyMaxContextChars(buildBrainContextBlock(result), maxContextChars);
}

function applyMaxContextChars(text: string, maxContextChars?: number): BudgetedBrainContext {
  return applyLegacyMaxContextChars(text, maxContextChars);
}

function mergeSystemPromptAddition(...parts: Array<string | undefined>): string | undefined {
  const merged = parts.filter(Boolean).join("\n\n");
  return merged || undefined;
}

export class BrainAssemblerExtension {
  constructor(private brain: BrainService) {}

  decide(params: {
    tokenBudget: number;
    liveMessages: AgentMessage[];
  }): BrainAssemblyDecision {
    const latestUserMessage = [...params.liveMessages]
      .reverse()
      .find((message) => message.role === "user");
    const queryText = latestUserMessage ? extractText(latestUserMessage.content) : "";

    if (!this.brain.isEnabled() || !this.brain.isInitialized()) {
      return { mode: "skip_uninitialized", queryText };
    }
    if (!this.brain.isEmbeddingConfigured()) {
      return { mode: "skip_no_embedding", queryText };
    }
    if (params.tokenBudget < 512) {
      return { mode: "skip_budget_too_small", queryText };
    }
    if (!queryText) {
      return { mode: "skip_no_query", queryText };
    }

    const normalized = queryText.toLowerCase();
    const looksStaticLookup =
      queryText.length < 72
      && (normalized.startsWith("read ")
        || normalized.startsWith("show ")
        || normalized.startsWith("open ")
        || normalized.startsWith("cat ")
        || normalized.startsWith("grep ")
        || normalized.includes(".ts")
        || normalized.includes(".md")
        || normalized.includes(".json")
        || normalized.includes("/"));
    if (looksStaticLookup) {
      return { mode: "skip_short_static_lookup", queryText };
    }

    return { mode: this.brain.isShadowMode() ? "shadow" : "use_brain", queryText };
  }

  async augmentAssembly(params: {
    conversationId: number;
    agentIdentity?: BrainAgentIdentity | null;
    tokenBudget: number;
    maxContextChars?: number;
    assembled: AssembleContextResult;
    liveMessages: AgentMessage[];
  }): Promise<BrainAssembledContextResult> {
    const compileStartedAt = Date.now();
    const compileDeadlineMs = resolveCompileDeadlineMs(this.brain.getCompileDeadlineMs());
    const decision = this.decide({
      tokenBudget: params.tokenBudget,
      liveMessages: params.liveMessages,
    });
    const summaryRouting = decideSummaryRouting({
      queryText: decision.queryText,
      summaryMetadata: params.assembled.summaryMetadata,
    });
    const summaryRoutingPrompt = buildSummaryRoutingPrompt(summaryRouting.mode, params.assembled.summaryMetadata);
    if (decision.mode !== "use_brain" && decision.mode !== "shadow") {
      const metadata = assemblyDecisionDetails({
        checkpoint: captureCompileCheckpoint(compileStartedAt, compileDeadlineMs),
        brainDropReason: decision.mode,
        brainDropStage: "decision",
        maxContextChars: params.maxContextChars,
        queryBudgetChars: 0,
      });
      this.brain.noteAssemblyDecision({
        mode: decision.mode,
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        footer: decisionFooter(decision.mode),
        ...metadata,
      });
      return {
        ...params.assembled,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode: decision.mode,
          footer: decisionFooter(decision.mode),
          ...metadata,
        },
      };
    }

    const budgetFraction = this.brain.getBudgetFraction();
    const queryBudgetChars = resolveBrainQueryBudgetChars(
      params.tokenBudget,
      budgetFraction,
    );
    const beforeQueryCheckpoint = captureCompileCheckpoint(compileStartedAt, compileDeadlineMs);
    if (beforeQueryCheckpoint.compileDeadlineHit) {
      const mode: BrainAssemblyOutcomeMode = "skip_deadline_before_query";
      const metadata = assemblyDecisionDetails({
        checkpoint: beforeQueryCheckpoint,
        brainDropReason: "deadline_before_query",
        brainDropStage: "decision",
        interruption: createInterruption({
          stage: "query",
          reason: "deadline_before_query",
        }),
        budgetFraction,
        maxContextChars: params.maxContextChars,
        queryBudgetChars,
        queryInterrupted: false,
        interruptionStage: "query",
        interruptionReason: "deadline_before_query",
        servedPartial: false,
      });
      metadata.prefetch = null;
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        footer: decisionFooter(mode),
        ...metadata,
      });
      return {
        ...params.assembled,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode,
          footer: decisionFooter(mode),
          ...metadata,
        },
      };
    }

    void this.brain.schedulePrefetch({
      conversationId: params.conversationId,
      queryText: decision.queryText,
      budgetChars: queryBudgetChars,
      deadlineAtMs: compileDeadlineMs === undefined ? undefined : compileStartedAt + compileDeadlineMs,
      summaryRoutingMode: summaryRouting.mode,
    });
    const result = await this.brain.query({
      conversationId: params.conversationId,
      agentIdentity: params.agentIdentity,
      queryText: decision.queryText,
      budgetChars: queryBudgetChars,
      summaryRoutingMode: summaryRouting.mode,
      ...(compileDeadlineMs === undefined ? {} : { deadlineAtMs: compileStartedAt + compileDeadlineMs }),
    });
    const prefetchDecision = this.brain.getLastPrefetchDecision();
    const queryInterruption = this.brain.getLastQueryInterruption();
    const afterQueryCheckpoint = captureCompileCheckpoint(compileStartedAt, compileDeadlineMs);
    if (!result && !queryInterruption) {
      const mode: BrainAssemblyOutcomeMode = "skip_query_returned_no_nodes";
      const metadata = assemblyDecisionDetails({
        checkpoint: afterQueryCheckpoint,
        brainDropReason: "query_returned_no_nodes",
        brainDropStage: "query",
        maxContextChars: params.maxContextChars,
        queryBudgetChars,
      });
      metadata.prefetch = prefetchDecision;
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        footer: decisionFooter(mode),
        ...metadata,
      });
      return {
        ...params.assembled,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode,
          footer: decisionFooter(mode),
          ...metadata,
        },
      };
    }
    const afterQueryInterruption = queryInterruption;
    if (!result) {
      if (afterQueryInterruption) {
        const mode: BrainAssemblyOutcomeMode = "skip_deadline_after_query";
        const traceSelectionMetadata = assemblyDecisionDetails({
          checkpoint: afterQueryCheckpoint,
          brainDropReason: "deadline_after_query",
          brainDropStage: "query",
          interruption: afterQueryInterruption,
          budgetFraction,
          maxContextChars: params.maxContextChars,
          queryBudgetChars,
        });
        traceSelectionMetadata.prefetch = prefetchDecision;
        this.brain.noteAssemblyDecision({
          mode,
          conversationId: params.conversationId,
          agentIdentity: params.agentIdentity,
          episodeId: null,
          traceId: null,
          footer: decisionFooter(mode),
          ...traceSelectionMetadata,
        });
        return {
          ...params.assembled,
          systemPromptAddition: mergeSystemPromptAddition(
            params.assembled.systemPromptAddition,
            summaryRoutingPrompt,
          ),
          brainDecision: {
            mode,
            episodeId: null,
            traceId: null,
            footer: decisionFooter(mode),
            ...traceSelectionMetadata,
          },
        };
      }
      const mode: BrainAssemblyOutcomeMode = "skip_query_returned_no_nodes";
      const metadata = assemblyDecisionDetails({
        checkpoint: afterQueryCheckpoint,
        brainDropReason: "query_returned_no_nodes",
        brainDropStage: "query",
        budgetFraction,
        maxContextChars: params.maxContextChars,
        queryBudgetChars,
      });
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        footer: decisionFooter(mode),
        ...metadata,
      });
      return {
        ...params.assembled,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode,
          footer: decisionFooter(mode),
          ...metadata,
        },
      };
    }
    if (afterQueryCheckpoint.compileDeadlineHit && decision.mode === "use_brain" && !afterQueryInterruption) {
      const interruptedMaxContextChars = resolveInterruptedMaxContextChars(params.maxContextChars);
      const interruptedBrainContext = buildInterruptedBudgetedBrainContext(
        result,
        "query",
        interruptedMaxContextChars,
      );
      const mode: BrainAssemblyOutcomeMode = "partial_deadline_after_query";
      const traceSelectionMetadata = withCompileReport(result.trace, assemblyDecisionDetails({
        checkpoint: afterQueryCheckpoint,
        brainDropReason: "deadline_after_query",
        brainDropStage: "query",
        interruption: createInterruption({
          stage: "query",
          reason: INTERRUPTION_REASON_SOFT_COMPILE_DEADLINE,
          servedPartial: true,
        }),
        budgetFraction,
        maxContextChars: interruptedMaxContextChars,
        queryBudgetChars,
        budgetedBrainContext: interruptedBrainContext,
        queryInterrupted: false,
      }), mode);
      this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
      const brainMessage: AgentMessage | null = interruptedBrainContext.brainContext.length > 0
        ? ({
            role: "user",
            content: interruptedBrainContext.brainContext,
          } as AgentMessage)
        : null;
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        episodeId: result.episode.id,
        traceId: result.trace.id,
        footer: decisionFooter(mode),
        ...traceSelectionMetadata,
      });
      return {
        ...params.assembled,
        messages: brainMessage ? [brainMessage, ...params.assembled.messages] : params.assembled.messages,
        estimatedTokens: brainMessage
          ? params.assembled.estimatedTokens + estimateTokens(extractText(brainMessage.content))
          : params.assembled.estimatedTokens,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          brainMessage
            ? "OpenClawBrain partial serves are committed prefixes; omitted context was not compiled in time."
            : undefined,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode,
          episodeId: result.episode.id,
          traceId: result.trace.id,
          footer: decisionFooter(mode),
          ...traceSelectionMetadata,
        },
      };
    }
    if (afterQueryInterruption) {
      if (decision.mode === "use_brain" && afterQueryInterruption.servedPartial) {
        const interruptedMaxContextChars = resolveInterruptedMaxContextChars(params.maxContextChars);
        const interruptedBrainContext = buildInterruptedBudgetedBrainContext(
          result,
          afterQueryInterruption.stage,
          interruptedMaxContextChars,
        );
        const mode: BrainAssemblyOutcomeMode = "partial_query_interruption";
        const traceSelectionMetadata = withCompileReport(result.trace, assemblyDecisionDetails({
          checkpoint: afterQueryCheckpoint,
          brainDropReason: "deadline_after_query",
          brainDropStage: "query",
          interruption: afterQueryInterruption,
          budgetFraction,
          maxContextChars: interruptedMaxContextChars,
          queryBudgetChars,
          budgetedBrainContext: interruptedBrainContext,
        }), mode);
        traceSelectionMetadata.prefetch = prefetchDecision;
        this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
        const brainMessage: AgentMessage | null = interruptedBrainContext.brainContext.length > 0
          ? ({
              role: "user",
              content: interruptedBrainContext.brainContext,
            } as AgentMessage)
          : null;
        this.brain.noteAssemblyDecision({
          mode,
          conversationId: params.conversationId,
          episodeId: result.episode.id,
          traceId: result.trace.id,
          footer: decisionFooter(mode),
          ...traceSelectionMetadata,
        });
        return {
          ...params.assembled,
          messages: brainMessage ? [brainMessage, ...params.assembled.messages] : params.assembled.messages,
          estimatedTokens: brainMessage
            ? params.assembled.estimatedTokens + estimateTokens(extractText(brainMessage.content))
            : params.assembled.estimatedTokens,
          systemPromptAddition: mergeSystemPromptAddition(
            params.assembled.systemPromptAddition,
            brainMessage
              ? "OpenClawBrain partial serves are committed prefixes; omitted context was cut by query-time deadline pressure."
              : undefined,
            summaryRoutingPrompt,
          ),
          brainDecision: {
            mode,
            episodeId: result.episode.id,
            traceId: result.trace.id,
            footer: decisionFooter(mode),
            ...traceSelectionMetadata,
          },
        };
      }
      const mode: BrainAssemblyOutcomeMode = "skip_deadline_after_query";
      const traceSelectionMetadata = withCompileReport(result.trace, assemblyDecisionDetails({
        checkpoint: afterQueryCheckpoint,
        brainDropReason: "deadline_after_query",
        brainDropStage: "query",
        interruption: afterQueryInterruption,
        budgetFraction,
        maxContextChars: params.maxContextChars,
        queryBudgetChars,
      }), mode);
      this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        episodeId: result.episode.id,
        traceId: result.trace.id,
        footer: decisionFooter(mode),
        ...traceSelectionMetadata,
      });
      return {
        ...params.assembled,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode,
          episodeId: result.episode.id,
          traceId: result.trace.id,
          footer: decisionFooter(mode),
          ...traceSelectionMetadata,
        },
      };
    }

    const budgetedBrainContext = buildBudgetedBrainContext(
      result,
      params.maxContextChars,
    );
    const beforeInjectionCheckpoint = captureCompileCheckpoint(compileStartedAt, compileDeadlineMs);
    if (beforeInjectionCheckpoint.compileDeadlineHit) {
      if (decision.mode === "use_brain") {
        const interruptedMaxContextChars = resolveInterruptedMaxContextChars(params.maxContextChars);
        const interruptedBrainContext = buildInterruptedBudgetedBrainContext(
          result,
          "injection",
          interruptedMaxContextChars,
        );
        const mode: BrainAssemblyOutcomeMode = "partial_deadline_before_injection";
        const traceSelectionMetadata = withCompileReport(result.trace, assemblyDecisionDetails({
          checkpoint: beforeInjectionCheckpoint,
          brainDropReason: "deadline_before_injection",
          brainDropStage: "injection",
          interruption: createInterruption({
            stage: "injection",
            reason: INTERRUPTION_REASON_SOFT_COMPILE_DEADLINE,
            servedPartial: true,
          }),
          budgetFraction,
          maxContextChars: interruptedMaxContextChars,
          queryBudgetChars,
          budgetedBrainContext: interruptedBrainContext,
          queryInterrupted: false,
        }), mode);
        traceSelectionMetadata.prefetch = prefetchDecision;
        this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
        const brainMessage: AgentMessage | null = interruptedBrainContext.brainContext.length > 0
          ? ({
              role: "user",
              content: interruptedBrainContext.brainContext,
            } as AgentMessage)
          : null;
        this.brain.noteAssemblyDecision({
          mode,
          conversationId: params.conversationId,
          agentIdentity: params.agentIdentity,
          episodeId: result.episode.id,
          traceId: result.trace.id,
          footer: decisionFooter(mode),
          ...traceSelectionMetadata,
        });
        return {
          ...params.assembled,
          messages: brainMessage ? [brainMessage, ...params.assembled.messages] : params.assembled.messages,
          estimatedTokens: brainMessage
            ? params.assembled.estimatedTokens + estimateTokens(extractText(brainMessage.content))
            : params.assembled.estimatedTokens,
          systemPromptAddition: mergeSystemPromptAddition(
            params.assembled.systemPromptAddition,
            brainMessage
              ? "OpenClawBrain partial serves are committed prefixes; omitted context was not compiled in time."
              : undefined,
            summaryRoutingPrompt,
          ),
          brainDecision: {
            mode,
            episodeId: result.episode.id,
            traceId: result.trace.id,
            footer: decisionFooter(mode),
            ...traceSelectionMetadata,
          },
        };
      }

      const mode: BrainAssemblyOutcomeMode = "skip_deadline_before_injection";
      const traceSelectionMetadata = withCompileReport(result.trace, assemblyDecisionDetails({
        checkpoint: beforeInjectionCheckpoint,
        brainDropReason: "deadline_before_injection",
        brainDropStage: "injection",
        interruption: createInterruption({
          stage: "injection",
          reason: INTERRUPTION_REASON_SOFT_COMPILE_DEADLINE,
        }),
        budgetFraction,
        maxContextChars: params.maxContextChars,
        queryBudgetChars,
        budgetedBrainContext,
        queryInterrupted: false,
      }), mode);
      traceSelectionMetadata.prefetch = prefetchDecision;
      this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        episodeId: result.episode.id,
        traceId: result.trace.id,
        footer: decisionFooter(mode),
        ...traceSelectionMetadata,
      });
      return {
        ...params.assembled,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode,
          episodeId: result.episode.id,
          traceId: result.trace.id,
          footer: decisionFooter(mode),
          ...traceSelectionMetadata,
        },
      };
    }

    const traceSelectionMetadata = withCompileReport(result.trace, assemblyDecisionDetails({
      checkpoint: beforeInjectionCheckpoint,
      brainDropReason: decision.mode === "shadow"
        ? "shadow_mode"
        : budgetedBrainContext.contextClipped
          ? "injection_cap_clipped"
          : "none",
      brainDropStage: decision.mode === "shadow" || budgetedBrainContext.contextClipped ? "injection" : undefined,
      budgetFraction,
      maxContextChars: params.maxContextChars,
      queryBudgetChars,
      budgetedBrainContext,
      servedPartial:
        decision.mode !== "shadow"
        && budgetedBrainContext.contextClipped
        && budgetedBrainContext.injectedChars > 0,
    }), decision.mode);
    traceSelectionMetadata.prefetch = prefetchDecision;
    this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
    const brainMessage: AgentMessage | null = budgetedBrainContext.brainContext.length > 0
      ? ({
          role: "user",
          content: budgetedBrainContext.brainContext,
        } as AgentMessage)
      : null;
    if (decision.mode === "shadow") {
      this.brain.noteAssemblyDecision({
        mode: "shadow",
        conversationId: params.conversationId,
        agentIdentity: params.agentIdentity,
        episodeId: result.episode.id,
        traceId: result.trace.id,
        footer: decisionFooter("shadow"),
        ...traceSelectionMetadata,
      });

      return {
        ...params.assembled,
        systemPromptAddition: mergeSystemPromptAddition(
          params.assembled.systemPromptAddition,
          summaryRoutingPrompt,
        ),
        brainDecision: {
          mode: "shadow",
          episodeId: result.episode.id,
          traceId: result.trace.id,
          footer: decisionFooter("shadow"),
          ...traceSelectionMetadata,
        },
      };
    }
    this.brain.noteAssemblyDecision({
      mode: "use_brain",
      conversationId: params.conversationId,
      agentIdentity: params.agentIdentity,
      episodeId: result.episode.id,
      traceId: result.trace.id,
      footer: result.trace.footer,
      ...traceSelectionMetadata,
    });

    return {
      ...params.assembled,
      messages: brainMessage ? [brainMessage, ...params.assembled.messages] : params.assembled.messages,
      estimatedTokens: brainMessage
        ? params.assembled.estimatedTokens + estimateTokens(extractText(brainMessage.content))
        : params.assembled.estimatedTokens,
      systemPromptAddition: mergeSystemPromptAddition(
        params.assembled.systemPromptAddition,
        brainMessage
          ? "OpenClawBrain sections are ranked by trust: correction cards, evidence, playbooks, then transcript support."
          : undefined,
        summaryRoutingPrompt,
      ),
      brainDecision: {
        mode: "use_brain",
        episodeId: result.episode.id,
        traceId: result.trace.id,
        footer: result.trace.footer,
        ...traceSelectionMetadata,
      },
    };
  }
}
