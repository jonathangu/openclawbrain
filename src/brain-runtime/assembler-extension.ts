import type { AssembleContextResult } from "../assembler.js";
import type { ContextEngine, AgentMessage } from "../openclaw-sdk-compat.js";
import type {
  BrainDropReason,
  BrainDropStage,
  BrainFitStrategy,
  BrainFittingDropReason,
  BrainInterruptionMetadata,
  BrainInterruptionStage,
  DecisionRouteTrace,
  DecisionTraceInjectedNodeSummary,
  TraversalResult,
} from "../brain-core/types.js";
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
  | "partial_deadline_after_query"
  | "partial_deadline_before_injection";

export type BrainAssemblyDecision = {
  mode: BrainAssemblyRouteMode;
  queryText: string;
};

export type BrainAssembledContextResult = AssembleContextResult;

const COMPACT_INJECTED_PREVIEW_CHARS = 96;
const INTERRUPTED_SERVE_MAX_CONTEXT_CHARS = 1024;

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

type CompileCheckpoint = {
  compileElapsedMs: number;
  compileDeadlineMs?: number | null;
  compileDeadlineHit?: boolean | null;
};

type AssemblyDecisionDetails = BudgetDecisionDetails & CompileDecisionDetails & InterruptionDecisionDetails;

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

function compactPreview(content: string): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "(preview unavailable)";
  }
  if (normalized.length <= COMPACT_INJECTED_PREVIEW_CHARS) {
    return normalized;
  }
  return `${normalized.slice(0, COMPACT_INJECTED_PREVIEW_CHARS - 1)}…`;
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

function buildSummaryRoutingPrompt(mode: ReturnType<typeof decideSummaryRouting>["mode"]): string | undefined {
  switch (mode) {
    case "summary_suffices":
      return "This turn looks like a broad recap. Summary-level context is a reasonable starting point unless the user asks for exact proof or current-truth conflict resolution.";
    case "prefer_typed_memory":
      return "This turn looks current-truth or conflict-sensitive. Prefer explicit correction cards and typed memory over summary recap; if typed memory is missing, expand toward source before asserting specifics.";
    case "expand_to_source":
      return "This turn looks precision-sensitive against compacted history. Use summaries only to locate the region, then expand toward source material before asserting exact details.";
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

function applyStructuredMaxContextChars(
  result: TraversalResult,
  routeTrace: DecisionRouteTrace,
  maxContextChars?: number,
): BudgetedBrainContext {
  const fullText = buildStructuredBrainContextBlock(result, routeTrace);
  if (typeof maxContextChars !== "number" || !Number.isFinite(maxContextChars)) {
    return {
      brainContext: fullText,
      injectedChars: fullText.length,
      droppedChars: 0,
      contextClipped: false,
    };
  }

  const limit = Math.max(0, Math.floor(maxContextChars));
  const retrievedNodeCount = routeTrace.injectedNodeSummaries.length;
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
        ? { omitted_for_max_context_chars: retrievedNodeCount }
        : null,
    };
  }

  const orderedNodes = orderedInjectedNodeSummaries(routeTrace);
  const fittedNodeSummaries: DecisionTraceInjectedNodeSummary[] = [];
  for (const summary of orderedNodes) {
    const candidateContext = buildCompactStructuredBrainContext([...fittedNodeSummaries, summary]);
    if (candidateContext.length > limit) {
      break;
    }
    fittedNodeSummaries.push(summary);
  }

  const brainContext = buildCompactStructuredBrainContext(fittedNodeSummaries);
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
      ? { omitted_for_max_context_chars: droppedNodeCount }
      : null,
  };
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
    const summaryRoutingPrompt = buildSummaryRoutingPrompt(summaryRouting.mode);
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
      });
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
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

    const result = await this.brain.query({
      conversationId: params.conversationId,
      queryText: decision.queryText,
      budgetChars: queryBudgetChars,
      ...(compileDeadlineMs === undefined ? {} : { deadlineAtMs: compileStartedAt + compileDeadlineMs }),
    });
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
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
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
    const afterQueryInterruption = queryInterruption ?? (
      afterQueryCheckpoint.compileDeadlineHit
        ? createInterruption({
            stage: "query",
            reason: "deadline_after_query",
          })
        : null
    );
    if (afterQueryInterruption) {
      const mode: BrainAssemblyOutcomeMode = "skip_deadline_after_query";
      const traceSelectionMetadata = assemblyDecisionDetails({
        checkpoint: afterQueryCheckpoint,
        brainDropReason: "deadline_after_query",
        brainDropStage: "query",
        interruption: afterQueryInterruption,
        maxContextChars: params.maxContextChars,
        queryBudgetChars,
      });
      if (result?.trace) {
        this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
      }
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        episodeId: result?.episode.id ?? null,
        traceId: result?.trace.id ?? null,
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
          episodeId: result?.episode.id ?? null,
          traceId: result?.trace.id ?? null,
          footer: decisionFooter(mode),
          ...traceSelectionMetadata,
        },
      };
    }
    if (!result) {
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
    if (afterQueryInterruption) {
      if (decision.mode === "use_brain" && result) {
        const interruptedMaxContextChars = resolveInterruptedMaxContextChars(params.maxContextChars);
        const interruptedBrainContext = applyMaxContextChars(
          buildPartialBrainContextBlock(result, afterQueryInterruption.stage === "injection" ? "injection" : "query"),
          interruptedMaxContextChars,
        );
        const interruption = {
          ...afterQueryInterruption,
          servedPartial: true,
        };
        const mode: BrainAssemblyOutcomeMode = "partial_deadline_after_query";
        const traceSelectionMetadata = assemblyDecisionDetails({
          checkpoint: afterQueryCheckpoint,
          brainDropReason: "deadline_after_query",
          brainDropStage: "query",
          interruption,
          maxContextChars: interruptedMaxContextChars,
          queryBudgetChars,
          budgetedBrainContext: interruptedBrainContext,
        });
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
      if (result?.trace) {
        this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
      }
      this.brain.noteAssemblyDecision({
        mode,
        conversationId: params.conversationId,
        episodeId: result?.episode.id ?? null,
        traceId: result?.trace.id ?? null,
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
          episodeId: result?.episode.id ?? null,
          traceId: result?.trace.id ?? null,
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
        const interruptedBrainContext = applyMaxContextChars(
          buildPartialBrainContextBlock(result, "injection"),
          interruptedMaxContextChars,
        );
        const mode: BrainAssemblyOutcomeMode = "partial_deadline_before_injection";
        const traceSelectionMetadata = assemblyDecisionDetails({
          checkpoint: beforeInjectionCheckpoint,
          brainDropReason: "deadline_before_injection",
          brainDropStage: "injection",
          interruption: createInterruption({
            stage: "injection",
            reason: "deadline_before_injection",
            servedPartial: true,
          }),
          maxContextChars: interruptedMaxContextChars,
          queryBudgetChars,
          budgetedBrainContext: interruptedBrainContext,
        });
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
      const traceSelectionMetadata = assemblyDecisionDetails({
        checkpoint: beforeInjectionCheckpoint,
        brainDropReason: "deadline_before_injection",
        brainDropStage: "injection",
        interruption: createInterruption({
          stage: "injection",
          reason: "deadline_before_injection",
        }),
        budgetFraction,
        maxContextChars: params.maxContextChars,
        queryBudgetChars,
        budgetedBrainContext,
      });
      this.brain.recordTraceSelectionMetadata(result.trace, traceSelectionMetadata);
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

    const traceSelectionMetadata = assemblyDecisionDetails({
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
    });
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
