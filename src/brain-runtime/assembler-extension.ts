import type { AssembleContextResult } from "../assembler.js";
import type { ContextEngine, AgentMessage } from "../openclaw-sdk-compat.js";
import type {
  DecisionRouteTrace,
  DecisionTraceInjectedNodeSummary,
  TraversalResult,
} from "../brain-core/types.js";
import type { BrainService } from "./service.js";
import { decideSummaryRouting } from "./summary-routing-policy.js";

export type BrainAssemblyDecisionMode =
  | "use_brain"
  | "shadow"
  | "skip_no_query"
  | "skip_short_static_lookup"
  | "skip_no_embedding"
  | "skip_uninitialized"
  | "skip_budget_too_small";

export type BrainAssemblyDecision = {
  mode: BrainAssemblyDecisionMode;
  queryText: string;
};

export type BrainAssembledContextResult = AssembleContextResult;

const COMPACT_INJECTED_PREVIEW_CHARS = 96;

type BudgetedBrainContext = {
  brainContext: string;
  injectedChars: number;
  droppedChars: number;
  contextClipped: boolean;
};

type BudgetDecisionDetails = {
  maxContextChars?: number | null;
  queryBudgetChars?: number | null;
  injectedChars?: number | null;
  droppedChars?: number | null;
  contextClipped?: boolean;
};

function decisionFooter(mode: BrainAssemblyDecisionMode): string {
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

function buildBrainContextBlock(result: TraversalResult): string {
  const routeTrace = result.trace.routeTrace;
  if (!routeTrace || routeTrace.injectedNodeSummaries.length === 0) {
    return buildLegacyBrainContextBlock(result);
  }
  return buildStructuredBrainContextBlock(result, routeTrace);
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

function resolveBrainQueryBudgetChars(tokenBudget: number, maxContextChars?: number): number {
  const derivedBudgetChars = Math.max(256, Math.floor(tokenBudget * 4 * 0.3));
  if (typeof maxContextChars !== "number" || !Number.isFinite(maxContextChars)) {
    return derivedBudgetChars;
  }
  return Math.max(0, Math.min(derivedBudgetChars, Math.floor(maxContextChars)));
}

function applyMaxContextChars(text: string, maxContextChars?: number): BudgetedBrainContext {
  if (typeof maxContextChars !== "number" || !Number.isFinite(maxContextChars)) {
    return {
      brainContext: text,
      injectedChars: text.length,
      droppedChars: 0,
      contextClipped: false,
    };
  }

  const limit = Math.max(0, Math.floor(maxContextChars));
  if (text.length <= limit) {
    return {
      brainContext: text,
      injectedChars: text.length,
      droppedChars: 0,
      contextClipped: false,
    };
  }

  if (limit === 0) {
    return {
      brainContext: "",
      injectedChars: 0,
      droppedChars: text.length,
      contextClipped: true,
    };
  }

  const hardSlice = text.slice(0, limit);
  const lineBoundary = hardSlice.lastIndexOf("\n");
  const clipped = (
    lineBoundary >= Math.floor(limit * 0.6)
      ? hardSlice.slice(0, lineBoundary)
      : hardSlice
  ).trimEnd();
  const brainContext = clipped.length > 0 ? clipped : hardSlice.trimEnd();

  return {
    brainContext,
    injectedChars: brainContext.length,
    droppedChars: Math.max(0, text.length - brainContext.length),
    contextClipped: true,
  };
}

function budgetDecisionDetails(params: {
  maxContextChars?: number;
  queryBudgetChars: number;
  budgetedBrainContext?: BudgetedBrainContext;
}): BudgetDecisionDetails {
  if (params.maxContextChars === undefined) {
    return {};
  }

  return {
    maxContextChars: params.maxContextChars,
    queryBudgetChars: params.queryBudgetChars,
    injectedChars: params.budgetedBrainContext?.injectedChars ?? 0,
    droppedChars: params.budgetedBrainContext?.droppedChars ?? 0,
    contextClipped: params.budgetedBrainContext?.contextClipped ?? false,
  };
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
      this.brain.noteAssemblyDecision({
        mode: decision.mode,
        conversationId: params.conversationId,
        footer: decisionFooter(decision.mode),
        ...budgetDecisionDetails({
          maxContextChars: params.maxContextChars,
          queryBudgetChars: 0,
        }),
      });
      return {
        ...params.assembled,
        systemPromptAddition: [params.assembled.systemPromptAddition, summaryRoutingPrompt].filter(Boolean).join("\n\n") || undefined,
        brainDecision: {
          mode: decision.mode,
          footer: decisionFooter(decision.mode),
        },
      };
    }

    const queryBudgetChars = resolveBrainQueryBudgetChars(params.tokenBudget, params.maxContextChars);
    const result = await this.brain.query({
      conversationId: params.conversationId,
      queryText: decision.queryText,
      budgetChars: queryBudgetChars,
    });
    if (!result) {
      this.brain.noteAssemblyDecision({
        mode: decision.mode,
        conversationId: params.conversationId,
        footer: decisionFooter(decision.mode),
        ...budgetDecisionDetails({
          maxContextChars: params.maxContextChars,
          queryBudgetChars,
        }),
      });
      return {
        ...params.assembled,
        systemPromptAddition: [params.assembled.systemPromptAddition, summaryRoutingPrompt].filter(Boolean).join("\n\n") || undefined,
        brainDecision: {
          mode: decision.mode,
          footer: decisionFooter(decision.mode),
          ...budgetDecisionDetails({
            maxContextChars: params.maxContextChars,
            queryBudgetChars,
          }),
        },
      };
    }

    const budgetedBrainContext = applyMaxContextChars(
      buildBrainContextBlock(result),
      params.maxContextChars,
    );
    const traceSelectionMetadata = budgetDecisionDetails({
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
        systemPromptAddition: [params.assembled.systemPromptAddition, summaryRoutingPrompt].filter(Boolean).join("\n\n") || undefined,
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
      systemPromptAddition: [
        params.assembled.systemPromptAddition,
        brainMessage
          ? "OpenClawBrain sections are ranked by trust: correction cards, evidence, playbooks, then transcript support."
          : undefined,
        summaryRoutingPrompt,
      ]
        .filter(Boolean)
        .join("\n\n"),
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
