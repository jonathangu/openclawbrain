import type { AssembleContextResult } from "../assembler.js";
import type { ContextEngine } from "openclaw/plugin-sdk";
import type { TraversalResult } from "../brain-core/types.js";
import type { BrainService } from "./service.js";

type AgentMessage = Parameters<ContextEngine["ingest"]>[0]["message"];
export type BrainAssemblyDecisionMode =
  | "use_brain"
  | "skip_short_static_lookup"
  | "skip_no_embedding"
  | "skip_uninitialized"
  | "skip_budget_too_small";

export type BrainAssemblyDecision = {
  mode: BrainAssemblyDecisionMode;
  queryText: string;
};

export type BrainAssembledContextResult = AssembleContextResult & {
  brainDecision?: {
    mode: BrainAssemblyDecisionMode;
    episodeId?: string | null;
    traceId?: string | null;
    footer?: string | null;
  };
};

function decisionFooter(mode: BrainAssemblyDecisionMode): string {
  switch (mode) {
    case "use_brain":
      return "[brain] used graph retrieval for this turn.";
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

function buildBrainContextBlock(result: TraversalResult): string {
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

    const normalized = queryText.toLowerCase();
    const looksStaticLookup =
      queryText.length < 48
      && (normalized.startsWith("read ")
        || normalized.startsWith("show ")
        || normalized.startsWith("open ")
        || normalized.includes(".ts")
        || normalized.includes(".md")
        || normalized.includes("/"));
    if (!queryText || looksStaticLookup) {
      return { mode: "skip_short_static_lookup", queryText };
    }

    return { mode: "use_brain", queryText };
  }

  async augmentAssembly(params: {
    conversationId: number;
    tokenBudget: number;
    assembled: AssembleContextResult;
    liveMessages: AgentMessage[];
  }): Promise<BrainAssembledContextResult> {
    const decision = this.decide({
      tokenBudget: params.tokenBudget,
      liveMessages: params.liveMessages,
    });
    if (decision.mode !== "use_brain") {
      this.brain.noteAssemblyDecision({
        mode: decision.mode,
        conversationId: params.conversationId,
        footer: decisionFooter(decision.mode),
      });
      return {
        ...params.assembled,
        brainDecision: {
          mode: decision.mode,
          footer: decisionFooter(decision.mode),
        },
      };
    }

    const result = await this.brain.query({
      conversationId: params.conversationId,
      queryText: decision.queryText,
      budgetChars: Math.max(256, Math.floor(params.tokenBudget * 4 * 0.3)),
    });
    if (!result) {
      this.brain.noteAssemblyDecision({
        mode: "use_brain",
        conversationId: params.conversationId,
        footer: decisionFooter("use_brain"),
      });
      return {
        ...params.assembled,
        brainDecision: {
          mode: "use_brain",
          footer: decisionFooter("use_brain"),
        },
      };
    }

    const brainMessage: AgentMessage = {
      role: "user",
      content: buildBrainContextBlock(result),
    } as AgentMessage;
    this.brain.noteAssemblyDecision({
      mode: "use_brain",
      conversationId: params.conversationId,
      episodeId: result.episode.id,
      traceId: result.trace.id,
      footer: result.trace.footer,
    });

    return {
      ...params.assembled,
      messages: [brainMessage, ...params.assembled.messages],
      estimatedTokens: params.assembled.estimatedTokens + estimateTokens(extractText(brainMessage.content)),
      systemPromptAddition: [
        params.assembled.systemPromptAddition,
        "OpenClawBrain sections are ranked by trust: correction cards, evidence, playbooks, then transcript support.",
      ]
        .filter(Boolean)
        .join("\n\n"),
      brainDecision: {
        mode: "use_brain",
        episodeId: result.episode.id,
        traceId: result.trace.id,
        footer: result.trace.footer,
      },
    };
  }
}
