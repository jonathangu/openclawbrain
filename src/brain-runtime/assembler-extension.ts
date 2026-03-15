import type { AssembleContextResult } from "../assembler.js";
import type { ContextEngine } from "openclaw/plugin-sdk";
import type { TraversalResult } from "../brain-core/types.js";
import type { BrainService } from "./service.js";

type AgentMessage = Parameters<ContextEngine["ingest"]>[0]["message"];

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

  async augmentAssembly(params: {
    conversationId: number;
    tokenBudget: number;
    assembled: AssembleContextResult;
    liveMessages: AgentMessage[];
  }): Promise<AssembleContextResult> {
    if (!this.brain.isEnabled() || !this.brain.isInitialized()) {
      return params.assembled;
    }

    const latestUserMessage = [...params.liveMessages]
      .reverse()
      .find((message) => message.role === "user");
    const queryText = latestUserMessage ? extractText(latestUserMessage.content) : "";
    if (!queryText) {
      return params.assembled;
    }

    const result = await this.brain.query({
      conversationId: params.conversationId,
      queryText,
      budgetChars: Math.max(256, Math.floor(params.tokenBudget * 4 * 0.3)),
    });
    if (!result) {
      return params.assembled;
    }

    const brainMessage: AgentMessage = {
      role: "user",
      content: buildBrainContextBlock(result),
    } as AgentMessage;

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
    };
  }
}
