import type { CompleteFn } from "../types.js";
import type { MessageRole } from "../store/conversation-store.js";
import type { SummaryKind } from "../store/summary-store.js";

export type UserMemoryObservation = {
  conversationId: number;
  messageId: number;
  episodeId?: string;
  userText: string;
  recentMessages: { role: MessageRole; content: string }[];
  recentSummaries: {
    summaryId: string;
    kind: SummaryKind;
    depth: number;
    content: string;
  }[];
};

export type UserMemoryProposal =
  | {
      kind: "noop";
      confidence: number;
      reason: string;
    }
  | {
      kind: "explicit_correction";
      canonicalInstruction: string;
      confidence: number;
      reason: string;
    };

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function trimPunctuation(text: string): string {
  return normalizeWhitespace(text).replace(/[\s.?!,:;]+$/g, "").trim();
}

function sentenceCase(text: string): string {
  if (!text) {
    return text;
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function toCanonicalFactInstruction(subject: string, value: string): string {
  const normalizedSubject = trimPunctuation(subject)
    .replace(/^(the|a|an)\s+/i, "")
    .trim();
  const normalizedValue = trimPunctuation(value);
  if (!normalizedSubject || !normalizedValue) {
    return "";
  }
  return `The ${normalizedSubject} is ${normalizedValue}.`;
}

function toCanonicalPreferenceInstruction(preferred: string, rejected?: string): string {
  const normalizedPreferred = trimPunctuation(preferred);
  const normalizedRejected = rejected ? trimPunctuation(rejected) : "";
  if (!normalizedPreferred) {
    return "";
  }
  return normalizedRejected
    ? `Use ${normalizedPreferred}, not ${normalizedRejected}.`
    : `Use ${normalizedPreferred}.`;
}

function inferSubjectFromText(text: string): string | null {
  const patterns = [
    /\b(?:what(?:'s|\s+is)|tell me|remember)\s+the\s+([a-z0-9][a-z0-9 _\/-]{1,60})\b/i,
    /\bthe\s+([a-z0-9][a-z0-9 _\/-]{1,60})\s+is\s+[^\n.?!]+/i,
    /\bthe\s+([a-z0-9][a-z0-9 _\/-]{1,60})\s+(?:changed to|should be|is now)\s+[^\n.?!]+/i,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    const candidate = trimPunctuation(match?.[1] ?? "");
    if (candidate) {
      return candidate;
    }
  }
  return null;
}

function inferSubjectFromContext(observation: UserMemoryObservation): string | null {
  const sources = [
    ...observation.recentMessages.map((message) => message.content),
    ...observation.recentSummaries.map((summary) => summary.content),
  ];
  for (let i = sources.length - 1; i >= 0; i -= 1) {
    const subject = inferSubjectFromText(sources[i] ?? "");
    if (subject) {
      return subject;
    }
  }
  return null;
}

function hasCorrectionCue(text: string): boolean {
  return [
    /^(?:no|wrong|actually|correction|instead)\b/i,
    /\bnot anymore\b/i,
    /\bit changed to\b/i,
    /\bis now\b/i,
    /\bshould be\b/i,
    /\buse\s+.+?,\s*not\s+.+/i,
  ].some((pattern) => pattern.test(text));
}

function buildFastProposal(observation: UserMemoryObservation): UserMemoryProposal | null {
  const rawText = normalizeWhitespace(observation.userText);
  if (!rawText || !hasCorrectionCue(rawText)) {
    return null;
  }

  const useMatch = rawText.match(/\buse\s+(.+?)\s*,\s*not\s+(.+?)(?:[.?!]|$)/i);
  if (useMatch) {
    const canonicalInstruction = toCanonicalPreferenceInstruction(useMatch[1] ?? "", useMatch[2] ?? "");
    if (canonicalInstruction) {
      return {
        kind: "explicit_correction",
        canonicalInstruction,
        confidence: 0.97,
        reason: "detected explicit use-X-not-Y correction",
      };
    }
  }

  const explicitSubjectMatch = rawText.match(
    /(?:^|\b(?:no|wrong|actually|correction|instead)[,:]?\s+)(?:the\s+)?([a-z0-9][a-z0-9 _\/-]{1,60})\s+(?:is|is now|should be|changed to)\s+(.+?)(?:[.?!]|$)/i,
  );
  if (explicitSubjectMatch) {
    const canonicalInstruction = toCanonicalFactInstruction(
      explicitSubjectMatch[1] ?? "",
      explicitSubjectMatch[2] ?? "",
    );
    if (canonicalInstruction) {
      return {
        kind: "explicit_correction",
        canonicalInstruction,
        confidence: 0.95,
        reason: "detected explicit subject/value correction",
      };
    }
  }

  const pronounMatch = rawText.match(/\b(?:it|that)\s+(?:changed to|is now|should be)\s+(.+?)(?:[.?!]|$)/i);
  if (pronounMatch) {
    const subject = inferSubjectFromContext(observation);
    const canonicalInstruction = subject
      ? toCanonicalFactInstruction(subject, pronounMatch[1] ?? "")
      : "";
    if (canonicalInstruction) {
      return {
        kind: "explicit_correction",
        canonicalInstruction,
        confidence: 0.9,
        reason: "detected pronoun correction and inferred subject from recent context",
      };
    }
  }

  const directSubject = rawText.match(/\b(?:the\s+)?([a-z0-9][a-z0-9 _\/-]{1,60})\s+(?:should be|is now)\s+(.+?)(?:[.?!]|$)/i);
  if (directSubject) {
    const canonicalInstruction = toCanonicalFactInstruction(directSubject[1] ?? "", directSubject[2] ?? "");
    if (canonicalInstruction) {
      return {
        kind: "explicit_correction",
        canonicalInstruction,
        confidence: 0.87,
        reason: "detected direct subject correction with explicit corrective cue",
      };
    }
  }

  return null;
}

export function proposeUserCorrectionFast(observation: UserMemoryObservation): UserMemoryProposal {
  return buildFastProposal(observation) ?? {
    kind: "noop",
    confidence: 0,
    reason: "no high-confidence deterministic correction pattern matched",
  };
}

function extractCompletionText(result: Awaited<ReturnType<CompleteFn>>): string {
  const blocks = Array.isArray(result.content) ? result.content : [];
  return blocks
    .map((block) => (typeof block?.text === "string" ? block.text : ""))
    .join("\n")
    .trim();
}

function parseProposalJson(text: string): UserMemoryProposal | null {
  const candidate = text.match(/\{[\s\S]*\}/)?.[0] ?? text;
  try {
    const parsed = JSON.parse(candidate) as Record<string, unknown>;
    const kind = typeof parsed.kind === "string" ? parsed.kind : "noop";
    const confidenceRaw = typeof parsed.confidence === "number" ? parsed.confidence : 0;
    const confidence = Math.max(0, Math.min(1, confidenceRaw));
    const reason = typeof parsed.reason === "string" ? parsed.reason : "model proposal";
    if (kind === "explicit_correction") {
      const canonicalInstruction = typeof parsed.canonicalInstruction === "string"
        ? sentenceCase(normalizeWhitespace(parsed.canonicalInstruction))
        : "";
      if (!canonicalInstruction) {
        return null;
      }
      return {
        kind,
        canonicalInstruction: /[.?!]$/.test(canonicalInstruction)
          ? canonicalInstruction
          : `${canonicalInstruction}.`,
        confidence,
        reason,
      };
    }
    return {
      kind: "noop",
      confidence,
      reason,
    };
  } catch {
    return null;
  }
}

const USER_MEMORY_PROPOSAL_SYSTEM = [
  "You decide whether the latest user message contains an explicit durable correction worth storing in OpenClawBrain.",
  "Summaries and recent messages are context only. The latest user quote is the authority.",
  "Return ONLY JSON with one of these shapes:",
  '{"kind":"noop","confidence":0.0,"reason":"brief reason"}',
  '{"kind":"explicit_correction","canonicalInstruction":"The codeword is giraffe.","confidence":0.93,"reason":"brief reason"}',
  "Store only explicit durable corrections/preferences/workflow rules. Do not store generic dissatisfaction, retries, or vague ambiguity.",
].join("\n");

export async function proposeUserCorrectionWithModel(params: {
  complete: CompleteFn;
  provider?: string;
  model: string;
  apiKey?: string;
  observation: UserMemoryObservation;
}): Promise<UserMemoryProposal> {
  const recentMessages = params.observation.recentMessages
    .map((message, index) => `${index + 1}. [${message.role}] ${normalizeWhitespace(message.content)}`)
    .join("\n");
  const recentSummaries = params.observation.recentSummaries
    .map((summary) => `${summary.summaryId} [${summary.kind} depth=${summary.depth}] ${normalizeWhitespace(summary.content)}`)
    .join("\n");

  const prompt = [
    `Conversation ID: ${params.observation.conversationId}`,
    params.observation.episodeId ? `Episode ID: ${params.observation.episodeId}` : "Episode ID: none",
    `Message ID: ${params.observation.messageId}`,
    "",
    "Latest user quote:",
    params.observation.userText,
    "",
    "Recent messages (oldest to newest):",
    recentMessages || "- none",
    "",
    "Recent context summaries:",
    recentSummaries || "- none",
  ].join("\n");

  const result = await params.complete({
    provider: params.provider,
    model: params.model,
    apiKey: params.apiKey,
    messages: [{ role: "user", content: prompt }],
    system: USER_MEMORY_PROPOSAL_SYSTEM,
    maxTokens: 250,
    temperature: 0,
  });

  return parseProposalJson(extractCompletionText(result)) ?? {
    kind: "noop",
    confidence: 0,
    reason: "model output was invalid JSON",
  };
}
