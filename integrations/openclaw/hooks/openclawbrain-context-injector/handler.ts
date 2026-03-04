import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { createHash } from "node:crypto";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const RECALL_OR_CORRECTION_KEYWORDS = [
  "remember",
  "recall",
  "last time",
  "earlier",
  "we decided",
  "correction",
  "audit",
];

const FEEDBACK_PREFIXES: Record<
  string,
  { kind: "CORRECTION" | "TEACHING"; outcome: number | null }
> = {
  correction: { kind: "CORRECTION", outcome: -1.0 },
  fix: { kind: "CORRECTION", outcome: -1.0 },
  teaching: { kind: "TEACHING", outcome: 0.0 },
  note: { kind: "TEACHING", outcome: 0.0 },
};

function normalizeText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function isSlashCommand(message: string): boolean {
  return message.trim().startsWith("/");
}

function keywordIncreasedBudget(message: string): boolean {
  const haystack = message.toLowerCase();
  return RECALL_OR_CORRECTION_KEYWORDS.some((keyword) => haystack.includes(keyword));
}

function parseFeedbackDirective(
  message: string,
): { kind: "CORRECTION" | "TEACHING"; content: string; outcome: number | null } | null {
  const match = message.match(/^\s*(Correction|Fix|Teaching|Note):\s*(.*)$/i);
  if (!match) {
    return null;
  }
  const prefix = match[1]?.toLowerCase();
  const content = normalizeText(match[2]);
  if (!prefix || !content) {
    return null;
  }
  const entry = FEEDBACK_PREFIXES[prefix];
  if (!entry) {
    return null;
  }
  return { kind: entry.kind, content, outcome: entry.outcome };
}

function resolveAgentId(context: any): string {
  const workspaceDir = normalizeText(context?.workspaceDir);
  const agents = Array.isArray(context?.cfg?.agents?.list) ? context.cfg.agents.list : [];

  for (const agent of agents) {
    if (!agent || typeof agent !== "object") {
      continue;
    }
    const candidateWorkspace = normalizeText(agent.workspace);
    if (candidateWorkspace && candidateWorkspace === workspaceDir) {
      return normalizeText(agent.id) || normalizeText(agent.name) || normalizeText(agent.agentId) || "main";
    }
  }

  return "main";
}

function resolveMessageId(context: any, event: any): string {
  return (
    normalizeText(context?.messageId) ||
    normalizeText(context?.message?.id) ||
    normalizeText(context?.message?.messageId) ||
    normalizeText(event?.messageId) ||
    ""
  );
}

function resolveMessageTimestamp(context: any, event: any): string {
  return (
    normalizeText(context?.messageTimestamp) ||
    normalizeText(context?.messageTs) ||
    normalizeText(context?.message?.timestamp) ||
    normalizeText(context?.message?.ts) ||
    normalizeText(event?.timestamp) ||
    normalizeText(event?.ts) ||
    ""
  );
}

function deriveDedupKey(
  chatId: string,
  message: string,
  messageId: string,
  timestamp: string,
): { dedupKey: string | null; messageId: string | null } {
  if (messageId) {
    return { dedupKey: null, messageId };
  }
  if (!chatId) {
    return { dedupKey: null, messageId: null };
  }
  const hash = createHash("sha256")
    .update(`${chatId}|${timestamp}|${message}`)
    .digest("hex");
  return { dedupKey: `${chatId}:${hash}`, messageId: null };
}

async function runQueryBrain(
  statePath: string,
  message: string,
  chatId: string,
  maxBudget: number,
): Promise<string | null> {
  const args = [
    "-m",
    "openclawbrain.openclaw_adapter.query_brain",
    statePath,
    message,
    "--format",
    "prompt",
    "--exclude-bootstrap",
    "--redact",
    "--max-prompt-context-chars",
    String(maxBudget),
  ];

  if (chatId) {
    args.push("--chat-id", chatId);
  }

  try {
    const { stdout } = await execFileAsync("python3", args, {
      timeout: 2000,
      maxBuffer: 1024 * 1024,
    });
    const text = normalizeText(stdout);
    if (!text) {
      return null;
    }
    await fs.access(statePath);
    return text;
  } catch (_error) {
    return null;
  }
}

function runCaptureFeedback(
  statePath: string,
  chatId: string,
  directive: { kind: "CORRECTION" | "TEACHING"; content: string; outcome: number | null },
  dedupKey: string | null,
  messageId: string | null,
): void {
  if (!chatId) {
    return;
  }
  const args = [
    "-m",
    "openclawbrain.openclaw_adapter.capture_feedback",
    "--state",
    statePath,
    "--chat-id",
    chatId,
    "--kind",
    directive.kind,
    "--content",
    directive.content,
    "--lookback",
    "1",
  ];
  if (directive.outcome !== null) {
    args.push("--outcome", String(directive.outcome));
  }
  if (messageId) {
    args.push("--message-id", messageId);
  } else if (dedupKey) {
    args.push("--dedup-key", dedupKey);
  }

  void execFileAsync("python3", args, {
    timeout: 1000,
    maxBuffer: 128 * 1024,
  }).catch(() => undefined);
}

export default async function handler(event: any): Promise<any> {
  const context = event?.context;
  const message = normalizeText(context?.message ?? event?.message);
  if (!message || isSlashCommand(message)) {
    return event;
  }

  const agentId = resolveAgentId(context);
  const statePath = path.join(homedir(), ".openclawbrain", agentId, "state.json");
  const budget = keywordIncreasedBudget(message) ? 80000 : 20000;
  const chatId =
    normalizeText(context?.channelId) && normalizeText(context?.conversationId)
      ? `${normalizeText(context.channelId)}:${normalizeText(context.conversationId)}`
      : normalizeText(context?.chatId) || normalizeText(context?.messageId);
  const directive = parseFeedbackDirective(message);
  if (directive) {
    const rawMessageId = resolveMessageId(context, event);
    const timestamp = resolveMessageTimestamp(context, event);
    const dedup = deriveDedupKey(chatId, message, rawMessageId, timestamp);
    runCaptureFeedback(statePath, chatId, directive, dedup.dedupKey, dedup.messageId);
  }

  const promptContext = await runQueryBrain(statePath, message, chatId, budget);
  if (!promptContext) {
    return event;
  }

  const bodyForAgent = normalizeText(context?.bodyForAgent);
  if (!bodyForAgent) {
    context.bodyForAgent = promptContext;
    return event;
  }

  context.bodyForAgent = `${promptContext}\n\n${context.bodyForAgent}`;
  return event;
}
