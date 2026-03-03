import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
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

export default async function handler(event: any): Promise<any> {
  const context = event?.context;
  const message = normalizeText(context?.message ?? event?.message);
  if (!message || isSlashCommand(message)) {
    return event;
  }

  const agentId = resolveAgentId(context);
  const statePath = path.join(homedir(), ".openclawbrain", agentId, "state.json");
  const budget = keywordIncreasedBudget(message) ? 20000 : 12000;
  const chatId =
    normalizeText(context?.channelId) && normalizeText(context?.conversationId)
      ? `${normalizeText(context.channelId)}:${normalizeText(context.conversationId)}`
      : normalizeText(context?.chatId) || normalizeText(context?.messageId);

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
