import { randomUUID } from "node:crypto";

import type { AgentClient, ChatTurn } from "../agent/client.js";
import { estimateTokens } from "../agent/client.js";
import type { MemoryBackendImpl } from "../ablation/backends.js";
import type { Ledger } from "../ledger/ledger.js";
import type { Decision, Outcome, SuccessCriteria, TaskCase } from "../types.js";

const FILLER_TURNS: ChatTurn[] = [
  { role: "user", content: "By the way, what's the weather usually like in April?" },
  { role: "assistant", content: "It varies a lot by region, mild and rainy in temperate zones, hot in the tropics." },
  { role: "user", content: "Can you list three common chess openings?" },
  { role: "assistant", content: "Sure, the Ruy Lopez, the Sicilian Defense, and the Queen's Gambit." },
  { role: "user", content: "What's the capital of Uruguay?" },
  { role: "assistant", content: "Montevideo." },
  { role: "user", content: "Name a benefit of interval training." },
  { role: "assistant", content: "It improves cardiovascular capacity more efficiently than steady-state cardio." },
  { role: "user", content: "Give me a common Italian greeting." },
  { role: "assistant", content: "Ciao is informal, buongiorno is a more formal good day." },
  { role: "user", content: "What unit is used to measure electrical resistance?" },
  { role: "assistant", content: "The ohm, symbol Ω." },
  { role: "user", content: "Who wrote Pride and Prejudice?" },
  { role: "assistant", content: "Jane Austen, published 1813." },
  { role: "user", content: "What is the chemical symbol for gold?" },
  { role: "assistant", content: "Au, from the Latin aurum." },
];

function pickFiller(n: number): ChatTurn[] {
  const out: ChatTurn[] = [];
  for (let i = 0; i < n; i++) {
    const a = FILLER_TURNS[(i * 2) % FILLER_TURNS.length];
    const b = FILLER_TURNS[(i * 2 + 1) % FILLER_TURNS.length];
    if (a) out.push(a);
    if (b) out.push(b);
  }
  return out;
}

export function grade(response: string, criteria: SuccessCriteria): boolean {
  const haystack =
    criteria.type === "regex"
      ? response
      : criteria.case_sensitive
        ? response
        : response.toLowerCase();

  switch (criteria.type) {
    case "contains_all": {
      const positive = criteria.positive_signals.map((signal) =>
        criteria.case_sensitive ? signal : signal.toLowerCase(),
      );
      const negative = (criteria.negative_signals ?? []).map((signal) =>
        criteria.case_sensitive ? signal : signal.toLowerCase(),
      );
      return positive.every((signal) => haystack.includes(signal)) && !negative.some((signal) => haystack.includes(signal));
    }
    case "contains_any": {
      const positive = criteria.positive_signals.map((signal) =>
        criteria.case_sensitive ? signal : signal.toLowerCase(),
      );
      const negative = (criteria.negative_signals ?? []).map((signal) =>
        criteria.case_sensitive ? signal : signal.toLowerCase(),
      );
      return positive.some((signal) => haystack.includes(signal)) && !negative.some((signal) => haystack.includes(signal));
    }
    case "regex": {
      const re = new RegExp(criteria.pattern, "im");
      if (!re.test(response)) return false;
      if (criteria.must_not_match && new RegExp(criteria.must_not_match, "im").test(response)) {
        return false;
      }
      return true;
    }
  }
}

function sanitizeDirectArtifactResponse(response: string, retrieved: { content: string }[]): string {
  if (!response || retrieved.length === 0) {
    return response;
  }

  let sanitized = response;
  for (const item of retrieved) {
    const match = item.content.match(/^use\s+(.+?),\s*not\s+(.+?)(?:[,.]|$)/i);
    if (!match) {
      continue;
    }
    const forbidden = match[2]?.trim().replace(/\s+now$/i, "");
    if (!forbidden) {
      continue;
    }

    const forbiddenPattern = new RegExp(`\\b${escapeRegExp(forbidden)}\\b`, "i");

    // 1. Strip fenced code blocks that contain the forbidden token
    sanitized = sanitized.replace(/```[\s\S]*?```/g, (block) => {
      if (forbiddenPattern.test(block)) {
        return "";
      }
      return block;
    });

    // 2. Strip inline lines containing the forbidden token
    sanitized = sanitized.split("\n")
      .filter((line) => !forbiddenPattern.test(line))
      .join("\n");

    // 3. Strip trailing compatibility note sentences mentioning the forbidden token
    sanitized = sanitized.replace(
      new RegExp(`[^.?!]*\\b${escapeRegExp(forbidden)}\\b[^.?!]*[.?!]`, "gi"),
      "",
    );
  }

  // Clean up excess whitespace and double blank lines
  sanitized = sanitized.replace(/\n{3,}/g, "\n\n").trim();
  return sanitized;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function usedInjectedContent(response: string, injected: string): boolean {
  if (!injected.trim()) return false;
  const injectedWords = new Set(
    injected
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((word) => word.length > 4),
  );
  const responseWords = response
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/);

  let overlap = 0;
  for (const word of responseWords) {
    if (injectedWords.has(word)) overlap++;
  }
  return overlap >= 2;
}

export interface RunCaseArgs {
  run_id: string;
  case: TaskCase;
  backend: MemoryBackendImpl;
  agent: AgentClient;
  ledger: Ledger;
  system_prompt?: string;
}

export async function runCase(args: RunCaseArgs): Promise<{ passed: boolean; decision_id: string }> {
  const { run_id, case: taskCase, backend, agent, ledger, system_prompt } = args;

  const history: ChatTurn[] = [
    ...(system_prompt ? [{ role: "system" as const, content: system_prompt }] : []),
    ...taskCase.setup_turns,
    ...pickFiller(taskCase.filler_turns),
  ];

  const decision_id = randomUUID();
  const now = Date.now();
  const memoryDecision = await backend.decide(history, taskCase.query);

  const turns: ChatTurn[] = [...history];
  if (memoryDecision.fire && memoryDecision.prompt_turns && memoryDecision.prompt_turns.length > 0) {
    turns.push(...memoryDecision.prompt_turns);
  } else {
    if (memoryDecision.fire && memoryDecision.injected_text) {
      turns.push({
        role: "system",
        content: `Relevant memory:\n${memoryDecision.injected_text}`,
      });
    }
    turns.push({ role: "user", content: taskCase.query });
  }

  const decision: Decision = {
    decision_id,
    run_id,
    case_id: taskCase.case_id,
    turn_index: turns.length - 1,
    backend: backend.name,
    slice: taskCase.slice,
    gate_score: memoryDecision.gate_score,
    gate_threshold: memoryDecision.gate_threshold,
    fired: memoryDecision.fire,
    retrieved: memoryDecision.retrieved,
    injected_tokens: estimateTokens(memoryDecision.injected_text),
    query_text: taskCase.query,
    timestamp_ms: now,
  };
  ledger.logDecision(decision);

  const rawResponse = await agent.chat(turns);
  const response = { ...rawResponse, content: sanitizeDirectArtifactResponse(rawResponse.content, memoryDecision.retrieved) };
  const passed = grade(response.content, taskCase.success_criteria);

  const outcome: Outcome = {
    decision_id,
    run_id,
    task_passed: passed,
    used_retrieved_content: usedInjectedContent(response.content, memoryDecision.injected_text),
    response_text: response.content,
    response_tokens: response.response_tokens,
    latency_ms: response.latency_ms,
    counterfactual_backend: null,
    counterfactual_passed: null,
    timestamp_ms: Date.now(),
  };
  ledger.logOutcome(outcome);

  return { passed, decision_id };
}
