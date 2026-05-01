import type { ContextSelection, MemoryNode } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import type { RoutePlan } from './route-fn.js';
import { clipText } from './redact.js';
import type { MemoryStore } from './memory-store.js';

export class ContextSelector {
  private config: any;

  constructor(config: any) {
    this.config = config;
  }

  select(input: { packet: TurnEventPacket; plan: RoutePlan; candidates: MemoryNode[]; store?: MemoryStore }): ContextSelection {
    const { packet, plan, store } = input;
    let candidates = [...input.candidates];
    if (plan.retrievalPlan.graphDepth > 0 && store) {
      const expanded = new Set(candidates.map(c => c.id));
      for (const candidate of candidates.slice(0, 5)) {
        for (const connected of store.getConnectedMemories(candidate.id)) {
          if (!expanded.has(connected.id)) {
            expanded.add(connected.id);
            candidates.push(connected);
          }
        }
      }
    }
    const ranked = rankCandidates(packet, plan, candidates);
    const selected: ContextSelection['selected'] = [];
    const omitted: ContextSelection['omitted'] = [];
    const lines: string[] = [];
    let usedChars = 0;

    for (const item of ranked) {
      if (selected.length >= plan.injectionPlan.maxItems) {
        omitted.push({ memoryId: item.memory.id, reason: 'budget' });
        continue;
      }
      if (item.memory.supersededBy || item.memory.deletedAt) {
        omitted.push({ memoryId: item.memory.id, reason: 'superseded' });
        continue;
      }
      if (item.memory.confidence < this.config.routing.minRouteConfidence * 0.5) {
        omitted.push({ memoryId: item.memory.id, reason: 'low_confidence' });
        continue;
      }
      const line = formatMemoryLine(item.memory, item.reason);
      if (usedChars + line.length + 1 > plan.injectionPlan.maxChars) {
        omitted.push({ memoryId: item.memory.id, reason: 'budget' });
        continue;
      }
      lines.push(line);
      usedChars += line.length + 1;
      selected.push({
        memoryId: item.memory.id,
        reason: item.reason,
        useHow: item.useHow,
        confidence: item.score,
      });
    }

    const distilledContext = lines.length > 0
      ? clipText(`Relevant memory:\n${lines.map((line) => `- ${line}`).join('\n')}`, plan.injectionPlan.maxChars)
      : '';

    return {
      shouldInject: lines.length > 0,
      confidence: selected.length > 0 ? Math.max(...selected.map((item) => item.confidence)) : 0,
      selectedMemoryIds: selected.map((item) => item.memoryId),
      distilledContext,
      selected,
      omitted,
      audit: {
        promptBudgetUsedChars: distilledContext.length,
        risk: selected.some((item) => item.useHow === 'must_follow') ? 'medium' : 'low',
      },
    };
  }
}

function rankCandidates(packet: TurnEventPacket, plan: RoutePlan, candidates: MemoryNode[]) {
  const lower = packet.latestUserMessageRedacted.toLowerCase();
  return candidates.map((memory) => {
    let score = memory.importance * 0.4 + memory.confidence * 0.4 + memory.freshness * 0.2;
    let reason: ContextSelection['selected'][number]['reason'] = 'supporting_context';
    let useHow: ContextSelection['selected'][number]['useHow'] = 'consider';

    if (memory.type === 'correction') {
      score += 0.25;
      reason = 'directly_relevant_correction';
      useHow = 'must_follow';
    }
    if (memory.type === 'preference') {
      score += 0.15;
      reason = 'matching_user_preference';
      useHow = 'prefer';
    }
    if (memory.type === 'workflow') {
      score += 0.12;
      reason = 'repo_workflow';
      useHow = 'consider';
    }
    if (/\b(pnpm|npm|yarn|install|dependency|dependencies|build|test)\b/.test(lower) && /pnpm|npm|yarn|install|dependency|build|test/i.test(memory.content)) {
      score += 0.25;
    }
    if (/\b(plan|architecture|implementation|file-by-file)\b/.test(lower) && /plan|architecture|implementation|file-by-file/i.test(memory.content)) {
      score += 0.2;
    }

    return { memory, score, reason, useHow };
  }).sort((a, b) => b.score - a.score);
}

function formatMemoryLine(memory: MemoryNode, reason: ContextSelection['selected'][number]['reason']) {
  switch (reason) {
    case 'directly_relevant_correction':
      return `Must follow: ${memory.content}`;
    case 'matching_user_preference':
      return `User preference: ${memory.content}`;
    case 'repo_workflow':
      return `Workflow: ${memory.content}`;
    case 'tool_guidance':
      return `Tool guidance: ${memory.content}`;
    case 'contradiction_resolution':
      return `Latest correction: ${memory.content}`;
    default:
      return memory.content;
  }
}
