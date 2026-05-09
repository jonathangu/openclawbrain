import type { ContextSelection, MemoryNode } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import type { RoutePlan } from './route-fn.js';
import { clipText } from './redact.js';
import type { MemoryStore } from './memory-store.js';
import { filterMemoriesForScope, scopeContextFromPacket } from './scope.js';
import { MemoryAuthorityResolver, authorityEventTypeForDecision } from './memory-authority.js';

export class ContextSelector {
  private config: any;

  constructor(config: any) {
    this.config = config;
  }

  select(input: { packet: TurnEventPacket; plan: RoutePlan; candidates: MemoryNode[]; store?: MemoryStore }): ContextSelection {
    const { packet, plan, store } = input;
    const scopeContext = scopeContextFromPacket(packet);
    let candidates = filterMemoriesForScope([...input.candidates], scopeContext);
    if (plan.retrievalPlan.graphDepth > 0 && store) {
      const expanded = new Set(candidates.map(c => c.id));
      for (const candidate of candidates.slice(0, 5)) {
        for (const connected of store.getConnectedMemories(candidate.id, plan.retrievalPlan.graphDepth, packet.agentId, scopeContext)) {
          if (!filterMemoriesForScope([connected], scopeContext).length) continue;
          if (!expanded.has(connected.id)) {
            expanded.add(connected.id);
            candidates.push(connected);
          }
        }
      }
    }
    const authority = new MemoryAuthorityResolver({ config: this.config, store }).resolve({ packet, plan, candidates });
    const authorityById = new Map(authority.map((item) => [item.memoryId, item]));
    recordAuthorityEvents(store, packet, authority);
    const ranked = rankCandidates(packet, plan, candidates);
    const selected: ContextSelection['selected'] = [];
    const omitted: ContextSelection['omitted'] = [];
    const lines: string[] = [];
    let usedChars = 0;

    for (const item of ranked) {
      const authorityDecision = authorityById.get(item.memory.id);
      if (authorityDecision && ['never_use', 'audit_only', 'abstain'].includes(authorityDecision.decision)) {
        omitted.push({
          memoryId: item.memory.id,
          reason: omittedReasonForAuthority(authorityDecision),
          authorityDecision: authorityDecision.decision,
          authorityReasons: authorityDecision.reasons,
        });
        continue;
      }
      if (selected.length >= plan.injectionPlan.maxItems) {
        omitted.push({ memoryId: item.memory.id, reason: 'budget', authorityDecision: authorityDecision?.decision, authorityReasons: authorityDecision?.reasons });
        continue;
      }
      if (item.memory.supersededBy || item.memory.deletedAt) {
        omitted.push({ memoryId: item.memory.id, reason: 'superseded', authorityDecision: authorityDecision?.decision, authorityReasons: authorityDecision?.reasons });
        continue;
      }
      if (item.memory.confidence < this.config.routing.minRouteConfidence * 0.5) {
        omitted.push({ memoryId: item.memory.id, reason: 'low_confidence', authorityDecision: authorityDecision?.decision, authorityReasons: authorityDecision?.reasons });
        continue;
      }
      if (item.memory.type === 'recall_rule' && plan.retrievalIntent?.intent !== 'recall_value_request') {
        omitted.push({ memoryId: item.memory.id, reason: 'would_pollute_prompt', authorityDecision: authorityDecision?.decision, authorityReasons: authorityDecision?.reasons });
        continue;
      }
      const line = formatMemoryLine(
        item.memory,
        item.reason,
        plan.retrievalIntent?.intent === 'recall_value_request' && canRevealRecallValue(item.memory, packet),
        authorityDecision?.decision,
      );
      if (usedChars + line.length + 1 > plan.injectionPlan.maxChars) {
        omitted.push({ memoryId: item.memory.id, reason: 'budget', authorityDecision: authorityDecision?.decision, authorityReasons: authorityDecision?.reasons });
        continue;
      }
      lines.push(line);
      usedChars += line.length + 1;
      selected.push({
        memoryId: item.memory.id,
        reason: item.reason,
        useHow: authorityDecision?.decision === 'confirm_before_use' || authorityDecision?.decision === 'verify_before_use' ? 'consider' : item.useHow,
        confidence: authorityDecision?.authorityScore ?? item.score,
        authorityDecision: authorityDecision?.decision,
        authorityReasons: authorityDecision?.reasons,
        requiredAction: authorityDecision?.requiredAction,
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
        risk: selected.some((item) => item.authorityDecision === 'confirm_before_use' || item.authorityDecision === 'verify_before_use' || item.useHow === 'must_follow') ? 'medium' : 'low',
        authority,
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
    if (memory.type === 'routing_rule') {
      score += 0.22;
      reason = 'tool_guidance';
      useHow = 'must_follow';
    }
    if (memory.type === 'tool_convention') {
      score += 0.18;
      reason = 'tool_guidance';
      useHow = 'consider';
    }
    if (memory.type === 'agent_assignment') {
      score += 0.2;
      reason = 'tool_guidance';
      useHow = 'must_follow';
    }
    if (memory.type === 'project_fact') {
      score += 0.12;
      reason = 'supporting_context';
      useHow = 'consider';
    }
    if (memory.type === 'outcome') {
      score += 0.1;
      reason = 'supporting_context';
      useHow = 'consider';
    }
    if (memory.type === 'recall_rule') {
      if (/\b(codeword|phrase|answer|what is|what's|tell me|give me)\b/.test(lower)) {
        score += 0.35;
        reason = 'directly_relevant_correction';
        useHow = 'must_follow';
      } else {
        score -= 1;
      }
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

function formatMemoryLine(memory: MemoryNode, reason: ContextSelection['selected'][number]['reason'], allowRecallValue = false, authorityDecision?: string) {
  const base = formatAuthoritativeMemoryLine(memory, reason, allowRecallValue);
  if (authorityDecision === 'verify_before_use') return `Verify before using: ${base}`;
  if (authorityDecision === 'confirm_before_use') return `Confirm before using: ${base}`;
  if (authorityDecision === 'weak_context') return `Soft context: ${base}`;
  return base;
}

function formatAuthoritativeMemoryLine(memory: MemoryNode, reason: ContextSelection['selected'][number]['reason'], allowRecallValue = false) {
  if (memory.type === 'recall_rule') {
    if (allowRecallValue && memory.positive) return `Recall rule: ${memory.content} Authorized answer: ${memory.positive}`;
    return `Recall rule: ${memory.content}`;
  }
  switch (reason) {
    case 'directly_relevant_correction':
      return `Must follow: ${memory.content}`;
    case 'matching_user_preference':
      return `User preference: ${memory.content}`;
    case 'repo_workflow':
      return `Workflow: ${memory.content}`;
    case 'tool_guidance':
      if (memory.type === 'routing_rule') return `Routing rule: ${memory.content}`;
      if (memory.type === 'agent_assignment') return `Agent assignment: ${memory.content}`;
      return `Tool convention: ${memory.content}`;
    case 'contradiction_resolution':
      return `Latest correction: ${memory.content}`;
    default:
      if (memory.type === 'project_fact') return `Project fact: ${memory.content}`;
      if (memory.type === 'outcome') return `Prior outcome: ${memory.content}`;
      return memory.content;
  }
}

function omittedReasonForAuthority(authority: NonNullable<ContextSelection['audit']['authority']>[number]): ContextSelection['omitted'][number]['reason'] {
  if (authority.decision === 'never_use') {
    if (authority.reasons.some((reason) => reason.includes('tombstoned'))) return 'tombstoned';
    if (authority.reasons.some((reason) => reason.includes('privacy'))) return 'privacy';
    return 'never_use';
  }
  if (authority.decision === 'audit_only') return 'audit_only';
  if (authority.reasons.some((reason) => reason.includes('expired'))) return 'expired';
  if (authority.reasons.some((reason) => reason.includes('stale'))) return 'stale';
  if (authority.reasons.some((reason) => reason.includes('current_instruction'))) return 'current_instruction_override';
  return 'irrelevant';
}

function recordAuthorityEvents(store: MemoryStore | undefined, packet: TurnEventPacket, authority: NonNullable<ContextSelection['audit']['authority']>) {
  if (!store?.insertMemoryAuthorityEvent) return;
  const seen = new Set<string>();
  for (const resolution of authority) {
    const eventType = authorityEventTypeForDecision(resolution.decision);
    const key = `${resolution.memoryId}:${eventType}:${resolution.reasons.join('|')}`;
    if (seen.has(key)) continue;
    seen.add(key);
    store.insertMemoryAuthorityEvent({
      agentId: packet.agentId,
      memoryId: resolution.memoryId,
      eventType,
      source: 'context_selector',
      turnId: packet.turnId,
      evidenceId: packet.metadata?.promptHash ? String(packet.metadata.promptHash) : undefined,
      reason: resolution.reasons.join('; '),
    });
  }
}

function canRevealRecallValue(memory: MemoryNode, packet: TurnEventPacket) {
  if (memory.type !== 'recall_rule' || !memory.positive || !memory.scopeKey) return false;
  const query = packet.latestUserMessageRedacted.toLowerCase();
  const terms = [memory.scopeKey, memory.normalizedKey, ...(memory.tags || [])]
    .flatMap((value) => String(value || '').toLowerCase().split(/[^a-z0-9]+/i))
    .filter((term) => term.length >= 3 && !GENERIC_RECALL_TERMS.has(term));
  return terms.some((term) => query.includes(term));
}

const GENERIC_RECALL_TERMS = new Set(['codeword', 'passphrase', 'phrase', 'answer', 'recall', 'rule', 'secret', 'value', 'app', 'project', 'repo']);
