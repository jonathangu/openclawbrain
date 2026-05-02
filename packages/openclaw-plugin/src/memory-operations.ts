import type { FeedbackDistillation, MemoryCandidate, MemoryNode } from './memory-types.js';
import type { TurnEventPacket } from './capture.js';
import { MemoryStore } from './memory-store.js';
import { redactText } from './redact.js';

export interface ApplyFeedbackResult {
  memoryIds: string[];
  storedCandidates: number;
  resolvedInjections: number;
}

export class MemoryOperationApplier {
  private store: MemoryStore;
  private config: any;

  constructor(options: { store: MemoryStore; config: any }) {
    this.store = options.store;
    this.config = options.config;
  }

  applyDistillation(distillation: FeedbackDistillation, packet: TurnEventPacket): ApplyFeedbackResult {
    if (!distillation.shouldStore && distillation.injectionFeedback.length === 0) {
      return { memoryIds: [], storedCandidates: 0, resolvedInjections: 0 };
    }

    const memoryIds: string[] = [];
    let storedCandidates = 0;
    let resolvedInjections = 0;

    this.store.transaction(() => {
      for (const candidate of distillation.memoryCandidates) {
        if (candidate.confidence < this.config.capture.minConfidence) continue;
        const node = this.upsertCandidate(candidate, packet);
        memoryIds.push(node.id);
        storedCandidates += 1;
        for (const contradiction of candidate.contradictions) {
          if (contradiction.action === 'supersede_existing' && contradiction.existingMemoryId) {
            this.store.supersedeMemory(contradiction.existingMemoryId, node.id);
          }
        }
      }

      for (const feedback of distillation.injectionFeedback) {
        this.store.resolveInjectionOutcome(feedback.injectionId, feedback.outcome, redactText(feedback.evidence, 300));
        resolvedInjections += 1;
      }

      this.store.insertProofEvent({
        agentId: packet.agentId,
        kind: 'llm_feedback_distillation_applied',
        sourceHook: packet.sourceHook,
        turnId: packet.turnId,
        sessionId: packet.sessionId,
        runId: packet.runId,
        rawTranscriptStored: false,
        payload: {
          confidence: distillation.confidence,
          feedbackType: distillation.feedbackType,
          memoryCount: memoryIds.length,
          resolvedInjections,
        },
      });
    });

    return { memoryIds, storedCandidates, resolvedInjections };
  }

  private upsertCandidate(candidate: MemoryCandidate, packet: TurnEventPacket): MemoryNode {
    const existing = this.store.findMemoryByNormalizedKey(
      packet.agentId,
      candidate.normalizedKey,
      candidate.scope.kind,
      candidate.scope.key,
    );

    if (existing) {
      const mergedTags = [...new Set([...(existing.tags || []), ...(candidate.tags || [])])];
      return this.store.updateMemory(existing.id, {
        content: redactText(candidate.distilledText, 2000),
        positive: candidate.positive ? redactText(candidate.positive, 500) : undefined,
        negative: candidate.negative ? redactText(candidate.negative, 500) : undefined,
        tags: mergedTags,
        importance: Math.max(existing.importance, clamp01(candidate.importanceHint)),
        confidence: Math.max(existing.confidence, clamp01(candidate.confidence)),
        captureCount: existing.captureCount + 1,
        lastSeenAt: new Date().toISOString(),
        sourceHook: packet.sourceHook,
        sourceTurnId: packet.turnId,
        sourceSessionId: packet.sessionId,
      }) || existing;
    }

    return this.store.insertMemory({
      agentId: packet.agentId,
      type: candidate.type,
      content: redactText(candidate.distilledText, 2000),
      positive: candidate.positive ? redactText(candidate.positive, 500) : undefined,
      negative: candidate.negative ? redactText(candidate.negative, 500) : undefined,
      scopeKind: candidate.scope.kind as any,
      scopeKey: candidate.scope.key,
      normalizedKey: candidate.normalizedKey,
      tags: candidate.tags,
      importance: clamp01(candidate.importanceHint),
      freshness: 1,
      confidence: clamp01(candidate.confidence),
      useCount: 0,
      usefulCount: 0,
      captureCount: 1,
      distilledByModel: this.config.llm.feedbackModel || this.config.llm.plannerModel || undefined,
      distillerPromptVersion: 'feedback-distiller-v1',
      distillationConfidence: clamp01(candidate.confidence),
      evidenceKind: packet.sourceHook,
      evidenceHash: String(packet.metadata.promptHash || ''),
      sourceHook: packet.sourceHook,
      sourceTurnId: packet.turnId,
      sourceSessionId: packet.sessionId,
    });
  }
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}
