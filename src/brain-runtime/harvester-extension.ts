/**
 * Label harvesting from ingested messages.
 *
 * Structured evidence flow:
 * - detect human/self/scanner evidence separately
 * - persist raw evidence first
 * - let the worker resolve evidence into labels with trust ordering
 */

import type { BrainStore } from "../brain-store/store.js";
import { detectEvidence, detectEvidenceBatch, type HarvestResult } from "./evidence-detectors.js";

export class LabelHarvester {
  constructor(
    private store: BrainStore,
    private log: { info: (msg: string) => void; warn: (msg: string) => void },
    private resolveEpisodeIdForConversation?: (conversationId: number) => string | null | undefined,
  ) {}

  /**
   * Called from engine.ts after message ingestion.
   * Detects evidence from message content and attaches it to the matching episode.
   */
  async harvestFromMessage(params: {
    conversationId: number;
    episodeId?: string;
    role: string;
    content: string;
  }): Promise<void> {
    const results = this.detectLabels(params.role, params.content);
    if (results.length === 0) return;

    const explicitEpisodeId = params.episodeId ?? null;
    const resolvedEpisodeId = explicitEpisodeId
      ?? this.resolveEpisodeIdForConversation?.(params.conversationId)
      ?? null;
    const matchedResolvedEpisode = (() => {
      if (!resolvedEpisodeId) {
        return null;
      }
      const episode = this.store.getEpisode(resolvedEpisodeId);
      return episode?.conversationId === params.conversationId ? episode : null;
    })();
    const matchingEpisode = matchedResolvedEpisode
      ?? this.store.getRecentEpisodesForConversation(params.conversationId, 5)[0]
      ?? null;

    if (!matchingEpisode) return;

    const attributionMode = matchedResolvedEpisode
      ? explicitEpisodeId
        ? "explicit"
        : "resolver"
      : "recent_conversation_fallback";

    for (const [index, result] of results.entries()) {
      this.store.insertEvidence({
        episodeId: matchingEpisode.id,
        conversationId: params.conversationId,
        source: result.source,
        kind: result.kind,
        value: result.value,
        confidence: result.confidence,
        reason: result.reason,
        contentSnippet: params.content.slice(0, 240),
        metadata: {
          harvestedFromRole: params.role,
          explicitEpisodeId,
          resolvedEpisodeId,
          matchedEpisodeId: matchingEpisode.id,
          attributionMode,
          extractor: result.extractor ?? null,
          evidenceIndex: index,
          evidenceCount: results.length,
        },
      });

      this.log.info(
        `[brain] Harvested ${result.source} evidence: ${result.value.toFixed(2)} for episode ${matchingEpisode.id} (${result.reason})`,
      );
    }
  }

  detectLabel(role: string, content: string): HarvestResult | null {
    return detectEvidence(role, content);
  }

  detectLabels(role: string, content: string): HarvestResult[] {
    return detectEvidenceBatch(role, content);
  }
}
