/**
 * Label harvesting from ingested messages.
 *
 * Structured evidence flow:
 * - detect human/self/scanner evidence separately
 * - persist raw evidence first
 * - let the worker resolve evidence into labels with trust ordering
 */

import type { BrainStore } from "../brain-store/store.js";
import { detectEvidence, type HarvestResult } from "./evidence-detectors.js";

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
    const result = this.detectLabel(params.role, params.content);
    if (!result) return;

    const exactEpisodeId = params.episodeId ?? this.resolveEpisodeIdForConversation?.(params.conversationId) ?? null;
    const matchingEpisode =
      (() => {
        if (!exactEpisodeId) {
          return null;
        }
        const episode = this.store.getEpisode(exactEpisodeId);
        return episode?.conversationId === params.conversationId ? episode : null;
      })()
      ?? this.store.getRecentEpisodesForConversation(params.conversationId, 5)[0]
      ?? null;

    if (!matchingEpisode) return;

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
        exactEpisodeId,
      },
    });

    this.log.info(
      `[brain] Harvested ${result.source} evidence: ${result.value.toFixed(2)} for episode ${matchingEpisode.id} (${result.reason})`,
    );
  }

  detectLabel(role: string, content: string): HarvestResult | null {
    return detectEvidence(role, content);
  }
}
