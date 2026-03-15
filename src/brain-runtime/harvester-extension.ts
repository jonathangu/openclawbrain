/**
 * Label harvesting from ingested messages.
 *
 * Four sources ranked by trust: human > self > scanner > teacher.
 * Human labels are detected from message content patterns.
 * Self labels come from tool success/failure signals.
 */

import type { RewardSource } from "../brain-core/types.js";
import type { BrainStore } from "../brain-store/store.js";

interface HarvestResult {
  value: number;
  source: RewardSource;
  reason: string;
}

// ─── Human label patterns ───

const NEGATIVE_HUMAN_PATTERNS = [
  /\bno[,.]?\s+(that'?s?\s+)?(not|wrong|incorrect)/i,
  /\bdon'?t\s+(use|do|try)/i,
  /\binstead\s+(use|do|try)/i,
  /\bactually[,]?\s+(it'?s|the|you\s+should)/i,
  /\bthat'?s\s+not\s+(right|correct|what)/i,
  /\bwrong\s+(file|path|approach|answer|tool)/i,
  /\bnot\s+what\s+i\s+(asked|wanted|meant)/i,
];

const POSITIVE_HUMAN_PATTERNS = [
  /\b(perfect|exactly|correct)\b/i,
  /\bthat('?s|\s+is)\s+(exactly\s+)?(right|correct|what\s+i)/i,
  /\bgreat[,!]\s+(that|this)\s+(work|help)/i,
  /\byes[,!.]\s+(that'?s?|exactly)/i,
];

// ─── Self label patterns (tool results) ───

const TOOL_FAILURE_PATTERNS = [
  /\berror\b/i,
  /\bfailed\b/i,
  /\bexception\b/i,
  /stack\s*trace/i,
  /\bEINVAL\b/,
  /\bENOENT\b/,
  /\bEACCES\b/,
  /exit\s+code\s+[1-9]/i,
];

const TOOL_SUCCESS_PATTERNS = [
  /\bsuccess(ful|fully)?\b/i,
  /\bpassed\b/i,
  /\bdeployed\b/i,
  /\bcreated\s+(commit|pr|branch)\b/i,
  /\b\d+\s+pass(ed|ing)\b/i,
  /\bfixed\b/i,
  /\bresolved\b/i,
];

const SCANNER_POSITIVE_PATTERNS = [
  /\n1\.\s.+\n2\.\s.+/i,
  /\b(runbook|workflow|playbook)\b/i,
  /\bissue\b.+\bpr\b.+\bcommit\b/i,
  /\bexpand for details about\b/i,
  /\bsummary bridge\b/i,
];

export class LabelHarvester {
  constructor(
    private store: BrainStore,
    private log: { info: (msg: string) => void; warn: (msg: string) => void },
  ) {}

  /**
   * Called from engine.ts after message ingestion.
   * Detects labels from message content and applies to recent episodes.
   */
  async harvestFromMessage(params: {
    conversationId: number;
    episodeId?: string;
    role: string;
    content: string;
  }): Promise<void> {
    const result = this.detectLabel(params.role, params.content);
    if (!result) return;

    const matchingEpisode =
      (() => {
        if (!params.episodeId) {
          return null;
        }
        const episode = this.store.getEpisode(params.episodeId);
        return episode?.conversationId === params.conversationId ? episode : null;
      })()
      ?? this.store.getRecentEpisodesForConversation(params.conversationId, 5)[0]
      ?? null;

    if (!matchingEpisode) return;

    this.store.insertLabel({
      episodeId: matchingEpisode.id,
      source: result.source,
      value: result.value,
      reason: result.reason,
    });

    this.log.info(`[brain] Harvested ${result.source} label: ${result.value.toFixed(2)} for episode ${matchingEpisode.id} (${result.reason})`);
  }

  detectLabel(role: string, content: string): HarvestResult | null {
    // Human labels from user messages
    if (role === "user") {
      const human = this.detectHumanLabel(content);
      if (human) return human;
    }

    // Self labels from tool results
    if (role === "tool" || role === "assistant") {
      const self = this.detectSelfLabel(content);
      if (self) return self;

      const scanner = this.detectScannerLabel(content);
      if (scanner) return scanner;
    }

    return null;
  }

  detectHumanLabel(content: string): HarvestResult | null {
    for (const pattern of NEGATIVE_HUMAN_PATTERNS) {
      if (pattern.test(content)) {
        return { value: -0.8, source: "human", reason: `negative pattern: ${pattern.source}` };
      }
    }
    for (const pattern of POSITIVE_HUMAN_PATTERNS) {
      if (pattern.test(content)) {
        return { value: 0.8, source: "human", reason: `positive pattern: ${pattern.source}` };
      }
    }
    return null;
  }

  detectSelfLabel(content: string): HarvestResult | null {
    for (const pattern of TOOL_FAILURE_PATTERNS) {
      if (pattern.test(content)) {
        return { value: -0.5, source: "self", reason: `tool failure: ${pattern.source}` };
      }
    }
    for (const pattern of TOOL_SUCCESS_PATTERNS) {
      if (pattern.test(content)) {
        return { value: 0.5, source: "self", reason: `tool success: ${pattern.source}` };
      }
    }
    return null;
  }

  detectScannerLabel(content: string): HarvestResult | null {
    for (const pattern of SCANNER_POSITIVE_PATTERNS) {
      if (pattern.test(content)) {
        return { value: 0.25, source: "scanner", reason: `scanner pattern: ${pattern.source}` };
      }
    }
    return null;
  }
}
