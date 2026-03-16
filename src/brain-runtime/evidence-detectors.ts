import type { BrainEvidenceKind, RewardSource } from "../brain-core/types.js";
import { detectHumanEvidence } from "../brain-harvest/human.js";
import { detectScannerEvidence } from "../brain-harvest/scanner.js";
import { detectSelfEvidence, detectStructuredSelfEvidence } from "../brain-harvest/self.js";

export type HarvestMessagePart = {
  partType: string;
  ordinal?: number;
  textContent?: string | null;
  toolCallId?: string | null;
  toolName?: string | null;
  toolInput?: string | null;
  toolOutput?: string | null;
  metadata?: string | null;
};

export interface HarvestResult {
  value: number;
  source: RewardSource;
  reason: string;
  confidence?: number;
  kind: BrainEvidenceKind;
  extractor?: string;
  metadata?: Record<string, unknown>;
}

export function detectEvidenceBatch(
  role: string,
  content: string,
  messageParts?: HarvestMessagePart[],
): HarvestResult[] {
  if (role === "user") {
    const human = detectHumanEvidence(content);
    return human ? [human] : [];
  }

  if (role === "tool" || role === "assistant") {
    const structuredSelf = detectStructuredSelfEvidence(messageParts);
    const self = structuredSelf ?? detectSelfEvidence(content);
    const results = [
      self,
      detectScannerEvidence(content, messageParts),
    ].filter((result): result is HarvestResult => result !== null);

    const deduped = new Map<string, HarvestResult>();
    for (const result of results) {
      const key = [
        result.source,
        result.kind,
        result.value,
        result.reason,
        result.extractor ?? "",
      ].join("::");
      if (!deduped.has(key)) {
        deduped.set(key, result);
      }
    }
    return Array.from(deduped.values());
  }

  return [];
}

export function detectEvidence(
  role: string,
  content: string,
  messageParts?: HarvestMessagePart[],
): HarvestResult | null {
  return detectEvidenceBatch(role, content, messageParts)[0] ?? null;
}
