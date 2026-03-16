import type { BrainEvidenceKind, RewardSource } from "../brain-core/types.js";
import { detectHumanEvidence } from "../brain-harvest/human.js";
import { detectScannerEvidence } from "../brain-harvest/scanner.js";
import { detectSelfEvidence } from "../brain-harvest/self.js";

export interface HarvestResult {
  value: number;
  source: RewardSource;
  reason: string;
  confidence?: number;
  kind: BrainEvidenceKind;
  extractor?: string;
}

export function detectEvidenceBatch(role: string, content: string): HarvestResult[] {
  if (role === "user") {
    const human = detectHumanEvidence(content);
    return human ? [human] : [];
  }

  if (role === "tool" || role === "assistant") {
    const results = [
      detectSelfEvidence(content),
      detectScannerEvidence(content),
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

export function detectEvidence(role: string, content: string): HarvestResult | null {
  return detectEvidenceBatch(role, content)[0] ?? null;
}
