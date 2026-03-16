import type { HarvestResult } from "../brain-runtime/evidence-detectors.js";

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

export function detectHumanEvidence(content: string): HarvestResult | null {
  for (const pattern of NEGATIVE_HUMAN_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: -0.8,
        source: "human",
        reason: `negative pattern: ${pattern.source}`,
        confidence: 0.9,
        kind: "human_feedback",
        extractor: "human_pattern",
      };
    }
  }
  for (const pattern of POSITIVE_HUMAN_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: 0.8,
        source: "human",
        reason: `positive pattern: ${pattern.source}`,
        confidence: 0.9,
        kind: "human_feedback",
        extractor: "human_pattern",
      };
    }
  }
  return null;
}
