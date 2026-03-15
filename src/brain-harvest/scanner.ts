import type { HarvestResult } from "../brain-runtime/evidence-detectors.js";

const SCANNER_POSITIVE_PATTERNS = [
  /\n1\.\s.+\n2\.\s.+/i,
  /\b(runbook|workflow|playbook)\b/i,
  /\bissue\b.+\bpr\b.+\bcommit\b/i,
  /\bexpand for details about\b/i,
  /\bsummary bridge\b/i,
];

export function detectScannerEvidence(content: string): HarvestResult | null {
  for (const pattern of SCANNER_POSITIVE_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: 0.25,
        source: "scanner",
        reason: `scanner pattern: ${pattern.source}`,
        confidence: 0.55,
        kind: "scanner_signal",
      };
    }
  }
  return null;
}
