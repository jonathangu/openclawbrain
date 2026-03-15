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
}

export function detectEvidence(role: string, content: string): HarvestResult | null {
  if (role === "user") {
    return detectHumanEvidence(content);
  }

  if (role === "tool" || role === "assistant") {
    return detectSelfEvidence(content) ?? detectScannerEvidence(content);
  }

  return null;
}
