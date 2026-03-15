import type { BrainEvidenceKind, RewardSource } from "../brain-core/types.js";

export interface HarvestResult {
  value: number;
  source: RewardSource;
  reason: string;
  confidence?: number;
  kind: BrainEvidenceKind;
}

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

function detectHumanEvidence(content: string): HarvestResult | null {
  for (const pattern of NEGATIVE_HUMAN_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: -0.8,
        source: "human",
        reason: `negative pattern: ${pattern.source}`,
        confidence: 0.9,
        kind: "human_feedback",
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
      };
    }
  }
  return null;
}

function detectSelfEvidence(content: string): HarvestResult | null {
  for (const pattern of TOOL_FAILURE_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: -0.5,
        source: "self",
        reason: `tool failure: ${pattern.source}`,
        confidence: 0.7,
        kind: "self_result",
      };
    }
  }
  for (const pattern of TOOL_SUCCESS_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: 0.5,
        source: "self",
        reason: `tool success: ${pattern.source}`,
        confidence: 0.7,
        kind: "self_result",
      };
    }
  }
  return null;
}

function detectScannerEvidence(content: string): HarvestResult | null {
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

export function detectEvidence(role: string, content: string): HarvestResult | null {
  if (role === "user") {
    return detectHumanEvidence(content);
  }

  if (role === "tool" || role === "assistant") {
    return detectSelfEvidence(content) ?? detectScannerEvidence(content);
  }

  return null;
}
