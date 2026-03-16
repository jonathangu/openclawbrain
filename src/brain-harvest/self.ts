import type { HarvestResult } from "../brain-runtime/evidence-detectors.js";

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

export function detectSelfEvidence(content: string): HarvestResult | null {
  for (const pattern of TOOL_FAILURE_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: -0.5,
        source: "self",
        reason: `tool failure: ${pattern.source}`,
        confidence: 0.7,
        kind: "self_result",
        extractor: "self_pattern",
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
        extractor: "self_pattern",
      };
    }
  }
  return null;
}
