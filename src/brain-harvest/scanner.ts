import type { HarvestResult } from "../brain-runtime/evidence-detectors.js";

const EXPLICIT_SCANNER_PATTERNS = [
  /\bexpand for details about\b/i,
  /\bsummary bridge\b/i,
];

const DOC_MARKER_PATTERNS = [
  /\b(runbook|workflow|playbook|checklist|troubleshooting|procedure)\b/i,
  /\b(deploy|deployment|incident|recovery|rollback|pull request|release)\b/i,
];

const COMMAND_LINE_PATTERN = /^\s*(?:[-*]\s+|\d+\.\s+)?`?(gh|git|pnpm|npm|node|openclaw|ollama|curl|python3?|bash)\b.*$/gim;
const NUMBERED_STEP_PATTERN = /^\s*\d+\.\s+\S.+$/gm;
const BULLET_PATTERN = /^\s*[-*]\s+\S.+$/gm;
const HEADING_PATTERN = /^\s{0,3}#{1,6}\s+\S.+$/m;
const FILE_REF_PATTERN = /(?:^|[\s(])(?:\.?\/)?[\w./-]+\.(?:md|txt|ts|tsx|js|jsx|json|yaml|yml|sh|mjs)(?=$|[\s):,])/gim;
const IMPERATIVE_STEP_PATTERN = /^\s*(?:[-*]\s+|\d+\.\s+)?(?:inspect|check|retry|run|use|open|read|edit|verify|restart|re-?run|apply|deploy|create|install|record|compare|promote|rollback)\b/gim;

function countMatches(pattern: RegExp, content: string): number {
  const flags = pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`;
  const matcher = new RegExp(pattern.source, flags);
  return Array.from(content.matchAll(matcher)).length;
}

export function detectScannerEvidence(content: string): HarvestResult | null {
  for (const pattern of EXPLICIT_SCANNER_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: 0.25,
        source: "scanner",
        reason: `scanner marker: ${pattern.source}`,
        confidence: 0.7,
        kind: "scanner_signal",
      };
    }
  }

  const signals: string[] = [];
  let score = 0;

  for (const pattern of DOC_MARKER_PATTERNS) {
    if (pattern.test(content)) {
      signals.push(`doc:${pattern.source}`);
      score += 1.0;
      break;
    }
  }

  const numberedSteps = countMatches(NUMBERED_STEP_PATTERN, content);
  if (numberedSteps >= 2) {
    signals.push(`numbered_steps:${numberedSteps}`);
    score += 1.0;
  }

  const bulletLines = countMatches(BULLET_PATTERN, content);
  if (bulletLines >= 3) {
    signals.push(`bullets:${bulletLines}`);
    score += 0.5;
  }

  const commandLines = countMatches(COMMAND_LINE_PATTERN, content);
  if (commandLines >= 1) {
    signals.push(`commands:${commandLines}`);
    score += commandLines >= 2 ? 1.0 : 0.6;
  }

  const imperativeLines = countMatches(IMPERATIVE_STEP_PATTERN, content);
  if (imperativeLines >= 2) {
    signals.push(`imperatives:${imperativeLines}`);
    score += 0.8;
  }

  if (HEADING_PATTERN.test(content) && (numberedSteps >= 1 || bulletLines >= 2)) {
    signals.push("heading");
    score += 0.4;
  }

  const fileRefs = countMatches(FILE_REF_PATTERN, content);
  if (fileRefs >= 1) {
    signals.push(`file_refs:${fileRefs}`);
    score += 0.3;
  }

  if (score < 1.8) {
    return null;
  }

  return {
    value: 0.25,
    source: "scanner",
    reason: `scanner heuristic: ${signals.join(", ")}`,
    confidence: Math.min(0.8, 0.5 + signals.length * 0.05),
    kind: "scanner_signal",
  };
}
