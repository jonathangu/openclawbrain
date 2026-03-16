import type { HarvestMessagePart, HarvestResult } from "../brain-runtime/evidence-detectors.js";

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

function parseJson(value: string | null | undefined): unknown {
  if (typeof value !== "string" || value.trim().length === 0) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function readPartMetadata(part: HarvestMessagePart): Record<string, unknown> {
  const parsed = parseJson(part.metadata);
  return parsed && typeof parsed === "object" && !Array.isArray(parsed)
    ? parsed as Record<string, unknown>
    : {};
}

function isStructuredToolResultPart(part: HarvestMessagePart): boolean {
  if (part.partType !== "tool") {
    return false;
  }
  const metadata = readPartMetadata(part);
  const rawType = typeof metadata.rawType === "string" ? metadata.rawType : "";
  const originalRole = typeof metadata.originalRole === "string" ? metadata.originalRole : "";
  return originalRole === "toolResult"
    || rawType === "tool_result"
    || rawType === "toolResult"
    || rawType === "function_call_output";
}

function classifyStructuredToolOutput(output: unknown): { ok: boolean; reason: string } | null {
  if (!output || typeof output !== "object" || Array.isArray(output)) {
    return null;
  }
  const record = output as Record<string, unknown>;

  if (record.isError === true || record.error !== undefined || record.errors !== undefined) {
    return { ok: false, reason: "structured tool output indicates error" };
  }
  if (record.ok === false || record.success === false || record.passed === false || record.failed === true) {
    return { ok: false, reason: "structured tool output indicates failure" };
  }
  if (typeof record.exitCode === "number" && record.exitCode > 0) {
    return { ok: false, reason: `structured tool output exitCode=${record.exitCode}` };
  }

  if (record.ok === true || record.success === true || record.passed === true) {
    return { ok: true, reason: "structured tool output indicates success" };
  }
  if (typeof record.exitCode === "number" && record.exitCode === 0) {
    return { ok: true, reason: "structured tool output exitCode=0" };
  }

  return null;
}

export function detectStructuredSelfEvidence(messageParts?: HarvestMessagePart[]): HarvestResult | null {
  if (!messageParts || messageParts.length === 0) {
    return null;
  }

  for (const part of messageParts) {
    if (!isStructuredToolResultPart(part)) {
      continue;
    }

    const metadata = readPartMetadata(part);
    if (metadata.isError === true) {
      return {
        value: -0.5,
        source: "self",
        reason: "structured tool result marked isError=true",
        confidence: 0.9,
        kind: "self_result",
        extractor: "structured_tool_result",
      };
    }

    const classified = classifyStructuredToolOutput(parseJson(part.toolOutput));
    if (!classified) {
      continue;
    }

    return {
      value: classified.ok ? 0.5 : -0.5,
      source: "self",
      reason: classified.reason,
      confidence: 0.9,
      kind: "self_result",
      extractor: "structured_tool_result",
    };
  }

  return null;
}

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
