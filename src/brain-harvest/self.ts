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

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function readPartMetadata(part: HarvestMessagePart): Record<string, unknown> {
  return asRecord(parseJson(part.metadata)) ?? {};
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

function readString(record: Record<string, unknown> | null, keys: string[]): string | undefined {
  if (!record) {
    return undefined;
  }
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim();
    }
  }
  return undefined;
}

function readNumber(record: Record<string, unknown> | null, keys: string[]): number | undefined {
  if (!record) {
    return undefined;
  }
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function readStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0);
  }
  if (typeof value === "string" && value.trim().length > 0) {
    return [value.trim()];
  }
  return [];
}

function readCommand(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim().length > 0) {
    return value.trim();
  }
  if (Array.isArray(value)) {
    const parts = value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0);
    return parts.length > 0 ? parts.join(" ") : undefined;
  }
  return undefined;
}

function extractCommand(input: unknown, output: unknown): string | undefined {
  const inputRecord = asRecord(input);
  const outputRecord = asRecord(output);

  return readString(inputRecord, ["command", "cmd", "shellCommand"])
    ?? readCommand(inputRecord?.args)
    ?? readString(outputRecord, ["command", "cmd", "shellCommand"])
    ?? readCommand(outputRecord?.args)
    ?? (typeof input === "string" && input.trim().length > 0 ? input.trim() : undefined);
}

function extractFilesTouched(input: unknown, output: unknown): string[] | undefined {
  const inputRecord = asRecord(input);
  const outputRecord = asRecord(output);
  const collected = new Set<string>();

  for (const value of [
    outputRecord?.filesTouched,
    outputRecord?.changedFiles,
    outputRecord?.files,
    outputRecord?.paths,
    inputRecord?.filesTouched,
    inputRecord?.files,
    inputRecord?.paths,
    readString(outputRecord, ["filePath", "path"]),
    readString(inputRecord, ["filePath", "path"]),
  ]) {
    for (const item of readStringArray(value)) {
      collected.add(item);
    }
  }

  return collected.size > 0 ? Array.from(collected) : undefined;
}

function extractArtifactPath(output: unknown): string | undefined {
  const record = asRecord(output);
  return readString(record, ["artifactPath", "outputPath", "reportPath", "logPath"]);
}

function buildStructuredToolMetadata(part: HarvestMessagePart): Record<string, unknown> {
  const metadata = readPartMetadata(part);
  const rawType = typeof metadata.rawType === "string" ? metadata.rawType : null;
  const parsedInput = parseJson(part.toolInput);
  const parsedOutput = parseJson(part.toolOutput);
  const result: Record<string, unknown> = {
    toolCallId: part.toolCallId ?? null,
    toolName: part.toolName ?? null,
    partOrdinal: part.ordinal ?? null,
    rawType,
  };

  const exitCode = readNumber(asRecord(parsedOutput), ["exitCode"]);
  if (exitCode !== undefined) {
    result.exitCode = exitCode;
  }

  const command = extractCommand(parsedInput, parsedOutput);
  if (command) {
    result.command = command;
  }

  const filesTouched = extractFilesTouched(parsedInput, parsedOutput);
  if (filesTouched) {
    result.filesTouched = filesTouched;
  }

  const artifactPath = extractArtifactPath(parsedOutput);
  if (artifactPath) {
    result.artifactPath = artifactPath;
  }

  return result;
}

function classifyStructuredToolOutput(output: unknown): { ok: boolean; reason: string } | null {
  const record = asRecord(output);
  if (!record) {
    return null;
  }

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
    const evidenceMetadata = buildStructuredToolMetadata(part);
    if (metadata.isError === true) {
      return {
        value: -0.5,
        source: "self",
        reason: "structured tool result marked isError=true",
        confidence: 0.9,
        kind: "self_result",
        extractor: "structured_tool_result",
        metadata: evidenceMetadata,
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
      metadata: evidenceMetadata,
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
