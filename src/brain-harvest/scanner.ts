import type { HarvestMessagePart, HarvestResult } from "../brain-runtime/evidence-detectors.js";

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
const STRUCTURED_TOOL_NAMES = new Set(["bash", "git", "gh", "pnpm", "npm", "node", "openclaw", "python", "python3", "curl", "ollama", "codex", "claude"]);
const STRUCTURED_GUIDANCE_PART_TYPES = new Set(["file", "snapshot", "subtask", "patch", "compaction", "step_start", "step_finish", "retry"]);

type ContentSignalSummary = {
  docMarker: string | null;
  numberedSteps: number;
  bulletLines: number;
  commandLines: number;
  imperativeLines: number;
  hasHeading: boolean;
  fileRefs: number;
  score: number;
  signals: string[];
};

function countMatches(pattern: RegExp, content: string): number {
  const flags = pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`;
  const matcher = new RegExp(pattern.source, flags);
  return Array.from(content.matchAll(matcher)).length;
}

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

function extractCommand(input: unknown): string | undefined {
  const inputRecord = asRecord(input);
  return readString(inputRecord, ["command", "cmd", "shellCommand"])
    ?? readCommand(inputRecord?.args)
    ?? (typeof input === "string" && input.trim().length > 0 ? input.trim() : undefined);
}

function collectStructuredFileHints(part: HarvestMessagePart, metadata: Record<string, unknown>): string[] {
  const parsedInput = parseJson(part.toolInput);
  const parsedOutput = parseJson(part.toolOutput);
  const inputRecord = asRecord(parsedInput);
  const outputRecord = asRecord(parsedOutput);
  const rawRecord = asRecord(metadata.raw);
  const collected = new Set<string>();

  for (const value of [
    outputRecord?.filesTouched,
    outputRecord?.changedFiles,
    outputRecord?.files,
    outputRecord?.paths,
    inputRecord?.filesTouched,
    inputRecord?.files,
    inputRecord?.paths,
    readString(outputRecord, ["artifactPath", "outputPath", "reportPath", "logPath", "filePath", "path"]),
    readString(inputRecord, ["artifactPath", "outputPath", "reportPath", "logPath", "filePath", "path"]),
    readString(rawRecord, ["path", "filePath", "artifactPath"]),
  ]) {
    for (const item of readStringArray(value)) {
      collected.add(item);
    }
  }

  return Array.from(collected);
}

function collectStructuredPartDetails(part: HarvestMessagePart, metadata: Record<string, unknown>): {
  paths: string[];
  labels: string[];
} {
  const rawRecord = asRecord(metadata.raw);
  const paths = new Set<string>();
  const labels = new Set<string>();

  for (const value of [
    readString(rawRecord, ["path", "filePath", "artifactPath", "storageUri", "sourcePath", "targetPath", "outputPath"]),
    readString(rawRecord, ["fileName", "title", "label", "name", "summaryId", "taskId", "stepId"]),
    readString(rawRecord, ["summary", "description"]),
    typeof part.textContent === "string" && part.textContent.trim().length > 0 ? part.textContent.trim() : undefined,
  ]) {
    for (const item of readStringArray(value)) {
      if (/[/\\.]|^[A-Z0-9_-]+$/i.test(item)) {
        paths.add(item);
      }
      labels.add(item);
    }
  }

  return {
    paths: Array.from(paths),
    labels: Array.from(labels),
  };
}

function collectContentSignals(content: string): ContentSignalSummary {
  const signals: string[] = [];
  let score = 0;
  let docMarker: string | null = null;

  for (const pattern of DOC_MARKER_PATTERNS) {
    if (pattern.test(content)) {
      docMarker = pattern.source;
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

  const hasHeading = HEADING_PATTERN.test(content);
  if (hasHeading && (numberedSteps >= 1 || bulletLines >= 2)) {
    signals.push("heading");
    score += 0.4;
  }

  const fileRefs = countMatches(FILE_REF_PATTERN, content);
  if (fileRefs >= 1) {
    signals.push(`file_refs:${fileRefs}`);
    score += 0.3;
  }

  return {
    docMarker,
    numberedSteps,
    bulletLines,
    commandLines,
    imperativeLines,
    hasHeading,
    fileRefs,
    score,
    signals,
  };
}

function hasGuidanceShape(contentSignals: ContentSignalSummary): boolean {
  return Boolean(contentSignals.docMarker)
    || contentSignals.numberedSteps >= 2
    || (contentSignals.hasHeading && contentSignals.bulletLines >= 2)
    || contentSignals.imperativeLines >= 2;
}

function detectStructuredScannerEvidence(
  contentSignals: ContentSignalSummary,
  messageParts?: HarvestMessagePart[],
): HarvestResult | null {
  if (!messageParts || messageParts.length === 0 || !hasGuidanceShape(contentSignals)) {
    return null;
  }

  const toolNames = new Set<string>();
  const commands = new Set<string>();
  const toolFileHints = new Set<string>();
  const structuredPartTypes = new Set<string>();
  const structuredPaths = new Set<string>();
  const structuredLabels = new Set<string>();
  const partOrdinals: number[] = [];
  const rawTypes = new Set<string>();

  for (const part of messageParts) {
    const metadata = readPartMetadata(part);
    const rawType = typeof metadata.rawType === "string" ? metadata.rawType : null;
    if (rawType) {
      rawTypes.add(rawType);
    }

    if (typeof part.ordinal === "number") {
      partOrdinals.push(part.ordinal);
    }

    if (part.partType === "tool") {
      const toolName = typeof part.toolName === "string" ? part.toolName.trim() : "";
      if (toolName && STRUCTURED_TOOL_NAMES.has(toolName)) {
        toolNames.add(toolName);
      }

      const command = extractCommand(parseJson(part.toolInput));
      if (command) {
        commands.add(command);
      }

      for (const hint of collectStructuredFileHints(part, metadata)) {
        toolFileHints.add(hint);
      }
      continue;
    }

    if (STRUCTURED_GUIDANCE_PART_TYPES.has(part.partType)) {
      structuredPartTypes.add(part.partType);
      const details = collectStructuredPartDetails(part, metadata);
      for (const path of details.paths) {
        structuredPaths.add(path);
      }
      for (const label of details.labels) {
        structuredLabels.add(label);
      }
    }
  }

  if (toolNames.size > 0 && (commands.size > 0 || toolFileHints.size > 0)) {
    return {
      value: 0.25,
      source: "scanner",
      reason: `scanner structured tool-chain: tools=${Array.from(toolNames).join(",")}`,
      confidence: 0.85,
      kind: "scanner_signal",
      extractor: "structured_tool_chain",
      metadata: {
        toolNames: Array.from(toolNames),
        commands: Array.from(commands),
        fileHints: Array.from(toolFileHints),
        partOrdinals,
        rawTypes: Array.from(rawTypes),
        guidanceSignals: contentSignals.signals,
      },
    };
  }

  if (structuredPartTypes.size === 0 || (structuredPaths.size === 0 && structuredLabels.size === 0)) {
    return null;
  }

  return {
    value: 0.25,
    source: "scanner",
    reason: `scanner structured guidance parts: ${Array.from(structuredPartTypes).join(",")}`,
    confidence: 0.83,
    kind: "scanner_signal",
    extractor: "structured_guidance_parts",
    metadata: {
      structuredPartTypes: Array.from(structuredPartTypes),
      pathHints: Array.from(structuredPaths),
      labels: Array.from(structuredLabels),
      partOrdinals,
      rawTypes: Array.from(rawTypes),
      guidanceSignals: contentSignals.signals,
    },
  };
}

export function detectScannerEvidence(content: string, messageParts?: HarvestMessagePart[]): HarvestResult | null {
  for (const pattern of EXPLICIT_SCANNER_PATTERNS) {
    if (pattern.test(content)) {
      return {
        value: 0.25,
        source: "scanner",
        reason: `scanner marker: ${pattern.source}`,
        confidence: 0.7,
        kind: "scanner_signal",
        extractor: "scanner_marker",
        metadata: { marker: pattern.source },
      };
    }
  }

  const contentSignals = collectContentSignals(content);
  const structured = detectStructuredScannerEvidence(contentSignals, messageParts);
  if (structured) {
    return structured;
  }

  if (contentSignals.score < 1.8) {
    return null;
  }

  return {
    value: 0.25,
    source: "scanner",
    reason: `scanner heuristic: ${contentSignals.signals.join(", ")}`,
    confidence: Math.min(0.8, 0.5 + contentSignals.signals.length * 0.05),
    kind: "scanner_signal",
    extractor: "scanner_heuristic",
    metadata: {
      guidanceSignals: contentSignals.signals,
      numberedSteps: contentSignals.numberedSteps,
      bulletLines: contentSignals.bulletLines,
      commandLines: contentSignals.commandLines,
      imperativeLines: contentSignals.imperativeLines,
      fileRefs: contentSignals.fileRefs,
      hasHeading: contentSignals.hasHeading,
    },
  };
}
