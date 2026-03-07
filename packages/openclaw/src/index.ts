import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";

import { compileRuntime } from "@openclawbrain/compiler";
import {
  CONTRACT_IDS,
  buildNormalizedEventExport,
  canonicalJson,
  checksumJsonPayload,
  createFeedbackEvent,
  createInteractionEvent,
  type ActivationPointerRecordV1,
  type FeedbackEventKind,
  type FeedbackEventV1,
  type InteractionEventV1,
  type NormalizedEventExportV1,
  type RouteMode,
  type RuntimeCompileResponseV1,
  validateNormalizedEventExport
} from "@openclawbrain/contracts";
import { inspectActivationState, type ActivationSlotInspection } from "@openclawbrain/pack-format";

const DEFAULT_AGENT_ID = "openclaw-runtime";
const FEEDBACK_KINDS = new Set<FeedbackEventKind>(["correction", "teaching", "approval", "suppression"]);


const RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT = "normalized_event_export_bundle.v1" as const;

export const RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT = {
  manifest: "manifest.json",
  payload: "normalized-event-export.json"
} as const;

export interface RuntimeEventExportBundleSummaryV1 {
  runtimeOwner: "openclaw";
  sessionId: string | null;
  channel: string | null;
  eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
  interactionCount: number;
  feedbackCount: number;
  sourceStreams: string[];
  contracts: NormalizedEventExportV1["provenance"]["contracts"];
}

export interface RuntimeEventExportBundleManifestV1 {
  contract: typeof RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT;
  exportName: string;
  exportedAt: string;
  payloadPath: string;
  payloadDigest: string;
  summary: RuntimeEventExportBundleSummaryV1;
}

export interface RuntimeEventExportBundleDescriptor {
  rootDir: string;
  manifestPath: string;
  payloadPath: string;
  manifest: RuntimeEventExportBundleManifestV1;
  normalizedEventExport: NormalizedEventExportV1;
}

export interface CompileRuntimeContextInput {
  activationRoot: string;
  message: string;
  agentId?: string;
  maxContextBlocks?: number;
  mode?: RouteMode;
  runtimeHints?: readonly string[];
}

export interface ActiveCompileTarget {
  activationRoot: string;
  activePointer: ActivationPointerRecordV1;
  inspection: ActivationSlotInspection;
}

export interface RuntimeCompileSuccess {
  ok: true;
  fallbackToStaticContext: false;
  activationRoot: string;
  activePackId: string;
  packRootDir: string;
  compileResponse: RuntimeCompileResponseV1;
  brainContext: string;
}

export interface RuntimeCompileFailure {
  ok: false;
  fallbackToStaticContext: true;
  activationRoot: string;
  error: string;
  brainContext: string;
}

export type RuntimeCompileResult = RuntimeCompileSuccess | RuntimeCompileFailure;

export interface RuntimeTurnCompileInput {
  createdAt?: string | null;
  sequence?: number | null;
  eventId?: string | null;
}

export interface RuntimeTurnDeliveryInput {
  createdAt?: string | null;
  sequence?: number | null;
  eventId?: string | null;
  messageId?: string | null;
}

export interface RuntimeTurnFeedbackInput {
  content: string;
  createdAt?: string | null;
  sequence?: number | null;
  eventId?: string | null;
  kind?: FeedbackEventKind | null;
  messageId?: string | null;
  relatedInteractionId?: string | null;
}

export interface RuntimeTurnExportInput {
  rootDir: string;
  exportName?: string | null;
  exportedAt?: string | null;
}

export interface OpenClawRuntimeTurnInput {
  activationRoot?: string | null;
  agentId?: string | null;
  sessionId: string;
  channel: string;
  sourceStream?: string | null;
  userMessage: string;
  createdAt?: string | null;
  sequenceStart?: number | null;
  maxContextBlocks?: number;
  mode?: RouteMode;
  runtimeHints?: readonly string[];
  compile?: RuntimeTurnCompileInput | null;
  delivery?: boolean | RuntimeTurnDeliveryInput | null;
  feedback?: readonly (RuntimeTurnFeedbackInput | null)[] | null;
  export?: RuntimeTurnExportInput | null;
}

export interface RuntimeEventExportNoWrite {
  ok: true;
  wroteBundle: false;
  normalizedEventExport: NormalizedEventExportV1;
}

export interface RuntimeEventExportWriteSuccess {
  ok: true;
  wroteBundle: true;
  normalizedEventExport: NormalizedEventExportV1;
  rootDir: string;
  manifestPath: string;
  payloadPath: string;
  manifest: RuntimeEventExportBundleManifestV1;
}

export interface RuntimeEventExportFailure {
  ok: false;
  wroteBundle: false;
  error: string;
}

export type RuntimeEventExportResult =
  | RuntimeEventExportNoWrite
  | RuntimeEventExportWriteSuccess
  | RuntimeEventExportFailure;

export interface RunRuntimeTurnOptions {
  activationRoot?: string;
  failOpen?: boolean;
}

export type RuntimeTurnResult = RuntimeCompileResult & {
  eventExport: RuntimeEventExportResult;
  warnings: string[];
};

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}


function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function resolveBundlePayloadPath(rootDir: string, payloadPath: string): string {
  const resolved = path.resolve(rootDir, payloadPath);
  const relative = path.relative(rootDir, resolved);

  if (path.isAbsolute(payloadPath) || relative.startsWith("..") || relative === "") {
    throw new Error("event export bundle payloadPath must stay within the bundle root");
  }

  return resolved;
}

export function buildRuntimeEventExportBundleManifest(input: {
  exportName: string;
  exportedAt: string;
  payloadPath: string;
  normalizedEventExport: NormalizedEventExportV1;
}): RuntimeEventExportBundleManifestV1 {
  return {
    contract: RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT,
    exportName: input.exportName,
    exportedAt: input.exportedAt,
    payloadPath: input.payloadPath,
    payloadDigest: checksumJsonPayload(input.normalizedEventExport),
    summary: {
      runtimeOwner: input.normalizedEventExport.provenance.runtimeOwner,
      sessionId: input.normalizedEventExport.provenance.sessionId,
      channel: input.normalizedEventExport.provenance.channel,
      eventRange: {
        start: input.normalizedEventExport.range.start,
        end: input.normalizedEventExport.range.end,
        count: input.normalizedEventExport.range.count
      },
      interactionCount: input.normalizedEventExport.provenance.interactionCount,
      feedbackCount: input.normalizedEventExport.provenance.feedbackCount,
      sourceStreams: [...input.normalizedEventExport.provenance.sourceStreams],
      contracts: [...input.normalizedEventExport.provenance.contracts]
    }
  };
}

export function validateRuntimeEventExportBundleManifest(
  value: RuntimeEventExportBundleManifestV1,
  normalizedEventExport?: NormalizedEventExportV1
): string[] {
  const errors: string[] = [];

  if (value.contract !== RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT) {
    errors.push("normalized_event_export_bundle.v1 contract is required");
  }
  if (value.exportName.length === 0) {
    errors.push("exportName is required");
  }
  if (Number.isNaN(Date.parse(value.exportedAt))) {
    errors.push("exportedAt must be an ISO timestamp");
  }
  if (value.payloadPath.length === 0) {
    errors.push("payloadPath is required");
  }
  if (value.payloadDigest.length === 0) {
    errors.push("payloadDigest is required");
  }

  if (normalizedEventExport !== undefined) {
    const exportErrors = validateNormalizedEventExport(normalizedEventExport);
    if (exportErrors.length > 0) {
      errors.push(...exportErrors);
    }

    const rebuilt = buildRuntimeEventExportBundleManifest({
      exportName: value.exportName,
      exportedAt: value.exportedAt,
      payloadPath: value.payloadPath,
      normalizedEventExport
    });

    if (rebuilt.payloadDigest !== value.payloadDigest) {
      errors.push("event export bundle payloadDigest does not match the supplied normalized event export");
    }
    if (canonicalJson(rebuilt.summary) !== canonicalJson(value.summary)) {
      errors.push("event export bundle summary does not match the supplied normalized event export");
    }
  }

  return errors;
}

export function loadRuntimeEventExportBundle(rootDir: string): RuntimeEventExportBundleDescriptor {
  const resolvedRoot = path.resolve(rootDir);
  const manifestPath = path.join(resolvedRoot, RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.manifest);
  const manifest = readJsonFile<RuntimeEventExportBundleManifestV1>(manifestPath);
  const payloadPath = resolveBundlePayloadPath(resolvedRoot, manifest.payloadPath);
  const normalizedEventExport = readJsonFile<NormalizedEventExportV1>(payloadPath);
  const validationErrors = validateRuntimeEventExportBundleManifest(manifest, normalizedEventExport);

  if (validationErrors.length > 0) {
    throw new Error(`event export bundle is invalid: ${validationErrors.join("; ")}`);
  }

  return {
    rootDir: resolvedRoot,
    manifestPath,
    payloadPath,
    manifest,
    normalizedEventExport
  };
}

function normalizeNonEmptyString(value: string | null | undefined, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${fieldName} is required`);
  }

  return value.trim();
}

function normalizeOptionalString(value: string | null | undefined): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function normalizeNonNegativeInteger(
  value: number | null | undefined,
  fieldName: string,
  fallbackValue: number
): number {
  if (value === undefined || value === null) {
    return fallbackValue;
  }

  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${fieldName} must be a non-negative integer`);
  }

  return value;
}

function normalizeIsoTimestamp(
  value: string | null | undefined,
  fieldName: string,
  fallbackValue?: string | null
): string {
  const candidate = value ?? fallbackValue;
  if (candidate === undefined || candidate === null || candidate === "") {
    throw new Error(`${fieldName} is required`);
  }

  if (typeof candidate !== "string" || Number.isNaN(Date.parse(candidate))) {
    throw new Error(`${fieldName} must be an ISO timestamp`);
  }

  return new Date(candidate).toISOString();
}

function normalizeMode(value: RouteMode | undefined): RouteMode {
  return value ?? "heuristic";
}

function normalizeRuntimeHints(value: readonly string[] | undefined): string[] {
  if (value === undefined) {
    return [];
  }

  if (value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
    throw new Error("runtimeHints must be an array of non-empty strings");
  }

  return value.map((item) => item.trim());
}

function deterministicEventId(value: unknown): string {
  const digest = createHash("sha256").update(JSON.stringify(value)).digest("hex");
  return `evt-${digest.slice(0, 16)}`;
}

function nextSequenceFactory(startValue = 1): (explicitValue?: number | null) => number {
  let cursor = normalizeNonNegativeInteger(startValue, "sequenceStart", 1);

  return (explicitValue?: number | null) => {
    if (explicitValue !== undefined && explicitValue !== null) {
      const normalized = normalizeNonNegativeInteger(explicitValue, "sequence", explicitValue);
      cursor = Math.max(cursor, normalized + 1);
      return normalized;
    }

    const current = cursor;
    cursor += 1;
    return current;
  };
}

function isFeedbackKind(value: string): value is FeedbackEventKind {
  return FEEDBACK_KINDS.has(value as FeedbackEventKind);
}

function isPresent<T>(value: T | null): value is T {
  return value !== null;
}

export function classifyFeedbackKind(content: string): FeedbackEventKind {
  const normalized = content.trim().toLowerCase();
  if (normalized.length === 0) {
    return "teaching";
  }
  if (/\b(stop|suppress|mute|silence|pause|don't send|do not send)\b/u.test(normalized)) {
    return "suppression";
  }
  if (/\b(approved|approve|looks good|ship it|exactly right|that works)\b/u.test(normalized)) {
    return "approval";
  }
  if (
    /\b(wrong|incorrect|correction|not right|do this instead|not x[, ]+y)\b/u.test(normalized) ||
    normalized.startsWith("no") ||
    normalized.startsWith("wrong")
  ) {
    return "correction";
  }
  return "teaching";
}

export function formatPromptContext(compileResponse: RuntimeCompileResponseV1): string {
  const lines = [
    "[BRAIN_CONTEXT v1]",
    `PACK_ID: ${compileResponse.packId}`,
    `MODE: ${compileResponse.diagnostics.modeEffective}`
  ];

  if (compileResponse.diagnostics.routerIdentity !== null) {
    lines.push(`ROUTER: ${compileResponse.diagnostics.routerIdentity}`);
  }

  if (compileResponse.selectedContext.length > 0) {
    lines.push("");
  }

  for (const block of compileResponse.selectedContext) {
    lines.push(`SOURCE: ${block.source}`);
    lines.push(`BLOCK_ID: ${block.id}`);
    lines.push(block.text.trim());
    lines.push("");
  }

  if (lines[lines.length - 1] === "") {
    lines.pop();
  }

  lines.push("[/BRAIN_CONTEXT]");
  return `${lines.join("\n")}\n`;
}

function fallbackCompileResult(error: unknown, activationRoot: string): RuntimeCompileFailure {
  return {
    ok: false,
    fallbackToStaticContext: true,
    activationRoot: path.resolve(activationRoot),
    error: toErrorMessage(error),
    brainContext: ""
  };
}

export function resolveActivePackForCompile(activationRoot: string): ActiveCompileTarget {
  const resolvedActivationRoot = path.resolve(normalizeNonEmptyString(activationRoot, "activationRoot"));
  const inspection = inspectActivationState(resolvedActivationRoot);
  const activePointer = inspection.pointers.active;

  if (inspection.active === null || activePointer === null) {
    throw new Error(`No active pack pointer found in ${resolvedActivationRoot}`);
  }

  if (!inspection.active.activationReady) {
    throw new Error(`Active pack is not activation-ready: ${inspection.active.findings.join("; ")}`);
  }

  return {
    activationRoot: resolvedActivationRoot,
    activePointer,
    inspection: inspection.active
  };
}

export function compileRuntimeContext(input: CompileRuntimeContextInput): RuntimeCompileResult {
  const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
  const agentId = normalizeOptionalString(input.agentId) ?? process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
  const runtimeHints = normalizeRuntimeHints(input.runtimeHints);

  try {
    const target = resolveActivePackForCompile(activationRoot);
    const compileResponse = compileRuntime(target.activePointer.packRootDir, {
      contract: CONTRACT_IDS.runtimeCompile,
      agentId,
      userMessage: normalizeNonEmptyString(input.message, "message"),
      maxContextBlocks: normalizeNonNegativeInteger(input.maxContextBlocks, "maxContextBlocks", 4),
      modeRequested: normalizeMode(input.mode),
      activePackId: target.activePointer.packId,
      ...(runtimeHints.length > 0 ? { runtimeHints } : {})
    });

    return {
      ok: true,
      fallbackToStaticContext: false,
      activationRoot,
      activePackId: target.activePointer.packId,
      packRootDir: path.resolve(target.activePointer.packRootDir),
      compileResponse,
      brainContext: formatPromptContext(compileResponse)
    };
  } catch (error) {
    return fallbackCompileResult(error, activationRoot);
  }
}

function buildCompileInteractionEvent(input: {
  turn: OpenClawRuntimeTurnInput;
  compileResult: RuntimeCompileResult;
  sourceStream: string;
  nextSequence: (explicitValue?: number | null) => number;
  createdAt: string;
  agentId: string;
}): InteractionEventV1 | null {
  if (!input.compileResult.ok) {
    return null;
  }

  const sequence = input.nextSequence(input.turn.compile?.sequence);
  const eventId = normalizeOptionalString(input.turn.compile?.eventId) ?? deterministicEventId({
    channel: input.turn.channel,
    createdAt: input.createdAt,
    kind: "memory_compiled",
    packId: input.compileResult.compileResponse.packId,
    sequence,
    sessionId: input.turn.sessionId,
    source: input.sourceStream
  });

  return createInteractionEvent({
    eventId,
    agentId: input.agentId,
    sessionId: input.turn.sessionId,
    channel: input.turn.channel,
    sequence,
    kind: "memory_compiled",
    createdAt: input.createdAt,
    source: {
      runtimeOwner: "openclaw",
      stream: input.sourceStream
    },
    packId: input.compileResult.compileResponse.packId
  });
}

function buildDeliveryInteractionEvent(input: {
  turn: OpenClawRuntimeTurnInput;
  compileResult: RuntimeCompileResult;
  sourceStream: string;
  nextSequence: (explicitValue?: number | null) => number;
  defaultCreatedAt: string;
  agentId: string;
}): InteractionEventV1 | null {
  if (input.turn.delivery === undefined || input.turn.delivery === null || input.turn.delivery === false) {
    return null;
  }

  const delivery = typeof input.turn.delivery === "object" ? input.turn.delivery : {};
  const createdAt = normalizeIsoTimestamp(delivery.createdAt, "delivery.createdAt", input.defaultCreatedAt);
  const sequence = input.nextSequence(delivery.sequence);
  const messageId = normalizeOptionalString(delivery.messageId);
  const eventId = normalizeOptionalString(delivery.eventId) ?? deterministicEventId({
    channel: input.turn.channel,
    createdAt,
    kind: "message_delivered",
    messageId: messageId ?? null,
    packId: input.compileResult.ok ? input.compileResult.compileResponse.packId : null,
    sequence,
    sessionId: input.turn.sessionId,
    source: input.sourceStream
  });

  return createInteractionEvent({
    eventId,
    agentId: input.agentId,
    sessionId: input.turn.sessionId,
    channel: input.turn.channel,
    sequence,
    kind: "message_delivered",
    createdAt,
    source: {
      runtimeOwner: "openclaw",
      stream: input.sourceStream
    },
    ...(input.compileResult.ok ? { packId: input.compileResult.compileResponse.packId } : {}),
    ...(messageId !== undefined ? { messageId } : {})
  });
}

function buildFeedbackEvents(input: {
  turn: OpenClawRuntimeTurnInput;
  sourceStream: string;
  nextSequence: (explicitValue?: number | null) => number;
  defaultCreatedAt: string;
  compileInteraction: InteractionEventV1 | null;
  agentId: string;
}): FeedbackEventV1[] {
  const feedbackItems = input.turn.feedback ?? [];

  return feedbackItems.map((item, index) => {
    if (item === null) {
      throw new Error(`feedback[${index}] must be an object`);
    }

    const content = normalizeNonEmptyString(item.content, `feedback[${index}].content`);
    const createdAt = normalizeIsoTimestamp(item.createdAt, `feedback[${index}].createdAt`, input.defaultCreatedAt);
    const sequence = input.nextSequence(item.sequence);
    const kind = item.kind === undefined || item.kind === null ? classifyFeedbackKind(content) : item.kind;

    if (!isFeedbackKind(kind)) {
      throw new Error(`feedback[${index}].kind must be correction, teaching, approval, or suppression`);
    }

    const messageId = normalizeOptionalString(item.messageId);
    const eventId = normalizeOptionalString(item.eventId) ?? deterministicEventId({
      channel: input.turn.channel,
      content,
      createdAt,
      kind,
      messageId: messageId ?? null,
      sequence,
      sessionId: input.turn.sessionId,
      source: input.sourceStream
    });
    const relatedInteractionId = normalizeOptionalString(item.relatedInteractionId) ?? input.compileInteraction?.eventId;

    return createFeedbackEvent({
      eventId,
      agentId: input.agentId,
      sessionId: input.turn.sessionId,
      channel: input.turn.channel,
      sequence,
      kind,
      createdAt,
      source: {
        runtimeOwner: "openclaw",
        stream: input.sourceStream
      },
      content,
      ...(messageId !== undefined ? { messageId } : {}),
      ...(relatedInteractionId !== undefined ? { relatedInteractionId } : {})
    });
  });
}

export function buildNormalizedRuntimeEventExport(
  turn: OpenClawRuntimeTurnInput,
  compileResult: RuntimeCompileResult
): NormalizedEventExportV1 {
  const agentId = normalizeOptionalString(turn.agentId) ?? process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
  const sessionId = normalizeNonEmptyString(turn.sessionId, "sessionId");
  const channel = normalizeNonEmptyString(turn.channel, "channel");
  const sourceStream = normalizeOptionalString(turn.sourceStream) ?? `openclaw/runtime/${channel}`;
  const compileCreatedAt = normalizeIsoTimestamp(turn.compile?.createdAt, "compile.createdAt", turn.createdAt);
  const nextSequence = nextSequenceFactory(turn.sequenceStart ?? 1);
  const normalizedTurn: OpenClawRuntimeTurnInput = {
    ...turn,
    agentId,
    channel,
    sessionId
  };

  const compileInteraction = buildCompileInteractionEvent({
    turn: normalizedTurn,
    compileResult,
    sourceStream,
    nextSequence,
    createdAt: compileCreatedAt,
    agentId
  });
  const feedbackEvents = buildFeedbackEvents({
    turn: normalizedTurn,
    sourceStream,
    nextSequence,
    defaultCreatedAt: compileCreatedAt,
    compileInteraction,
    agentId
  });
  const deliveryInteraction = buildDeliveryInteractionEvent({
    turn: normalizedTurn,
    compileResult,
    sourceStream,
    nextSequence,
    defaultCreatedAt: compileCreatedAt,
    agentId
  });
  const interactionEvents = [compileInteraction, deliveryInteraction].filter(isPresent);

  if (interactionEvents.length === 0 && feedbackEvents.length === 0) {
    throw new Error("runtime turn did not produce any normalized events");
  }

  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents,
    feedbackEvents
  });
  const validationErrors = validateNormalizedEventExport(normalizedEventExport);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
  }

  return normalizedEventExport;
}

export function writeRuntimeEventExportBundle(
  turn: OpenClawRuntimeTurnInput,
  normalizedEventExport: NormalizedEventExportV1
): RuntimeEventExportNoWrite | RuntimeEventExportWriteSuccess {
  if (turn.export === undefined || turn.export === null) {
    return {
      ok: true,
      wroteBundle: false,
      normalizedEventExport
    };
  }

  const rootDir = normalizeNonEmptyString(turn.export.rootDir, "export.rootDir");
  const exportName =
    normalizeOptionalString(turn.export.exportName) ??
    `${turn.sessionId}-${normalizedEventExport.range.start}-${normalizedEventExport.range.end}`;
  const exportedAt = normalizeIsoTimestamp(
    turn.export.exportedAt,
    "export.exportedAt",
    normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? new Date().toISOString()
  );

  const resolvedRoot = path.resolve(rootDir);
  const manifest = buildRuntimeEventExportBundleManifest({
    exportName,
    exportedAt,
    payloadPath: RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.payload,
    normalizedEventExport
  });
  const manifestPath = path.join(resolvedRoot, RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.manifest);
  const payloadPath = path.join(resolvedRoot, manifest.payloadPath);

  mkdirSync(path.dirname(payloadPath), { recursive: true });
  writeFileSync(payloadPath, canonicalJson(normalizedEventExport), "utf8");
  writeFileSync(manifestPath, canonicalJson(manifest), "utf8");
  const descriptor = loadRuntimeEventExportBundle(resolvedRoot);

  return {
    ok: true,
    wroteBundle: true,
    normalizedEventExport,
    rootDir: descriptor.rootDir,
    manifestPath: descriptor.manifestPath,
    payloadPath: descriptor.payloadPath,
    manifest: descriptor.manifest
  };
}

export function runRuntimeTurn(turn: OpenClawRuntimeTurnInput, options: RunRuntimeTurnOptions = {}): RuntimeTurnResult {
  const agentId = normalizeOptionalString(turn.agentId);
  const compileInput: CompileRuntimeContextInput = {
    activationRoot: options.activationRoot ?? normalizeNonEmptyString(turn.activationRoot ?? undefined, "activationRoot"),
    message: normalizeNonEmptyString(turn.userMessage, "userMessage"),
    ...(agentId !== undefined ? { agentId } : {}),
    ...(turn.maxContextBlocks !== undefined ? { maxContextBlocks: turn.maxContextBlocks } : {}),
    ...(turn.mode !== undefined ? { mode: turn.mode } : {}),
    ...(turn.runtimeHints !== undefined ? { runtimeHints: turn.runtimeHints } : {})
  };
  const compileResult = compileRuntimeContext(compileInput);
  const warnings: string[] = [];

  try {
    const normalizedEventExport = buildNormalizedRuntimeEventExport(turn, compileResult);
    const eventExport = writeRuntimeEventExportBundle(turn, normalizedEventExport);
    return {
      ...compileResult,
      eventExport,
      warnings
    };
  } catch (error) {
    if (options.failOpen === false) {
      throw error;
    }

    warnings.push(toErrorMessage(error));
    return {
      ...compileResult,
      eventExport: {
        ok: false,
        wroteBundle: false,
        error: toErrorMessage(error)
      },
      warnings
    };
  }
}
