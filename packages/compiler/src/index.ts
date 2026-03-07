import {
  CONTRACT_IDS,
  checksumJsonPayload,
  type ActivationPointerSlot,
  type ContextCompactionMode,
  type PackContextBlockRecordV1,
  type PackVectorEntryV1,
  type RouteMode,
  type RuntimeCompileExpectationV1,
  type RuntimeCompileRequestV1,
  type RuntimeCompileResponseV1,
  type RuntimeCompileTargetV1,
  type RuntimeContextBlockV1,
  validateRuntimeCompileExpectation,
  validateRuntimeCompileRequest,
  validateRuntimeCompileResponse,
  validateRuntimeCompileTargetExpectation
} from "@openclawbrain/contracts";
import { describePackCompileTarget, inspectActivationState, loadPack, loadPackFromActivation, type PackDescriptor } from "@openclawbrain/pack-format";

export type LoadedPack = PackDescriptor;

export interface RankedContextBlock {
  blockId: string;
  source: string;
  text: string;
  score: number;
  priority: number;
  matchedTokens: string[];
  tokenCount: number;
  compactedFrom?: string[];
  packOrder: number;
}

export interface ActivationCompileOptions {
  slot?: ActivationPointerSlot;
  requireActivationReady?: boolean;
  requirePromotionSafe?: boolean;
  expectedTarget?: RuntimeCompileExpectationV1;
}

export interface ActivationCompileResolution {
  slot: ActivationPointerSlot;
  pack: LoadedPack;
  target: RuntimeCompileTargetV1;
}

export interface ActivationCompileResult extends RuntimeCompileResponseV1 {
  slot: ActivationPointerSlot;
  target: RuntimeCompileTargetV1;
  response: RuntimeCompileResponseV1;
}

type LegacyActivationCompileOptions = ActivationCompileOptions & {
  expectation?: RuntimeCompileExpectationV1;
};

function normalizeTokens(value: string): string[] {
  return [...new Set(value.toLowerCase().split(/[^a-z0-9]+/u).filter((token) => token.length >= 2))];
}

function compareIsoDates(left: string, right: string): number {
  return Date.parse(left) - Date.parse(right);
}

function estimateTokenCount(value: string): number {
  return normalizeTokens(value).length;
}

function requestTokens(request: RuntimeCompileRequestV1): string[] {
  const hints = request.runtimeHints ?? [];
  return normalizeTokens([request.userMessage, ...hints].join(" "));
}

function buildKeywordWeights(block: PackContextBlockRecordV1, vectorEntry: PackVectorEntryV1 | undefined): Map<string, number> {
  const weights = new Map<string, number>();

  function assignWeight(keyword: string, weight: number): void {
    for (const token of normalizeTokens(keyword)) {
      weights.set(token, Math.max(weights.get(token) ?? 0, weight));
    }
  }

  for (const keyword of block.keywords) {
    assignWeight(keyword, 3);
  }

  if (vectorEntry !== undefined) {
    for (const keyword of vectorEntry.keywords) {
      assignWeight(keyword, 4);
    }
    for (const [keyword, weight] of Object.entries(vectorEntry.weights ?? {})) {
      const numericWeight = Number.isFinite(weight) ? weight : 0;
      assignWeight(keyword, numericWeight);
    }
  }

  return weights;
}

function blockTokenCount(block: Pick<RuntimeContextBlockV1, "text" | "tokenCount">): number {
  return block.tokenCount ?? estimateTokenCount(block.text);
}

function buildContextBlock(block: RuntimeContextBlockV1): RuntimeContextBlockV1 {
  return {
    id: block.id,
    source: block.source,
    text: block.text,
    tokenCount: blockTokenCount(block),
    ...(block.compactedFrom !== undefined ? { compactedFrom: [...block.compactedFrom] } : {})
  };
}

function flattenContextIds(block: RuntimeContextBlockV1): string[] {
  return block.compactedFrom ?? [block.id];
}

function flattenRankedContextIds(block: Pick<RankedContextBlock, "blockId" | "compactedFrom">): string[] {
  return block.compactedFrom ?? [block.blockId];
}

function buildSelectedContextBlock(entry: RankedContextBlock): RuntimeContextBlockV1 {
  return buildContextBlock({
    id: entry.blockId,
    source: entry.source,
    text: entry.text,
    tokenCount: entry.tokenCount,
    ...(entry.compactedFrom !== undefined ? { compactedFrom: entry.compactedFrom } : {})
  });
}

function selectContextBlocks(
  ranked: readonly RankedContextBlock[],
  maxBlocks: number
): { selected: RuntimeContextBlockV1[]; matchedSelectedCount: number; overlapPrunedCount: number } {
  if (maxBlocks === 0) {
    return {
      selected: [],
      matchedSelectedCount: 0,
      overlapPrunedCount: 0
    };
  }

  const selected: RuntimeContextBlockV1[] = [];
  const coveredIds = new Set<string>();
  let matchedSelectedCount = 0;
  let overlapPrunedCount = 0;

  const trySelect = (entry: RankedContextBlock, matchedTier: boolean): void => {
    if (selected.length >= maxBlocks) {
      return;
    }

    const coverageIds = flattenRankedContextIds(entry);
    if (coverageIds.some((coverageId) => coveredIds.has(coverageId))) {
      overlapPrunedCount += 1;
      return;
    }

    selected.push(buildSelectedContextBlock(entry));
    for (const coverageId of coverageIds) {
      coveredIds.add(coverageId);
    }

    if (matchedTier) {
      matchedSelectedCount += 1;
    }
  };

  const matched = ranked.filter((entry) => entry.matchedTokens.length > 0);
  const unmatched = ranked.filter((entry) => entry.matchedTokens.length === 0);

  for (const entry of matched) {
    trySelect(entry, true);
  }
  for (const entry of unmatched) {
    trySelect(entry, false);
  }

  return {
    selected,
    matchedSelectedCount,
    overlapPrunedCount
  };
}

function totalCharCount(blocks: readonly RuntimeContextBlockV1[]): number {
  return blocks.reduce((sum, block) => sum + block.text.length, 0);
}

function totalTokenCount(blocks: readonly RuntimeContextBlockV1[]): number {
  return blocks.reduce((sum, block) => sum + blockTokenCount(block), 0);
}

function mergeDiagnosticNotes(existing: readonly string[], additions: readonly (string | undefined)[]): string[] {
  const seen = new Set<string>();
  const merged: string[] = [];

  for (const note of [...existing, ...additions]) {
    if (note === undefined || note.length === 0 || seen.has(note)) {
      continue;
    }
    seen.add(note);
    merged.push(note);
  }

  return merged;
}

function truncateText(value: string, maxChars: number): string {
  if (maxChars <= 0) {
    return "";
  }
  if (value.length <= maxChars) {
    return value;
  }
  if (maxChars === 1) {
    return value.slice(0, 1);
  }
  return `${value.slice(0, maxChars - 1).trimEnd()}…`;
}

function buildCompactedText(blocks: readonly RuntimeContextBlockV1[], maxChars: number): string {
  const prefix = "Compacted pack-backed context: ";
  const separator = " | ";
  const available = Math.max(0, maxChars - prefix.length);
  const perBlockBudget = Math.max(20, Math.floor((available - separator.length * Math.max(0, blocks.length - 1)) / Math.max(1, blocks.length)));
  const parts = blocks.map((block) => `${block.id}: ${truncateText(block.text, perBlockBudget)}`);
  return truncateText(`${prefix}${parts.join(separator)}`, maxChars);
}

function mergeContextBlocks(blocks: readonly RuntimeContextBlockV1[], maxChars: number): RuntimeContextBlockV1 {
  const compactedFrom = [...new Set(blocks.flatMap((block) => flattenContextIds(block)))];
  const sources = [...new Set(blocks.map((block) => block.source))];

  return buildContextBlock({
    id: `compact:${compactedFrom.join("+")}`,
    source: sources.length === 1 ? `compact:${sources[0]}` : `compact:${sources.join("|")}`,
    text: buildCompactedText(blocks, maxChars),
    tokenCount: totalTokenCount(blocks),
    compactedFrom
  });
}

function fitContextToCharBudget(
  blocks: readonly RuntimeContextBlockV1[],
  maxChars: number | undefined,
  compactionMode: ContextCompactionMode
): { blocks: RuntimeContextBlockV1[]; compactionApplied: boolean } {
  const normalized = blocks.map((block) => buildContextBlock(block));

  if (maxChars === undefined || totalCharCount(normalized) <= maxChars) {
    return {
      blocks: normalized,
      compactionApplied: false
    };
  }

  if (maxChars <= 0 || normalized.length === 0) {
    return {
      blocks: [],
      compactionApplied: false
    };
  }

  if (compactionMode === "none") {
    const fitted: RuntimeContextBlockV1[] = [];
    let remaining = maxChars;
    let modified = false;

    for (const block of normalized) {
      if (remaining <= 0) {
        break;
      }
      if (block.text.length <= remaining) {
        fitted.push(block);
        remaining -= block.text.length;
        continue;
      }
      const truncated = buildContextBlock({
        ...block,
        text: truncateText(block.text, remaining)
      });
      if (truncated.text.length > 0) {
        fitted.push(truncated);
        modified = truncated.text !== block.text;
      }
      break;
    }

    return {
      blocks: fitted,
      compactionApplied: modified
    };
  }

  if (normalized.length === 1) {
    const block = normalized[0] as RuntimeContextBlockV1;
    const truncated = buildContextBlock({
      ...block,
      text: truncateText(block.text, maxChars)
    });
    return {
      blocks: truncated.text.length === 0 ? [] : [truncated],
      compactionApplied: truncated.text !== block.text
    };
  }

  const head = normalized[0];
  if (head === undefined) {
    return {
      blocks: [],
      compactionApplied: false
    };
  }

  const tail = normalized.slice(1);
  if (head.text.length < maxChars) {
    const tailBudget = maxChars - head.text.length;
    const compactedTail = mergeContextBlocks(tail, tailBudget);
    if (compactedTail.text.length > 0 && totalCharCount([head, compactedTail]) <= maxChars) {
      return {
        blocks: [head, compactedTail],
        compactionApplied: true
      };
    }
  }

  return {
    blocks: [mergeContextBlocks(normalized, maxChars)],
    compactionApplied: true
  };
}

export function determineRouteMode(pack: LoadedPack, requested: RouteMode): RouteMode {
  return pack.manifest.routePolicy === "requires_learned_routing" ? "learned" : requested;
}

function isStrictlyFresherTarget(candidate: RuntimeCompileTargetV1, active: RuntimeCompileTargetV1): boolean {
  return (
    compareIsoDates(candidate.builtAt, active.builtAt) > 0 ||
    candidate.eventRange.end > active.eventRange.end ||
    candidate.eventRange.count > active.eventRange.count
  );
}

function assertRequestPackExpectation(pack: LoadedPack, request: RuntimeCompileRequestV1): void {
  if (request.activePackId !== undefined && request.activePackId !== pack.manifest.packId) {
    throw new Error(`Compile request activePackId ${request.activePackId} does not match loaded pack ${pack.manifest.packId}`);
  }
}

export function loadPackForCompile(rootDir: string): LoadedPack {
  return loadPack(rootDir);
}

function resolveActivationCompileExpectation(options: ActivationCompileOptions): RuntimeCompileExpectationV1 | undefined {
  const legacyOptions = options as LegacyActivationCompileOptions;
  if (Object.prototype.hasOwnProperty.call(legacyOptions, "expectation")) {
    throw new Error("Activation compile options expectation has been removed; use expectedTarget");
  }

  return options.expectedTarget;
}

function assertActivationCompileSafety(rootDir: string, slot: ActivationPointerSlot, options: ActivationCompileOptions): void {
  if (slot !== "candidate" || options.requirePromotionSafe === false) {
    return;
  }

  const inspection = inspectActivationState(rootDir);
  if (!inspection.promotion.allowed) {
    throw new Error(`Candidate compile blocked: ${inspection.promotion.findings.join("; ")}`);
  }
}

function activationFreshnessNotes(rootDir: string, slot: ActivationPointerSlot, target: RuntimeCompileTargetV1): string[] {
  if (slot !== "active") {
    return [];
  }

  const inspection = inspectActivationState(rootDir);
  const candidate = inspection.candidate;
  if (candidate === null) {
    return [];
  }

  if (!inspection.promotion.allowed) {
    return [`candidate_rejected=${candidate.packId}:${inspection.promotion.findings.join(" | ")}`];
  }

  const candidateTarget: RuntimeCompileTargetV1 = {
    packId: candidate.packId,
    routePolicy: candidate.routePolicy,
    routerIdentity: candidate.routerIdentity,
    workspaceSnapshot: candidate.workspaceSnapshot,
    workspaceRevision: candidate.workspaceRevision,
    eventRange: {
      start: candidate.eventRange.start,
      end: candidate.eventRange.end,
      count: candidate.eventRange.count
    },
    eventExportDigest: candidate.eventExportDigest,
    builtAt: candidate.builtAt
  };

  if (!isStrictlyFresherTarget(candidateTarget, target)) {
    return [];
  }

  return [`stale_route_warning=active pack ${target.packId} is behind promotion-ready candidate ${candidate.packId}`];
}

export function resolveActivationCompileTarget(rootDir: string, options: ActivationCompileOptions = {}): ActivationCompileResolution {
  const slot = options.slot ?? "active";
  const pack = loadPackFromActivation(rootDir, slot, {
    requireActivationReady: options.requireActivationReady !== false
  });

  if (pack === null) {
    throw new Error(`Activation slot ${slot} is empty`);
  }

  const target = describePackCompileTarget(pack);
  const expectation = resolveActivationCompileExpectation(options);

  if (expectation !== undefined) {
    const expectationErrors = validateRuntimeCompileExpectation(expectation);
    if (expectationErrors.length > 0) {
      throw new Error(`Invalid compile expectation: ${expectationErrors.join("; ")}`);
    }

    const compatibilityErrors = validateRuntimeCompileTargetExpectation(target, expectation);
    if (compatibilityErrors.length > 0) {
      throw new Error(`Activation compile target mismatch: ${compatibilityErrors.join("; ")}`);
    }
  }

  assertActivationCompileSafety(rootDir, slot, options);

  return {
    slot,
    pack,
    target
  };
}

export function loadPackForActivationCompile(rootDir: string, options: ActivationCompileOptions = {}): LoadedPack {
  return resolveActivationCompileTarget(rootDir, options).pack;
}

export function rankContextBlocks(pack: LoadedPack, request: RuntimeCompileRequestV1): RankedContextBlock[] {
  const tokens = requestTokens(request);
  const vectorsByBlockId = new Map(pack.vectors.entries.map((entry) => [entry.blockId, entry] as const));

  return pack.graph.blocks
    .map((block, packOrder) => {
      const vectorEntry = vectorsByBlockId.get(block.id);
      const weights = buildKeywordWeights(block, vectorEntry);
      const textTokens = new Set(normalizeTokens(`${block.source} ${block.text}`));
      const matchedTokens: string[] = [];
      let score = block.priority;

      for (const token of tokens) {
        const weightedScore = weights.get(token);
        if (weightedScore !== undefined) {
          matchedTokens.push(token);
          score += weightedScore;
          continue;
        }
        if (textTokens.has(token)) {
          matchedTokens.push(token);
          score += 1;
        }
      }

      if (matchedTokens.length > 0 && vectorEntry !== undefined) {
        score += vectorEntry.boost;
      }

      return {
        blockId: block.id,
        source: block.source,
        text: block.text,
        matchedTokens,
        priority: block.priority,
        score,
        tokenCount: block.tokenCount ?? estimateTokenCount(block.text),
        ...(block.compactedFrom !== undefined ? { compactedFrom: [...block.compactedFrom] } : {}),
        packOrder
      } satisfies RankedContextBlock;
    })
    .sort((left, right) => {
      if (right.matchedTokens.length !== left.matchedTokens.length) {
        return right.matchedTokens.length - left.matchedTokens.length;
      }
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      if (right.priority !== left.priority) {
        return right.priority - left.priority;
      }
      return left.packOrder - right.packOrder;
    });
}

export function compileRuntime(packOrRoot: LoadedPack | string, request: RuntimeCompileRequestV1): RuntimeCompileResponseV1 {
  const requestErrors = validateRuntimeCompileRequest(request);
  if (requestErrors.length > 0) {
    throw new Error(`Invalid compile request: ${requestErrors.join("; ")}`);
  }

  const pack = typeof packOrRoot === "string" ? loadPackForCompile(packOrRoot) : packOrRoot;
  assertRequestPackExpectation(pack, request);
  const modeEffective = determineRouteMode(pack, request.modeRequested);
  const usedLearnedRouteFn = modeEffective === "learned";

  if (usedLearnedRouteFn && pack.router === null) {
    throw new Error("learned-routing pack cannot compile without a router artifact");
  }

  const ranked = rankContextBlocks(pack, request);
  const maxBlocks = Math.max(0, request.maxContextBlocks);
  const matched = ranked.filter((entry) => entry.matchedTokens.length > 0);
  const selection = selectContextBlocks(ranked, maxBlocks);
  const selected = selection.selected;
  const compactionMode = request.compactionMode ?? "native";
  const fitted = fitContextToCharBudget(selected, request.maxContextChars, compactionMode);
  const selectedContext = fitted.blocks;

  const notes: string[] = [];
  notes.push(`selected_context_ids=${selectedContext.map((block) => block.id).join(",")}`);
  notes.push(matched.length > 0 ? `selection_mode=token_match(${requestTokens(request).join(",")})` : "selection_mode=priority_fallback");
  notes.push(
    matched.length > 0 && selection.matchedSelectedCount < maxBlocks && selection.selected.length > selection.matchedSelectedCount
      ? "selection_tiers=token_match+priority_fallback"
      : matched.length > 0
        ? "selection_tiers=token_match_only"
        : "selection_tiers=priority_fallback_only"
  );
  notes.push("selection_strategy=pack_keyword_overlap_v1");
  if (modeEffective !== request.modeRequested) {
    notes.push(`learned_required_enforced=requested_${request.modeRequested}->${modeEffective}`);
  }
  if (selection.overlapPrunedCount > 0) {
    notes.push(`selection_overlap_pruned=${selection.overlapPrunedCount}`);
  }
  notes.push(`pack_graph_blocks=${pack.graph.blocks.length}`);
  if (request.maxContextChars !== undefined) {
    notes.push(`max_context_chars=${request.maxContextChars}`);
  }
  if (fitted.compactionApplied) {
    notes.push("native_structural_compaction=applied");
  }
  if (usedLearnedRouteFn && pack.router !== null) {
    notes.push(`router_strategy=${pack.router.strategy}`);
  }
  notes.push("OpenClaw remains the runtime owner.");

  const response: RuntimeCompileResponseV1 = {
    contract: CONTRACT_IDS.runtimeCompile,
    packId: pack.manifest.packId,
    selectedContext,
    diagnostics: {
      modeRequested: request.modeRequested,
      modeEffective,
      usedLearnedRouteFn,
      routerIdentity: pack.router?.routerIdentity ?? null,
      candidateCount: ranked.length,
      selectedCount: selectedContext.length,
      selectedCharCount: totalCharCount(selectedContext),
      selectedTokenCount: totalTokenCount(selectedContext),
      selectionStrategy: "pack_keyword_overlap_v1",
      selectionDigest: checksumJsonPayload({
        packId: pack.manifest.packId,
        selectedContext: selectedContext.map((block) => ({
          id: block.id,
          source: block.source,
          text: block.text,
          tokenCount: block.tokenCount ?? null,
          compactedFrom: block.compactedFrom ?? null
        }))
      }),
      compactionMode,
      compactionApplied: fitted.compactionApplied,
      notes
    }
  };

  const responseErrors = validateRuntimeCompileResponse(response);
  if (responseErrors.length > 0) {
    throw new Error(`Invalid compile response: ${responseErrors.join("; ")}`);
  }

  return response;
}

export function compileRuntimeFromActivation(
  rootDir: string,
  request: RuntimeCompileRequestV1,
  options: ActivationCompileOptions = {}
): ActivationCompileResult {
  const resolved = resolveActivationCompileTarget(rootDir, options);
  const compiledRequest =
    request.activePackId === undefined && resolved.slot === "active"
      ? {
          ...request,
          activePackId: resolved.target.packId
        }
      : request;
  const response = compileRuntime(resolved.pack, compiledRequest);
  const freshnessNotes = activationFreshnessNotes(rootDir, resolved.slot, resolved.target);
  const targetNotes = [
    `activation_slot=${resolved.slot}`,
    `target_pack_id=${resolved.target.packId}`,
    `target_route_policy=${resolved.target.routePolicy}`,
    `target_workspace_snapshot=${resolved.target.workspaceSnapshot}`,
    resolved.target.workspaceRevision === null ? undefined : `target_workspace_revision=${resolved.target.workspaceRevision}`,
    `target_event_range=${resolved.target.eventRange.start}-${resolved.target.eventRange.end}#${resolved.target.eventRange.count}`,
    resolved.target.eventExportDigest === null ? undefined : `target_event_export_digest=${resolved.target.eventExportDigest}`,
    `target_built_at=${resolved.target.builtAt}`,
    resolved.target.routerIdentity === null ? undefined : `target_router_identity=${resolved.target.routerIdentity}`
  ];
  const resolvedResponse = {
    ...response,
    diagnostics: {
      ...response.diagnostics,
      notes: mergeDiagnosticNotes(response.diagnostics.notes, [...freshnessNotes, ...targetNotes])
    }
  };

  const responseErrors = validateRuntimeCompileResponse(resolvedResponse);
  if (responseErrors.length > 0) {
    throw new Error(`Invalid compile response: ${responseErrors.join("; ")}`);
  }

  return {
    ...resolvedResponse,
    slot: resolved.slot,
    target: resolved.target,
    response: resolvedResponse
  };
}
