import {
  CONTRACT_IDS,
  type PackContextBlockRecordV1,
  type PackVectorEntryV1,
  type RouteMode,
  type RuntimeCompileRequestV1,
  type RuntimeCompileResponseV1,
  validateRuntimeCompileRequest,
  validateRuntimeCompileResponse
} from "@openclawbrain/contracts";
import { loadPack, type PackDescriptor } from "@openclawbrain/pack-format";

export type LoadedPack = PackDescriptor;

export interface RankedContextBlock {
  blockId: string;
  score: number;
  priority: number;
  matchedTokens: string[];
}

function normalizeTokens(value: string): string[] {
  return [...new Set(value.toLowerCase().split(/[^a-z0-9]+/u).filter((token) => token.length >= 2))];
}

function requestTokens(request: RuntimeCompileRequestV1): string[] {
  const hints = request.runtimeHints ?? [];
  return normalizeTokens([request.userMessage, ...hints].join(" "));
}

function buildKeywordWeights(block: PackContextBlockRecordV1, vectorEntry: PackVectorEntryV1 | undefined): Map<string, number> {
  const weights = new Map<string, number>();

  for (const keyword of block.keywords) {
    weights.set(keyword.toLowerCase(), Math.max(weights.get(keyword.toLowerCase()) ?? 0, 3));
  }

  if (vectorEntry !== undefined) {
    for (const keyword of vectorEntry.keywords) {
      weights.set(keyword.toLowerCase(), Math.max(weights.get(keyword.toLowerCase()) ?? 0, 4));
    }
    for (const [keyword, weight] of Object.entries(vectorEntry.weights ?? {})) {
      const numericWeight = Number.isFinite(weight) ? weight : 0;
      weights.set(keyword.toLowerCase(), Math.max(weights.get(keyword.toLowerCase()) ?? 0, numericWeight));
    }
  }

  return weights;
}

export function determineRouteMode(pack: LoadedPack, requested: RouteMode): RouteMode {
  return pack.manifest.routePolicy === "requires_learned_routing" ? "learned" : requested;
}

export function loadPackForCompile(rootDir: string): LoadedPack {
  return loadPack(rootDir);
}

export function rankContextBlocks(pack: LoadedPack, request: RuntimeCompileRequestV1): RankedContextBlock[] {
  const tokens = requestTokens(request);
  const vectorsByBlockId = new Map(pack.vectors.entries.map((entry) => [entry.blockId, entry] as const));

  return pack.graph.blocks
    .map((block) => {
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
        matchedTokens,
        priority: block.priority,
        score
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
      return left.blockId.localeCompare(right.blockId);
    });
}

export function compileRuntime(packOrRoot: LoadedPack | string, request: RuntimeCompileRequestV1): RuntimeCompileResponseV1 {
  const requestErrors = validateRuntimeCompileRequest(request);
  if (requestErrors.length > 0) {
    throw new Error(`Invalid compile request: ${requestErrors.join("; ")}`);
  }

  const pack = typeof packOrRoot === "string" ? loadPackForCompile(packOrRoot) : packOrRoot;
  const modeEffective = determineRouteMode(pack, request.modeRequested);
  const usedLearnedRouteFn = modeEffective === "learned";

  if (usedLearnedRouteFn && pack.router === null) {
    throw new Error("learned-routing pack cannot compile without a router artifact");
  }

  const ranked = rankContextBlocks(pack, request);
  const maxBlocks = Math.max(0, request.maxContextBlocks);
  const matched = ranked.filter((entry) => entry.matchedTokens.length > 0);
  const chosen = maxBlocks === 0 ? [] : (matched.length > 0 ? matched : ranked).slice(0, maxBlocks);
  const blocksById = new Map(pack.graph.blocks.map((block) => [block.id, block] as const));

  const selectedContext = chosen.map((entry) => {
    const block = blocksById.get(entry.blockId);
    if (block === undefined) {
      throw new Error(`ranked block missing from graph: ${entry.blockId}`);
    }
    return {
      id: block.id,
      source: block.source,
      text: block.text
    };
  });

  const notes: string[] = [];
  notes.push(`selected_context_ids=${selectedContext.map((block) => block.id).join(",")}`);
  notes.push(matched.length > 0 ? `selection_mode=token_match(${requestTokens(request).join(",")})` : "selection_mode=priority_fallback");
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
      routerIdentity: usedLearnedRouteFn ? (pack.router?.routerIdentity ?? null) : null,
      notes
    }
  };

  const responseErrors = validateRuntimeCompileResponse(response);
  if (responseErrors.length > 0) {
    throw new Error(`Invalid compile response: ${responseErrors.join("; ")}`);
  }

  return response;
}
