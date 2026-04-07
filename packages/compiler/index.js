import { createHash } from "node:crypto";
import { loadPackFromActivation } from "../pack-format/index.js";

export const DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text";

function tokenize(value) {
  return String(value ?? "")
    .toLowerCase()
    .split(/[^a-z0-9]+/u)
    .map((token) => token.trim())
    .filter((token) => token.length > 0);
}

function unique(values) {
  return [...new Set(values)];
}

function blockText(block) {
  return String(block?.text ?? block?.content ?? block?.summary ?? "");
}

function scoreBlock(block, request) {
  const queryTokens = unique([
    ...tokenize(request?.userMessage),
    ...((Array.isArray(request?.runtimeHints) ? request.runtimeHints : []).flatMap((hint) => tokenize(hint))),
  ]);
  const haystack = tokenize(blockText(block));
  let overlap = 0;
  for (const token of queryTokens) {
    if (haystack.includes(token)) {
      overlap += 1;
    }
  }
  return overlap + (block?.scoreHint ?? 0) + (block?.priority ?? 0);
}

function resolveBlocks(pack) {
  const graph = pack?.graph ?? pack?.payloads?.graph ?? null;
  if (Array.isArray(graph?.blocks)) {
    return graph.blocks;
  }
  if (Array.isArray(graph?.nodes)) {
    return graph.nodes.map((node) => ({
      id: node.id ?? node.blockId ?? `node-${Math.random().toString(36).slice(2, 8)}`,
      text: node.text ?? node.summary ?? node.content ?? "",
      edges: node.edges ?? [],
    }));
  }
  return [];
}

function buildDiagnostics(pack, request, selectedContext, options) {
  const router = pack?.router ?? pack?.payloads?.router ?? null;
  const modeEffective = request?.modeRequested === "learned" && router ? "learned" : request?.modeRequested ?? "heuristic";
  const selectionDigest = createHash("sha256")
    .update(JSON.stringify(selectedContext.map((block) => block.id)))
    .digest("hex");
  return {
    modeEffective,
    routerIdentity: router?.routerIdentity ?? null,
    usedLearnedRouteFn: modeEffective === "learned" && router !== null,
    servedArtifact: router !== null ? "router" : "graph",
    selectionDigest: `sha256:${selectionDigest}`,
    selectedCount: selectedContext.length,
    selectedCharCount: selectedContext.reduce((sum, block) => sum + String(block.text ?? "").length, 0),
    selectedTokenCount: selectedContext.reduce((sum, block) => sum + tokenize(block.text).length, 0),
    notes: [
      `selection_mode=${options?.selectionMode ?? "flat_rank_v1"}`,
      `selected_context_count=${selectedContext.length}`,
    ],
  };
}

export function rankContextBlocks(pack, request = {}) {
  return resolveBlocks(pack)
    .map((block) => ({
      blockId: block.id ?? block.blockId ?? `block-${Math.random().toString(36).slice(2, 8)}`,
      score: scoreBlock(block, request),
    }))
    .sort((left, right) => right.score - left.score || left.blockId.localeCompare(right.blockId));
}

export function compileRuntime(pack, request = {}, options = {}) {
  const blocks = resolveBlocks(pack);
  const ranked = blocks
    .map((block) => ({ block, score: scoreBlock(block, request) }))
    .sort((left, right) => right.score - left.score || String(left.block?.id ?? "").localeCompare(String(right.block?.id ?? "")));
  const selectedContext = ranked
    .slice(0, Math.max(1, request?.maxContextBlocks ?? 4))
    .map(({ block }) => ({
      id: block.id ?? block.blockId ?? "block",
      text: blockText(block),
    }));
  const router = pack?.router ?? pack?.payloads?.router ?? null;
  const stopLocalActive = Array.isArray(router?.policyUpdates)
    && router.policyUpdates.some((update) => String(update?.blockId ?? "").toLowerCase().includes("stop"));
  return {
    target: {
      packId: pack?.manifest?.packId ?? pack?.manifest?.pack_id ?? request?.activePackId ?? "pack-stub",
    },
    response: {
      packId: pack?.manifest?.packId ?? pack?.manifest?.pack_id ?? request?.activePackId ?? "pack-stub",
      selectedContext,
      diagnostics: buildDiagnostics(pack, request, selectedContext, options),
      structuralSignals: {
        graphWalkHopCount: options?.selectionMode === "graph_walk_v1" && stopLocalActive ? 0 : Math.max(0, selectedContext.length - 1),
      },
    },
  };
}

export function compileRuntimeFromActivation(activationRoot, request = {}, options = {}) {
  const pack = loadPackFromActivation(activationRoot, "active") ?? {
    manifest: { packId: request?.activePackId ?? "pack-stub" },
    graph: { blocks: [] },
    router: null,
  };
  return compileRuntime(pack, request, options);
}

export function createOllamaEmbedder(options = {}) {
  const model = options.model ?? DEFAULT_OLLAMA_EMBEDDING_MODEL;
  const embedOne = (text) => {
    const digest = createHash("sha256").update(String(text ?? "")).digest();
    return Array.from(digest.subarray(0, 8)).map((value) => value / 255);
  };
  return {
    provider: "ollama",
    model,
    async embed(text) {
      return embedOne(text);
    },
    async embedBatch(texts) {
      return (Array.isArray(texts) ? texts : []).map((text) => embedOne(text));
    },
  };
}

export function describeCompileFallbackUsage() {
  return {
    usedFallback: false,
    detail: "compiler shim uses deterministic local ranking",
  };
}
