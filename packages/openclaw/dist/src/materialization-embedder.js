import { createHash } from "node:crypto";

function stableJson(value) {
    if (value === null || typeof value !== "object") {
        return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
        return `[${value.map((entry) => entry === undefined ? "null" : stableJson(entry)).join(",")}]`;
    }
    return `{${Object.entries(value)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`)
        .join(",")}}`;
}

function computePayloadChecksum(value) {
    return `sha256-${createHash("sha256").update(stableJson(value)).digest("hex")}`;
}

function embeddingTextForBlock(block) {
    const keywords = Array.isArray(block?.keywords) ? block.keywords.filter((keyword) => typeof keyword === "string") : [];
    return [block?.source, block?.text, ...keywords]
        .filter((candidate) => typeof candidate === "string" && candidate.length > 0)
        .join("\n\n");
}

function isFiniteEmbeddingResult(value) {
    return value !== undefined &&
        typeof value.model === "string" &&
        value.model.length > 0 &&
        Array.isArray(value.values) &&
        value.values.length > 0 &&
        value.values.every((candidate) => Number.isFinite(candidate));
}

async function collectEmbeddingsWithRetry(targets, embedder, embeddingsByBlockId) {
    if (targets.length === 0) {
        return;
    }
    try {
        const embeddings = await embedder.embed(targets.map((target) => target.text));
        if (embeddings.length !== targets.length) {
            throw new Error("embedding batch length mismatch");
        }
        for (const [index, target] of targets.entries()) {
            const embedding = embeddings[index];
            if (!isFiniteEmbeddingResult(embedding)) {
                continue;
            }
            embeddingsByBlockId.set(target.blockId, {
                model: embedding.model,
                values: [...embedding.values]
            });
        }
        return;
    }
    catch {
        if (targets.length === 1) {
            return;
        }
    }
    const midpoint = Math.ceil(targets.length / 2);
    await collectEmbeddingsWithRetry(targets.slice(0, midpoint), embedder, embeddingsByBlockId);
    await collectEmbeddingsWithRetry(targets.slice(midpoint), embedder, embeddingsByBlockId);
}

function needsEmbeddingReindex(vectors, embedder) {
    if (!vectors || !Array.isArray(vectors.entries) || vectors.entries.length === 0) {
        return false;
    }
    return vectors.entries.some((entry) => entry.embedding === undefined || entry.embedding?.model !== embedder.model);
}

function embeddingModelFingerprints(vectors) {
    if (!vectors || !Array.isArray(vectors.entries)) {
        return [];
    }
    return [...new Set(vectors.entries.flatMap((entry) => typeof entry.embedding?.model === "string" ? [entry.embedding.model] : []))];
}

export async function reindexCandidatePackBuildResultWithEmbedder(result, embedder) {
    if (!result || !embedder || !needsEmbeddingReindex(result.payloads?.vectors, embedder)) {
        return result;
    }
    const graphBlocks = Array.isArray(result.payloads?.graph?.blocks) ? result.payloads.graph.blocks : [];
    const embeddingsByBlockId = new Map();
    try {
        await collectEmbeddingsWithRetry(graphBlocks.map((block) => ({
            blockId: block.id,
            text: embeddingTextForBlock(block)
        })), embedder, embeddingsByBlockId);
    }
    catch {
        return result;
    }
    if (embeddingsByBlockId.size === 0) {
        return result;
    }
    const vectors = {
        ...result.payloads.vectors,
        entries: result.payloads.vectors.entries.map((entry) => {
            const embedding = embeddingsByBlockId.get(entry.blockId);
            return embedding === undefined
                ? entry
                : {
                    ...entry,
                    embedding: {
                        model: embedding.model,
                        values: [...embedding.values]
                    }
                };
        })
    };
    const manifest = result.manifest ?? {};
    const payloadChecksums = manifest.payloadChecksums ?? {};
    const modelFingerprints = Array.isArray(manifest.modelFingerprints)
        ? manifest.modelFingerprints.filter((model) => typeof model === "string")
        : [];
    return {
        ...result,
        payloads: {
            ...result.payloads,
            vectors
        },
        manifest: {
            ...manifest,
            payloadChecksums: {
                ...payloadChecksums,
                vector: computePayloadChecksum(vectors)
            },
            modelFingerprints: [...new Set([...modelFingerprints, ...embeddingModelFingerprints(vectors)])]
        }
    };
}

export async function reindexMaterializationCandidateWithEmbedder(materialization, embedder) {
    if (materialization === null || embedder === null) {
        return materialization;
    }
    return {
        ...materialization,
        candidate: await reindexCandidatePackBuildResultWithEmbedder(materialization.candidate, embedder)
    };
}
