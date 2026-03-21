function isNumericArray(value) {
    return Array.isArray(value) &&
        value.length > 0 &&
        value.every((entry) => typeof entry === "number" && Number.isFinite(entry));
}

function isNumericTypedArray(value) {
    return ArrayBuffer.isView(value) &&
        !(value instanceof DataView) &&
        typeof value.length === "number" &&
        value.length > 0 &&
        Array.from(value).every((entry) => typeof entry === "number" && Number.isFinite(entry));
}

function hasNumericValues(value) {
    return isNumericArray(value) || isNumericTypedArray(value);
}

function normalizeOptionalString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}

function extractNumericEmbeddingShape(entry) {
    const embedding = entry?.embedding;
    const candidates = [
        embedding,
        embedding?.values,
        embedding?.vector,
        embedding?.embedding,
        entry?.values,
        entry?.vector,
        entry?.numericEmbedding,
        entry?.numericEmbeddingValues,
    ];
    return candidates.find((candidate) => hasNumericValues(candidate)) ?? null;
}

function extractEmbeddingModel(entry) {
    const candidates = [
        entry?.embedding?.model,
        entry?.embeddingModel,
        entry?.model,
    ];
    for (const candidate of candidates) {
        const normalized = normalizeOptionalString(candidate);
        if (normalized !== null) {
            return normalized;
        }
    }
    return null;
}

export function summarizePackVectorEmbeddingState(vectors) {
    if (!vectors || !Array.isArray(vectors.entries)) {
        return {
            vectorEntryCount: null,
            numericEmbeddingEntryCount: null,
            embeddingModels: []
        };
    }
    const embeddingModels = [...new Set(vectors.entries
            .flatMap((entry) => {
            if (extractNumericEmbeddingShape(entry) === null) {
                return [];
            }
            const model = extractEmbeddingModel(entry);
            return model === null ? [] : [model];
        }))].sort((left, right) => left.localeCompare(right));
    return {
        vectorEntryCount: vectors.entries.length,
        numericEmbeddingEntryCount: vectors.entries.filter((entry) => extractNumericEmbeddingShape(entry) !== null).length,
        embeddingModels
    };
}
