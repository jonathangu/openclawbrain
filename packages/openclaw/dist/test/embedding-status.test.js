import test from "node:test";
import assert from "node:assert/strict";
import { summarizePackVectorEmbeddingState } from "../src/embedding-status.js";

test("embedding status counts direct embedding objects with models", () => {
    const summary = summarizePackVectorEmbeddingState({
        entries: [
            {
                embedding: {
                    model: "bge-large:latest",
                    values: [0.1, 0.2, 0.3],
                },
            },
            {
                embedding: {
                    model: "bge-large:latest",
                    values: [0.4, 0.5, 0.6],
                },
            },
        ],
    });
    assert.deepEqual(summary, {
        vectorEntryCount: 2,
        numericEmbeddingEntryCount: 2,
        embeddingModels: ["bge-large:latest"],
    });
});

test("embedding status recognizes numeric vectors surfaced in alternate pack shapes", () => {
    const summary = summarizePackVectorEmbeddingState({
        entries: [
            {
                embeddingModel: "bge-large:latest",
                vector: Float32Array.from([0.1, 0.2]),
            },
            {
                model: "nomic-embed-text",
                numericEmbeddingValues: [0.3, 0.4],
            },
            {
                embedding: null,
            },
        ],
    });
    assert.deepEqual(summary, {
        vectorEntryCount: 3,
        numericEmbeddingEntryCount: 2,
        embeddingModels: ["bge-large:latest", "nomic-embed-text"],
    });
});

test("embedding status ignores empty or non-numeric payloads", () => {
    const summary = summarizePackVectorEmbeddingState({
        entries: [
            { embedding: { model: "bad", values: [] } },
            { embedding: { model: "bad", values: [1, Number.NaN] } },
            { values: ["1", "2"] },
        ],
    });
    assert.deepEqual(summary, {
        vectorEntryCount: 3,
        numericEmbeddingEntryCount: 0,
        embeddingModels: [],
    });
});

