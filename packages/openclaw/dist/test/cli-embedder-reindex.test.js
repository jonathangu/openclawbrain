import test from "node:test";
import assert from "node:assert/strict";
import { reindexMaterializationCandidateWithEmbedder } from "../src/materialization-embedder.js";

function makeMaterialization() {
    return {
        reason: "learn_cli",
        lane: "live",
        candidate: {
            summary: {
                packId: "pack-test"
            },
            payloads: {
                graph: {
                    packId: "pack-test",
                    blocks: [
                        {
                            id: "block-1",
                            source: "docs/guide.md",
                            text: "Ship the embedder reindex fix.",
                            keywords: ["embedder", "reindex"],
                            priority: 2,
                            learning: {
                                humanLabels: 0,
                                selfLabels: 0,
                                hebbianPulse: 0,
                                role: "default"
                            },
                            state: {
                                freshness: 1,
                                traversalBias: 0
                            },
                            edges: []
                        }
                    ]
                },
                vectors: {
                    packId: "pack-test",
                    entries: [
                        {
                            blockId: "block-1",
                            keywords: ["embedder", "reindex"],
                            boost: 1,
                            weights: {
                                embedder: 1,
                                reindex: 1
                            }
                        }
                    ]
                }
            },
            manifest: {
                payloadChecksums: {
                    graph: "graph-checksum",
                    vector: "vector-checksum"
                },
                modelFingerprints: []
            }
        }
    };
}

test("reindexMaterializationCandidateWithEmbedder leaves materialization untouched without an embedder", async () => {
    const materialization = makeMaterialization();
    const result = await reindexMaterializationCandidateWithEmbedder(materialization, null);
    assert.strictEqual(result, materialization);
    assert.equal(result.candidate.payloads.vectors.entries[0].embedding, undefined);
});

test("reindexMaterializationCandidateWithEmbedder enriches candidate vectors with numeric embeddings", async () => {
    const materialization = makeMaterialization();
    const result = await reindexMaterializationCandidateWithEmbedder(materialization, {
        model: "bge-large",
        embed: async (texts) => texts.map(() => ({
            model: "bge-large",
            values: [0.25, 0.5, 0.75]
        }))
    });
    assert.notStrictEqual(result, materialization);
    assert.deepEqual(result.candidate.payloads.vectors.entries[0].embedding, {
        model: "bge-large",
        values: [0.25, 0.5, 0.75]
    });
    assert.deepEqual(result.candidate.manifest.modelFingerprints, ["bge-large"]);
});
