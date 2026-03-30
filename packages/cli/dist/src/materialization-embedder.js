import { reindexCandidatePackBuildResultWithEmbedder } from "./local-learner.js";

export async function reindexMaterializationCandidateWithEmbedder(materialization, embedder) {
    if (materialization === null || embedder === null) {
        return materialization;
    }
    return {
        ...materialization,
        candidate: await reindexCandidatePackBuildResultWithEmbedder(materialization.candidate, embedder)
    };
}
