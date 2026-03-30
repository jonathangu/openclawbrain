import { reindexCandidatePackBuildResultWithEmbedder } from "@openclawbrain/learner";

export async function reindexMaterializationCandidateWithEmbedder(materialization, embedder) {
    if (materialization === null || embedder === null) {
        return materialization;
    }
    return {
        ...materialization,
        candidate: await reindexCandidatePackBuildResultWithEmbedder(materialization.candidate, embedder)
    };
}
