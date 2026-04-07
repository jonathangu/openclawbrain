export function buildArtifactProvenance(input = {}) {
  return {
    contract: "openclawbrain_artifact_provenance.v1",
    ...input,
  };
}
