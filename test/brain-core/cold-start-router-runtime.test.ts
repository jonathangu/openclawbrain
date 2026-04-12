import { describe, expect, it } from "vitest";

import { summarizeColdStartRouterArtifactManifestRuntimeTruthV1 } from "../../src/brain-core/cold-start-router-runtime.js";

describe("cold-start router runtime truth", () => {
  it("surfaces mixed-pack lineage from the manifest", () => {
    const runtimeTruth = summarizeColdStartRouterArtifactManifestRuntimeTruthV1({
      schema_version: 1,
      artifact_id: "router-artifact-mixed-001",
      artifact_version: "0.1.0",
      pack_type: "mixed",
      base_model_ref: "base-model.json#sha256:base001",
      weights_ref: "weights.json#sha256:w001",
      calibration_ref: "calibration.json#sha256:c001",
      feature_normalizers_ref: "feature-normalizers.json#sha256:n001",
      source_priors_ref: "source-priors.json#sha256:p001",
      safety_rules_ref: "safety-rules.json#sha256:s001",
      compatible_runtime_version: "openclawbrain-runtime@0.4.43",
      training_data_refs: ["dataset_hotpotqa_v1"],
      replay_gate_refs: ["replay:gate:001"],
      prior_base_artifact_id: "router-artifact-base-000",
      prior_base_artifact_checksum: "sha256:router-artifact-base-000",
      artifact_checksum: "sha256:router-artifact-mixed-001",
      created_at: "2026-04-05T16:08:00Z",
      router_identity: "router:gen1:mixed",
    });

    expect(runtimeTruth).toMatchObject({
      artifactId: "router-artifact-mixed-001",
      artifactVersion: "0.1.0",
      artifactChecksum: "sha256:router-artifact-mixed-001",
      packType: "mixed",
      routerIdentity: "router:gen1:mixed",
      priorBaseArtifactId: "router-artifact-base-000",
      priorBaseArtifactChecksum: "sha256:router-artifact-base-000",
      mixedPackFromBaseArtifactId: "router-artifact-base-000",
    });
    expect(runtimeTruth.summary).toContain("pack=mixed");
    expect(runtimeTruth.summary).toContain("mixedFrom=router-artifact-base-000");
  });
});
