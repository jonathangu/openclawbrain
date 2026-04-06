import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  readAndValidateColdStartSourceIntakeRegistryV1,
  summarizeColdStartSourceIntakeRegistryV1,
  validateColdStartSourceIntakeRegistryV1,
} from "../../src/brain-core/cold-start-source-intake.js";

const registryPath = fileURLToPath(new URL("../../data/cold-start/registry.bootstrap.json", import.meta.url));
const registryJson = JSON.parse(readFileSync(registryPath, "utf8"));

describe("cold-start source intake bootstrap", () => {
  it("validates the governed intake registry and exposes a useful summary", () => {
    const validation = validateColdStartSourceIntakeRegistryV1(registryJson);
    expect(validation.valid).toBe(true);
    expect(validation.issues).toEqual([]);

    const { registry, validation: loadedValidation, summary } = readAndValidateColdStartSourceIntakeRegistryV1(
      registryPath,
    );
    expect(loadedValidation.valid).toBe(true);
    expect(summary).not.toBeNull();

    expect(registry.registry_id).toBe("cold-start-source-intake-bootstrap-v1");
    expect(registry.cards).toHaveLength(7);
    expect(registry.intake_order).toEqual([
      "hotpotqa_v1",
      "2wikimultihopqa_v1",
      "musique_v1",
      "toolmind_v1",
      "memgpt_traces_v1",
      "swe_bench_v1",
      "repo_bench_v1",
    ]);

    expect(summary).toMatchObject({
      registryId: "cold-start-source-intake-bootstrap-v1",
      owner: "guclaw",
      entryCount: 7,
      materializedSnapshotCount: 5,
      placeholderCount: 7,
      datasetIds: [
        "hotpotqa_v1",
        "2wikimultihopqa_v1",
        "musique_v1",
        "toolmind_v1",
        "memgpt_traces_v1",
        "swe_bench_v1",
        "repo_bench_v1",
      ],
      sourceFamilyCounts: {
        qa: 3,
        tools: 1,
        repo: 2,
        agent_traces: 1,
        memory: 0,
        docs: 0,
      },
      approvalStatusCounts: {
        proposed: 2,
        under_review: 2,
        approved_train: 3,
        approved_eval_only: 0,
        rejected: 0,
        archived: 0,
      },
      piiRiskCounts: {
        none: 3,
        low: 3,
        medium: 1,
        high: 0,
        unknown: 0,
      },
    });

    expect(summarizeColdStartSourceIntakeRegistryV1(registry)).toEqual(summary);
  });
});
