import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import type { MutationBundleRecord } from "../../src/brain-core/types.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";

const tempDirs: string[] = [];

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "brain-store-journal-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);
  return new BrainStore(db);
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("BrainStore learning journal", () => {
  it("round-trips bundle persistence and reconstructs mutation stories chronologically", () => {
    const store = setup();
    const bundle: MutationBundleRecord = {
      id: "mb_story_1",
      mutationIds: ["bm_story_1", "bm_story_2", "bm_story_3"],
      bundleSize: 3,
      status: "evaluating",
      baseScore: null,
      candidateScore: null,
      expectedGain: 0.45,
      rejectionReason: null,
      verdict: null,
      createdAt: 100,
      resolvedAt: null,
    };

    store.insertMutationBundle(bundle);
    store.updateMutationBundle(bundle.id, {
      status: "rejected",
      baseScore: 0.2,
      candidateScore: 0.18,
      rejectionReason: "candidate regressed",
      resolvedAt: 400,
    });

    store.appendLearningJournal({
      eventType: "mutation_proposed",
      mutationId: "bm_story_1",
      mutationIds: ["bm_story_1"],
      packVersion: 7,
      payload: {
        mutationKind: "connect",
        expectedGain: 0.2,
        proposal: { nodeA: "a", nodeB: "b", coFireCount: 4 },
        evidence: { episodeCount: 4 },
      },
      createdAt: 100,
    });
    store.appendLearningJournal({
      eventType: "bundle_evaluation_started",
      bundleId: bundle.id,
      mutationIds: bundle.mutationIds,
      packVersion: 7,
      payload: {
        mutationKinds: ["connect", "connect", "connect"],
        bundleSize: 3,
        expectedGain: bundle.expectedGain,
        candidateMutationCount: 5,
        recentEpisodeIds: ["ep_a", "ep_b"],
        config: {
          minBundleSize: 3,
          maxBundleSize: 10,
          minRewardThreshold: 0.3,
          maxContextInflation: 1.5,
          minImprovementRatio: 1.05,
        },
      },
      createdAt: 200,
    });
    store.appendLearningJournal({
      eventType: "bundle_evaluation_completed",
      bundleId: bundle.id,
      mutationIds: bundle.mutationIds,
      packVersion: 7,
      payload: {
        mutationKinds: ["connect", "connect", "connect"],
        bundleSize: 3,
        expectedGain: bundle.expectedGain,
        qualifyingEpisodeIds: ["ep_a"],
        baseScore: 0.2,
        candidateScore: 0.18,
        shouldPromote: false,
        rejectionReason: "candidate regressed",
      },
      createdAt: 300,
    });
    store.appendLearningJournal({
      eventType: "promotion_rejected",
      bundleId: bundle.id,
      mutationIds: bundle.mutationIds,
      packVersion: 7,
      payload: {
        gate: "bundle_evaluation",
        mutationKinds: ["connect", "connect", "connect"],
        mutationCount: 3,
        reason: "candidate regressed",
        baseScore: 0.2,
        candidateScore: 0.18,
        metadata: { reviewer: "worker" },
      },
      createdAt: 400,
    });

    const persistedBundle = store.getMutationBundle(bundle.id);
    expect(persistedBundle).not.toBeNull();
    expect(persistedBundle?.status).toBe("rejected");
    expect(persistedBundle?.candidateScore).toBe(0.18);
    expect(persistedBundle?.rejectionReason).toBe("candidate regressed");

    const story = store.getLearningJournal({ mutationId: "bm_story_1" });
    expect(story.map((record) => record.eventType)).toEqual([
      "mutation_proposed",
      "bundle_evaluation_started",
      "bundle_evaluation_completed",
      "promotion_rejected",
    ]);
    expect(story[0]?.mutationId).toBe("bm_story_1");
    if (story[1]?.eventType !== "bundle_evaluation_started") {
      throw new Error("expected bundle evaluation start record");
    }
    expect(story[1].payload.config.minBundleSize).toBe(3);

    const recentBundleEvents = store.getLearningJournal({ bundleId: bundle.id, limit: 2 });
    expect(recentBundleEvents.map((record) => record.eventType)).toEqual([
      "bundle_evaluation_completed",
      "promotion_rejected",
    ]);
  });
});
