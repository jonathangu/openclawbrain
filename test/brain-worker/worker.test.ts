import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { DEFAULT_BRAIN_CONFIG } from "../../src/brain-core/types.js";
import type { Episode } from "../../src/brain-core/types.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { BrainWorker } from "../../src/brain-worker/worker.js";

const tempDirs: string[] = [];

function makeEpisode(params: {
  id: string;
  conversationId: number;
  reward?: number | null;
  rewardSource?: Episode["rewardSource"];
}): Episode {
  return {
    id: params.id,
    conversationId: params.conversationId,
    queryText: "test query",
    queryEmbedding: null,
    trajectory: [],
    firedNodes: [],
    vetoedNodes: [],
    contextChars: 0,
    reward: params.reward ?? null,
    rewardSource: params.rewardSource ?? null,
    packVersion: 1,
    createdAt: Date.now(),
  };
}

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "brain-worker-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);

  const store = new BrainStore(db);
  const graph = new BrainGraph();
  const worker = new BrainWorker(
    store,
    graph,
    null,
    {
      proposeMutations: vi.fn(() => []),
      applyToCandidateGraph: vi.fn(),
      applyMutation: vi.fn(),
    } as never,
    {
      replayGate: vi.fn(() => ({ passed: true, reason: null })),
    } as never,
    {
      ...DEFAULT_BRAIN_CONFIG,
      mutationsEnabled: false,
    },
    {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    },
    {
      onPromotionReady: vi.fn(async () => undefined),
    },
  );

  return { store, worker };
}

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("BrainWorker evidence resolution", () => {
  it("keeps only the highest-trust pending evidence per episode in a worker cycle", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_1", conversationId: 7 }));
    store.insertEvidence({
      episodeId: "ep_1",
      conversationId: 7,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.55,
      reason: "scanner pattern",
    });
    store.insertEvidence({
      episodeId: "ep_1",
      conversationId: 7,
      source: "human",
      kind: "human_feedback",
      value: 0.8,
      confidence: 0.9,
      reason: "user confirmed",
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("human");
    expect(pendingLabels[0]?.value).toBe(0.8);

    const resolved = store.getResolvedLabelsForEpisode("ep_1", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.source === "human")?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.source === "scanner")?.resolution).toBe("discarded_lower_trust");

    await (worker as any).processLabels();
    const episode = store.getEpisode("ep_1");
    expect(episode?.reward).toBe(0.8);
    expect(episode?.rewardSource).toBe("human");
  });

  it("collapses same-trust pending evidence to one promoted label using confidence", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_2", conversationId: 8 }));
    store.insertEvidence({
      episodeId: "ep_2",
      conversationId: 8,
      source: "self",
      kind: "self_result",
      value: -0.5,
      confidence: 0.4,
      reason: "weak failure signal",
    });
    store.insertEvidence({
      episodeId: "ep_2",
      conversationId: 8,
      source: "self",
      kind: "self_result",
      value: 0.5,
      confidence: 0.9,
      reason: "strong success signal",
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("self");
    expect(pendingLabels[0]?.value).toBe(0.5);
    expect(pendingLabels[0]?.confidence).toBe(0.9);

    const resolved = store.getResolvedLabelsForEpisode("ep_2", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.value === 0.5)?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.value === -0.5)?.resolution).toBe("discarded_duplicate");
  });

  it("prefers structured scanner evidence over heuristic scanner evidence when scanner signals conflict", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_2b", conversationId: 82 }));
    store.insertEvidence({
      episodeId: "ep_2b",
      conversationId: 82,
      source: "scanner",
      kind: "scanner_signal",
      value: -0.25,
      confidence: 0.95,
      reason: "heuristic scanner signal",
      metadata: { extractor: "scanner_heuristic" },
    });
    store.insertEvidence({
      episodeId: "ep_2b",
      conversationId: 82,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.6,
      reason: "structured guidance signal",
      metadata: { extractor: "structured_guidance_parts" },
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("scanner");
    expect(pendingLabels[0]?.value).toBe(0.25);
    expect(pendingLabels[0]?.confidence).toBe(0.6);

    const resolved = store.getResolvedLabelsForEpisode("ep_2b", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.value === 0.25)?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.value === -0.25)?.resolution).toBe("discarded_duplicate");
    expect(resolved.find((entry) => entry.value === -0.25)?.note).toContain("more-structured scanner evidence");
  });

  it("keeps the higher-confidence scanner label when same-value scanner evidence is only corroborating", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_2c", conversationId: 83 }));
    store.insertEvidence({
      episodeId: "ep_2c",
      conversationId: 83,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.95,
      reason: "high-confidence heuristic scanner signal",
      metadata: { extractor: "scanner_heuristic" },
    });
    store.insertEvidence({
      episodeId: "ep_2c",
      conversationId: 83,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.6,
      reason: "structured corroborating scanner signal",
      metadata: { extractor: "structured_guidance_parts" },
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("scanner");
    expect(pendingLabels[0]?.value).toBe(0.25);
    expect(pendingLabels[0]?.confidence).toBe(0.95);

    const resolved = store.getResolvedLabelsForEpisode("ep_2c", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.confidence === 0.95)?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.confidence === 0.6)?.resolution).toBe("discarded_duplicate");
    expect(resolved.find((entry) => entry.confidence === 0.6)?.note).toContain("matching scanner evidence already queued");
  });

  it("does not auto-override an existing equal-trust reward", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({
      id: "ep_3",
      conversationId: 9,
      reward: 0.8,
      rewardSource: "human",
    }));
    store.insertEvidence({
      episodeId: "ep_3",
      conversationId: 9,
      source: "human",
      kind: "human_feedback",
      value: -0.8,
      confidence: 0.95,
      reason: "later conflicting human signal",
    });

    await (worker as any).processEvidence();

    expect(store.getPendingLabels()).toHaveLength(0);
    const resolved = store.getResolvedLabelsForEpisode("ep_3", 10);
    expect(resolved).toHaveLength(1);
    expect(resolved[0]?.resolution).toBe("discarded_duplicate");
    expect(resolved[0]?.note).toContain("equal-trust override");
  });

  it("promotes higher-trust evidence over an existing lower-trust reward", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({
      id: "ep_4",
      conversationId: 10,
      reward: -0.5,
      rewardSource: "self",
    }));
    store.insertEvidence({
      episodeId: "ep_4",
      conversationId: 10,
      source: "human",
      kind: "human_feedback",
      value: 0.8,
      confidence: 0.95,
      reason: "user confirmed correct behavior",
    });

    await (worker as any).processEvidence();
    await (worker as any).processLabels();

    const episode = store.getEpisode("ep_4");
    expect(episode?.reward).toBe(0.8);
    expect(episode?.rewardSource).toBe("human");

    const resolved = store.getResolvedLabelsForEpisode("ep_4", 10);
    expect(resolved).toHaveLength(1);
    expect(resolved[0]?.resolution).toBe("promoted_to_label");
    expect(resolved[0]?.source).toBe("human");
  });
});
