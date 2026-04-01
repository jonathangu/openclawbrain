import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { closeLcmConnection, getLcmConnection } from "../src/db/connection.js";
import { runLcmMigrations } from "../src/db/migration.js";
import { ConversationStore } from "../src/store/conversation-store.js";
import { SummaryStore } from "../src/store/summary-store.js";
import { CompactionEngine } from "../src/compaction.js";

const tempDirs: string[] = [];

function makeConfig() {
  return {
    contextThreshold: 0.75,
    freshTailCount: 0,
    leafMinFanout: 2,
    condensedMinFanout: 2,
    condensedMinFanoutHard: 2,
    incrementalMaxDepth: 1,
    leafChunkTokens: 200,
    leafTargetTokens: 1200,
    condensedTargetTokens: 50,
    maxRounds: 3,
    timezone: "UTC",
  };
}

afterEach(() => {
  closeLcmConnection();
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("CompactionEngine lineage-aware behavior", () => {
  it("creates a snapshot instead of condensing across mixed branches", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "lossless-claw-compaction-"));
    tempDirs.push(tempDir);
    const db = getLcmConnection(join(tempDir, "compaction.db"));
    runLcmMigrations(db);

    const conversationStore = new ConversationStore(db);
    const conversation = await conversationStore.createConversation({
      sessionId: "compaction-lineage-session",
      title: "Compaction lineage test",
    });

    const summaryStore = new SummaryStore(db);
    const summaryA = await summaryStore.insertSummary({
      summaryId: "sum_branch_a",
      conversationId: conversation.conversationId,
      kind: "leaf",
      depth: 0,
      content: "Branch A leaf summary.",
      tokenCount: 30,
    });
    await summaryStore.insertSummaryLineage({
      summaryId: summaryA.summaryId,
      conversationId: conversation.conversationId,
      branchId: "branch_a",
      episodeId: "ep_a",
      summaryRole: "support",
      truthBasis: "derived",
      typedMemoryRefs: [],
      forkReason: "seed_a",
    });
    await summaryStore.appendContextSummary(conversation.conversationId, summaryA.summaryId);

    const summaryB = await summaryStore.insertSummary({
      summaryId: "sum_branch_b",
      conversationId: conversation.conversationId,
      kind: "leaf",
      depth: 0,
      content: "Branch B leaf summary.",
      tokenCount: 30,
    });
    await summaryStore.insertSummaryLineage({
      summaryId: summaryB.summaryId,
      conversationId: conversation.conversationId,
      branchId: "branch_b",
      episodeId: "ep_b",
      summaryRole: "support",
      truthBasis: "derived",
      typedMemoryRefs: [],
      forkReason: "seed_b",
    });
    await summaryStore.appendContextSummary(conversation.conversationId, summaryB.summaryId);

    const compaction = new CompactionEngine(conversationStore as any, summaryStore as any, makeConfig());
    const summarize = vi.fn(async () => "condensed branch summary");

    const result = await compaction.compactFullSweep({
      conversationId: conversation.conversationId,
      tokenBudget: 1_000,
      summarize,
      force: true,
      hardTrigger: true,
    });

    expect(result.actionTaken).toBe(true);
    expect(result.condensed).toBe(false);
    expect(result.createdSummaryId).toBeUndefined();
    expect(result.createdSnapshotId).toMatch(/^snap_/);
    expect(summarize).toHaveBeenCalled();

    const snapshots = await summaryStore.getBranchSnapshots(conversation.conversationId);
    expect(snapshots).toHaveLength(1);
    expect(snapshots[0]?.stateJson).toContain("branch_conflict");
    expect(snapshots[0]?.summarySpineIds).toEqual(["sum_branch_a", "sum_branch_b"]);

    const contextItems = await summaryStore.getContextItems(conversation.conversationId);
    expect(contextItems).toHaveLength(2);
    const summaries = await summaryStore.getSummariesByConversation(conversation.conversationId);
    expect(summaries).toHaveLength(2);
  });
});
