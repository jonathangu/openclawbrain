import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { closeLcmConnection, getLcmConnection } from "../../src/db/connection.js";
import { runLcmMigrations } from "../../src/db/migration.js";
import { ConversationStore } from "../../src/store/conversation-store.js";
import { SummaryStore } from "../../src/store/summary-store.js";

const tempDirs: string[] = [];

afterEach(() => {
  closeLcmConnection();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("SummaryStore lineage and snapshots", () => {
  it("round-trips lineage records and branch snapshots", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "lossless-claw-lineage-"));
    tempDirs.push(tempDir);
    const db = getLcmConnection(join(tempDir, "lineage.db"));
    runLcmMigrations(db);

    const conversationStore = new ConversationStore(db);
    const conversation = await conversationStore.createConversation({
      sessionId: "lineage-session",
      title: "Lineage store test",
    });

    const summaryStore = new SummaryStore(db);
    await summaryStore.insertSummary({
      summaryId: "sum_lineage_001",
      conversationId: conversation.conversationId,
      kind: "leaf",
      content: "Leaf summary for lineage testing.",
      tokenCount: 8,
    });

    const fallback = await summaryStore.getSummaryLineage("sum_lineage_001");
    expect(fallback).not.toBeNull();
    expect(fallback?.branchId).toBe(`branch_${conversation.conversationId}_main`);
    expect(fallback?.summaryRole).toBe("support");
    expect(fallback?.truthBasis).toBe("derived");
    expect(fallback?.freshnessState).toBe("fresh");

    const lineage = await summaryStore.insertSummaryLineage({
      summaryId: "sum_lineage_001",
      conversationId: conversation.conversationId,
      branchId: "branch_test_a",
      episodeId: "ep_test_a",
      summaryRole: "episode",
      truthBasis: "canonical",
      parentBranchId: null,
      typedMemoryRefs: ["bn_correction_1"],
      forkReason: "branch_fork",
    });

    expect(lineage.branchId).toBe("branch_test_a");
    expect(lineage.snapshotId).toBeNull();
    expect(lineage.typedMemoryRefs).toEqual(["bn_correction_1"]);
    expect(lineage.freshnessState).toBe("fresh");
    expect(lineage.invalidatedAt).toBeNull();
    expect(lineage.invalidationReason).toBeNull();

    const snapshot = await summaryStore.insertBranchSnapshot({
      snapshotId: "snap_test_a",
      conversationId: conversation.conversationId,
      branchId: "branch_test_a",
      episodeId: "ep_test_a",
      activeSummaryId: "sum_lineage_001",
      contextOrdinal: 7,
      summarySpineIds: ["sum_lineage_001"],
      typedMemoryRefs: ["bn_correction_1"],
      openQuestionRefs: ["question_1"],
      stateJson: JSON.stringify({ reason: "episode_summary", branchId: "branch_test_a" }),
    });

    expect(snapshot.snapshotId).toBe("snap_test_a");
    expect(snapshot.summarySpineIds).toEqual(["sum_lineage_001"]);
    expect(snapshot.openQuestionRefs).toEqual(["question_1"]);

    const reloadedLineage = await summaryStore.getSummaryLineage("sum_lineage_001");
    expect(reloadedLineage?.snapshotId).toBeNull();

    await summaryStore.insertSummaryLineage({
      summaryId: "sum_lineage_001",
      conversationId: conversation.conversationId,
      branchId: "branch_test_a",
      episodeId: "ep_test_a",
      summaryRole: "snapshot",
      truthBasis: "canonical",
      parentBranchId: null,
      typedMemoryRefs: ["bn_correction_1"],
      snapshotId: "snap_test_a",
      forkReason: "episode_closed",
    });

    const linkedLineage = await summaryStore.getSummaryLineage("sum_lineage_001");
    expect(linkedLineage?.snapshotId).toBe("snap_test_a");
    expect(linkedLineage?.summaryRole).toBe("snapshot");

    const superseded = await summaryStore.invalidateSummaryLineage({
      summaryId: "sum_lineage_001",
      freshnessState: "superseded",
      reason: "condensed_into:sum_lineage_002",
    });
    expect(superseded?.freshnessState).toBe("superseded");
    expect(superseded?.invalidatedAt).not.toBeNull();
    expect(superseded?.invalidationReason).toBe("condensed_into:sum_lineage_002");

    const latestSnapshot = await summaryStore.getLatestBranchSnapshot(conversation.conversationId);
    expect(latestSnapshot?.snapshotId).toBe("snap_test_a");
    expect(latestSnapshot?.stateJson).toContain("episode_summary");
  });
});
