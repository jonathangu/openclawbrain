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

describe("SummaryStore marbles", () => {
  it("round-trips marble records, source links, search, and invalidation", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "lossless-claw-marbles-"));
    tempDirs.push(tempDir);
    const dbPath = join(tempDir, "marbles.db");
    const db = getLcmConnection(dbPath);

    runLcmMigrations(db);

    const conversationStore = new ConversationStore(db);
    const conversation = await conversationStore.createConversation({
      sessionId: "marble-session",
      title: "Marble store test",
    });

    const store = new SummaryStore(db);
    const marble = await store.insertMarble({
      marbleId: "mar_typed_001",
      conversationId: conversation.conversationId,
      marbleKind: "typed_extraction",
      compressionVersion: 1,
      renderVersion: 1,
      content: "Typed extraction: gh pr create --draft for the packaging branch.",
      payloadJson: JSON.stringify({
        mode: "typed_extraction",
        command: "gh pr create --draft",
      }),
      tokenCount: 13,
      confidence: 0.87,
      freshnessState: "fresh",
      sourceFingerprint: "srcfp_001",
      contentHash: "hash_001",
      provenanceRef: "prov_001",
      sourceArtifactTokenCount: 39,
      sources: [
        {
          sourceKind: "tool_result",
          sourceId: "tool_001",
          sourceSubId: "stdout",
          sourceDigest: "digest_001",
          sourceProvenanceRef: "prov_src_001",
          sourceUri: "tool://result/001",
          ordinal: 0,
        },
        {
          sourceKind: "message",
          sourceId: "msg_002",
          sourceDigest: "digest_002",
          sourceProvenanceRef: "prov_src_002",
          ordinal: 1,
        },
      ],
    });

    expect(marble.marbleId).toBe("mar_typed_001");
    expect(marble.marbleKind).toBe("typed_extraction");
    expect(marble.freshnessState).toBe("fresh");
    expect(marble.sourceCount).toBe(2);
    expect(marble.sourceArtifactTokenCount).toBe(39);

    const stored = await store.getMarble("mar_typed_001");
    expect(stored).not.toBeNull();
    expect(stored?.payloadJson).toContain("gh pr create --draft");
    expect(stored?.invalidatedAt).toBeNull();

    const sources = await store.getMarbleSources("mar_typed_001");
    expect(sources).toHaveLength(2);
    expect(sources[0]?.sourceSubId).toBe("stdout");
    expect(sources[1]?.sourceSubId).toBeNull();

    const byConversation = await store.getMarblesByConversation(conversation.conversationId);
    expect(byConversation).toHaveLength(1);
    expect(byConversation[0]?.marbleId).toBe("mar_typed_001");

    const searchResults = await store.searchMarbles({
      query: "packaging branch",
      mode: "full_text",
      conversationId: conversation.conversationId,
      limit: 5,
    });
    expect(searchResults).toHaveLength(1);
    expect(searchResults[0]?.marbleId).toBe("mar_typed_001");
    expect(searchResults[0]?.freshnessState).toBe("fresh");

    const invalidated = await store.invalidateMarble({
      marbleId: "mar_typed_001",
      freshnessState: "stale_source",
      reason: "source output changed",
    });
    expect(invalidated?.freshnessState).toBe("stale_source");
    expect(invalidated?.invalidatedAt).not.toBeNull();
    expect(invalidated?.invalidationReason).toBe("source output changed");

    const reloaded = await store.getMarble("mar_typed_001");
    expect(reloaded?.freshnessState).toBe("stale_source");
    expect(reloaded?.invalidatedAt).not.toBeNull();
  });

  it("searches marbles without FTS5 via LIKE fallback", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "lossless-claw-marbles-no-fts-"));
    tempDirs.push(tempDir);
    const dbPath = join(tempDir, "marbles-no-fts.db");
    const db = getLcmConnection(dbPath);

    runLcmMigrations(db, { fts5Available: false });

    const conversationStore = new ConversationStore(db, { fts5Available: false });
    const conversation = await conversationStore.createConversation({
      sessionId: "marble-session-no-fts",
      title: "Marble store fallback test",
    });

    const store = new SummaryStore(db, { fts5Available: false });
    await store.insertMarble({
      marbleId: "mar_replay_002",
      conversationId: conversation.conversationId,
      marbleKind: "replay_stub",
      compressionVersion: 1,
      renderVersion: 1,
      content: "Replay stub: expand to source for the exact README command.",
      payloadJson: JSON.stringify({ kind: "replay_stub" }),
      tokenCount: 10,
      confidence: 0.61,
      freshnessState: "fresh",
      sourceFingerprint: "srcfp_002",
      contentHash: "hash_002",
      provenanceRef: "prov_002",
      sources: [],
    });

    const results = await store.searchMarbles({
      query: "README command",
      mode: "full_text",
      conversationId: conversation.conversationId,
      limit: 5,
    });

    expect(results).toHaveLength(1);
    expect(results[0]?.marbleId).toBe("mar_replay_002");
    expect(results[0]?.snippet.toLowerCase()).toContain("readme command");

    const ftsTables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts%'")
      .all() as Array<{ name: string }>;
    expect(ftsTables).toEqual([]);
  });
});
