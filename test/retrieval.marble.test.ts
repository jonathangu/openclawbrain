import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { getLcmConnection, closeLcmConnection } from "../src/db/connection.js";
import { runLcmMigrations } from "../src/db/migration.js";
import { ConversationStore } from "../src/store/conversation-store.js";
import { SummaryStore } from "../src/store/summary-store.js";
import { RetrievalEngine } from "../src/retrieval.js";
import { createLcmDescribeTool } from "../src/tools/lcm-describe-tool.js";
import { createLcmGrepTool } from "../src/tools/lcm-grep-tool.js";

const tempDirs: string[] = [];

afterEach(() => {
  closeLcmConnection();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function makeDeps() {
  return {
    config: { maxExpandTokens: 1_000 },
    isSubagentSessionKey: () => false,
    normalizeAgentId: (agentId: string) => agentId,
    parseAgentSessionKey: () => null,
  } as never;
}

function makeLcm(retrieval: RetrievalEngine, conversationStore: ConversationStore) {
  return {
    getRetrieval: () => retrieval,
    timezone: "UTC",
    getConversationStore: () => conversationStore,
  } as never;
}

describe("marble retrieval surfaces", () => {
  it("describes marbles and greps them alongside messages", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "openclawbrain-marble-retrieval-"));
    tempDirs.push(tempDir);
    const db = getLcmConnection(join(tempDir, "db.sqlite"));
    runLcmMigrations(db);

    const conversationStore = new ConversationStore(db);
    const summaryStore = new SummaryStore(db);
    const conversation = await conversationStore.createConversation({
      sessionId: "marble-session",
      title: "Marble retrieval test",
    });

    const message = await conversationStore.createMessage({
      conversationId: conversation.conversationId,
      seq: 1,
      role: "user",
      content: "Use gh pr create for the packaging branch.",
      tokenCount: 10,
    });

    await summaryStore.insertMarble({
      marbleId: "mar_typed_001",
      conversationId: conversation.conversationId,
      marbleKind: "typed_extraction",
      compressionVersion: 1,
      renderVersion: 1,
      content: "Typed extraction: use gh pr create for the packaging branch.",
      payloadJson: JSON.stringify({
        kind: "typed_extraction",
        command: "gh pr create",
      }),
      tokenCount: 14,
      confidence: 0.91,
      freshnessState: "fresh",
      sourceFingerprint: "srcfp_001",
      contentHash: "hash_001",
      provenanceRef: "prov_001",
      sourceArtifactTokenCount: 24,
      sources: [
        {
          sourceKind: "message",
          sourceId: String(message.messageId),
          sourceDigest: "digest_msg_001",
          sourceProvenanceRef: "prov_msg_001",
          ordinal: 0,
        },
        {
          sourceKind: "tool_result",
          sourceId: "tool_001",
          sourceSubId: "stdout",
          sourceDigest: "digest_tool_001",
          sourceProvenanceRef: "prov_tool_001",
          sourceUri: "tool://result/001",
          ordinal: 1,
        },
      ],
    });

    const retrieval = new RetrievalEngine(conversationStore as never, summaryStore as never);

    const described = await retrieval.describe("mar_typed_001");
    expect(described).not.toBeNull();
    expect(described?.type).toBe("marble");
    expect(described?.marble?.provenanceRef).toBe("prov_001");
    expect(described?.marble?.sourceRefs).toEqual([
      `message:${message.messageId}`,
      "tool_result:tool_001#stdout",
    ]);
    expect(described?.marble?.sources).toHaveLength(2);

    const marblesOnly = await retrieval.grep({
      query: "gh pr create",
      mode: "full_text",
      scope: "marbles",
      conversationId: conversation.conversationId,
      limit: 10,
    });
    expect(marblesOnly.marbles).toHaveLength(1);
    expect(marblesOnly.totalMatches).toBe(1);
    expect(marblesOnly.marbles[0]?.provenanceRef).toBe("prov_001");

    const both = await retrieval.grep({
      query: "gh pr create",
      mode: "full_text",
      scope: "both",
      conversationId: conversation.conversationId,
      limit: 10,
    });
    expect(both.messages).toHaveLength(1);
    expect(both.marbles).toHaveLength(1);
    expect(both.totalMatches).toBe(2);
  });

  it("renders marble provenance through the operator tools", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "openclawbrain-marble-tools-"));
    tempDirs.push(tempDir);
    const db = getLcmConnection(join(tempDir, "db.sqlite"));
    runLcmMigrations(db);

    const conversationStore = new ConversationStore(db);
    const summaryStore = new SummaryStore(db);
    const conversation = await conversationStore.createConversation({
      sessionId: "marble-session-tools",
      title: "Marble tools test",
    });

    const message = await conversationStore.createMessage({
      conversationId: conversation.conversationId,
      seq: 1,
      role: "user",
      content: "Use gh pr create for the packaging branch.",
      tokenCount: 10,
    });

    await summaryStore.insertMarble({
      marbleId: "mar_tools_001",
      conversationId: conversation.conversationId,
      marbleKind: "typed_extraction",
      compressionVersion: 1,
      renderVersion: 1,
      content: "Typed extraction: use gh pr create for the packaging branch.",
      payloadJson: JSON.stringify({
        kind: "typed_extraction",
        command: "gh pr create",
      }),
      tokenCount: 14,
      confidence: 0.91,
      freshnessState: "fresh",
      sourceFingerprint: "srcfp_002",
      contentHash: "hash_002",
      provenanceRef: "prov_002",
      sourceArtifactTokenCount: 24,
      sources: [
        {
          sourceKind: "message",
          sourceId: String(message.messageId),
          sourceDigest: "digest_msg_002",
          sourceProvenanceRef: "prov_msg_002",
          ordinal: 0,
        },
      ],
    });

    const retrieval = new RetrievalEngine(conversationStore as never, summaryStore as never);
    const lcm = makeLcm(retrieval, conversationStore);
    const deps = makeDeps();

    const describeTool = createLcmDescribeTool({ deps, lcm, sessionId: "session-1" });
    const describeOutput = await describeTool.execute("tool-1", {
      id: "mar_tools_001",
      conversationId: conversation.conversationId,
    });
    const describeText =
      describeOutput.content[0]?.type === "text" ? describeOutput.content[0].text : "";
    expect(describeText).toContain("## Marble mar_tools_001");
    expect(describeText).toContain("**Freshness:** fresh");
    expect(describeText).toContain("**Provenance:** prov_002");
    expect(describeText).toContain("message:");
    expect(describeText).not.toContain("tool_result:");

    const grepTool = createLcmGrepTool({ deps, lcm, sessionId: "session-1" });
    const grepOutput = await grepTool.execute("tool-2", {
      pattern: "gh pr create",
      mode: "full_text",
      scope: "marbles",
      conversationId: conversation.conversationId,
      limit: 10,
    });
    const grepText = grepOutput.content[0]?.type === "text" ? grepOutput.content[0].text : "";
    expect(grepText).toContain("### Marbles");
    expect(grepText).toContain("prov prov_002");
    expect(grepText).toContain("src srcfp_002");
  });
});
