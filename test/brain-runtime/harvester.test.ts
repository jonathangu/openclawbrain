import { describe, it, expect, vi, afterEach } from "vitest";
import { DatabaseSync } from "node:sqlite";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LabelHarvester } from "../../src/brain-runtime/harvester-extension.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import type { Episode } from "../../src/brain-core/types.js";

const tempDirs: string[] = [];

function setup(resolveEpisodeIdForConversation?: (conversationId: number) => string | null | undefined) {
  const dir = mkdtempSync(join(tmpdir(), "brain-harvester-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);
  const store = new BrainStore(db);
  const log = { info: vi.fn(), warn: vi.fn() };
  const harvester = new LabelHarvester(store, log, resolveEpisodeIdForConversation);
  return { store, harvester };
}

function makeEpisode(params: {
  id: string;
  conversationId: number;
  createdAt: number;
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
    reward: null,
    rewardSource: null,
    packVersion: 1,
    createdAt: params.createdAt,
  };
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("LabelHarvester", () => {
  describe("system message filtering", () => {
    it("blocks heartbeat prompts from producing evidence", () => {
      const { harvester } = setup();
      expect(harvester.detectLabel("user", "Read HEARTBEAT.md if it exists (workspace context). Follow it strictly.")).toBeNull();
    });

    it("blocks metadata wrappers from producing evidence", () => {
      const { harvester } = setup();
      expect(harvester.detectLabel("user", 'Conversation info (untrusted metadata):\n```json\n{"chat_id":"telegram:123"}\n```')).toBeNull();
    });

    it("blocks NO_REPLY from producing evidence", () => {
      const { harvester } = setup();
      expect(harvester.detectLabel("user", "NO_REPLY")).toBeNull();
    });

    it("blocks HEARTBEAT_OK from producing evidence", () => {
      const { harvester } = setup();
      expect(harvester.detectLabel("user", "HEARTBEAT_OK")).toBeNull();
    });

    it("blocks session startup from producing evidence", () => {
      const { harvester } = setup();
      expect(harvester.detectLabel("user", "A new session was started via /new or /reset.")).toBeNull();
    });

    it("still allows real corrections through", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "No, that's not right. Use the other approach.");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("human");
      expect(result!.value).toBeLessThan(0);
    });
  });

  describe("detectLabel (human patterns)", () => {
    it("detects negative human correction", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "No, that's not right. Use the other approach.");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("human");
      expect(result!.value).toBeLessThan(0);
    });

    it("detects 'don't use' pattern", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "Don't use that tool, try something else");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("human");
      expect(result!.value).toBeLessThan(0);
    });

    it("detects 'instead use' pattern", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "Instead use gh pr create for this");
      expect(result).not.toBeNull();
      expect(result!.value).toBeLessThan(0);
    });

    it("detects positive human feedback", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("user", "Perfect, that's exactly right!");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("human");
      expect(result!.value).toBeGreaterThan(0);
    });

    it("returns null for neutral messages", () => {
      const { harvester } = setup();
      expect(harvester.detectLabel("user", "Can you read the file src/index.ts?")).toBeNull();
    });
  });

  describe("detectLabel (self patterns)", () => {
    it("detects tool failure", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("tool", "Error: ENOENT: no such file or directory");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("self");
      expect(result!.value).toBeLessThan(0);
    });

    it("detects tool success", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("tool", "Successfully deployed to production");
      expect(result).not.toBeNull();
      expect(result!.source).toBe("self");
      expect(result!.value).toBeGreaterThan(0);
    });

    it("detects structured tool-result failure without relying on text keywords", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("tool", "", [
        {
          partType: "tool",
          ordinal: 2,
          toolCallId: "call_1",
          toolName: "bash",
          toolInput: JSON.stringify({ command: "pnpm test" }),
          toolOutput: JSON.stringify({ ok: false, code: "ENOENT", exitCode: 2, filesTouched: ["docs/EVIDENCE.md"] }),
          metadata: JSON.stringify({ originalRole: "toolResult", rawType: "tool_result" }),
        },
      ]);
      expect(result).not.toBeNull();
      expect(result!.source).toBe("self");
      expect(result!.value).toBeLessThan(0);
      expect(result!.extractor).toBe("structured_tool_result");
      expect(result!.metadata).toMatchObject({
        partOrdinal: 2,
        toolCallId: "call_1",
        toolName: "bash",
        command: "pnpm test",
        exitCode: 2,
        filesTouched: ["docs/EVIDENCE.md"],
        rawType: "tool_result",
      });
    });

    it("detects structured tool-result success without relying on text keywords", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("tool", "", [
        {
          partType: "tool",
          toolOutput: JSON.stringify({ ok: true, cwd: "/tmp" }),
          metadata: JSON.stringify({ originalRole: "toolResult", rawType: "function_call_output" }),
        },
      ]);
      expect(result).not.toBeNull();
      expect(result!.source).toBe("self");
      expect(result!.value).toBeGreaterThan(0);
      expect(result!.extractor).toBe("structured_tool_result");
    });

    it("detects test pass", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel("assistant", "All 247 tests passed");
      expect(result).not.toBeNull();
      expect(result!.value).toBeGreaterThan(0);
    });

    it("detects scanner workflow patterns", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel(
        "assistant",
        "Runbook:\n1. Inspect CI logs\n2. Retry deployment with the known-safe flag",
      );
      expect(result).not.toBeNull();
      expect(result!.source).toBe("scanner");
      expect(result!.value).toBeGreaterThan(0);
    });

    it("detects structured scanner guidance without relying on one exact pattern", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel(
        "assistant",
        "## Deployment checklist\n- Inspect docs/DEPLOY.md\n- Run `pnpm test`\n- Run `openclawbrain doctor`\n- Retry the rollout only after verification",
      );
      expect(result).not.toBeNull();
      expect(result!.source).toBe("scanner");
      expect(result!.reason).toContain("scanner heuristic");
      expect(result!.metadata).toMatchObject({
        guidanceSignals: expect.arrayContaining(["heading"]),
        fileRefs: 1,
        hasHeading: true,
      });
    });

    it("detects structured scanner tool-chain guidance from message parts", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel(
        "assistant",
        "## Release runbook\n1. Inspect the deploy report\n2. Compare the current branch state before promotion",
        [
          {
            partType: "tool",
            ordinal: 1,
            toolCallId: "call_1",
            toolName: "bash",
            toolInput: JSON.stringify({ command: "pnpm test" }),
            toolOutput: JSON.stringify({ artifactPath: "docs/evidence/run.txt" }),
            metadata: JSON.stringify({ originalRole: "assistant", rawType: "function_call" }),
          },
          {
            partType: "tool",
            ordinal: 2,
            toolCallId: "call_2",
            toolName: "gh",
            toolInput: JSON.stringify({ command: "gh pr view 12" }),
            metadata: JSON.stringify({ originalRole: "assistant", rawType: "function_call" }),
          },
        ],
      );
      expect(result).not.toBeNull();
      expect(result!.source).toBe("scanner");
      expect(result!.extractor).toBe("structured_tool_chain");
      expect(result!.metadata).toMatchObject({
        toolNames: ["bash", "gh"],
        commands: ["pnpm test", "gh pr view 12"],
        fileHints: ["docs/evidence/run.txt"],
        partOrdinals: [1, 2],
      });
    });

    it("detects structured scanner guidance from explicit non-tool message parts", () => {
      const { harvester } = setup();
      const result = harvester.detectLabel(
        "assistant",
        "## Incident workflow\n1. Open the captured snapshot\n2. Compare the rollout notes before retrying",
        [
          {
            partType: "file",
            ordinal: 1,
            textContent: "docs/evidence/run.txt",
            metadata: JSON.stringify({
              originalRole: "assistant",
              rawType: "file",
              raw: { path: "docs/evidence/run.txt", fileName: "run.txt" },
            }),
          },
          {
            partType: "snapshot",
            ordinal: 2,
            metadata: JSON.stringify({
              originalRole: "assistant",
              rawType: "snapshot",
              raw: { title: "worker crash snapshot", path: "docs/evidence/snapshot.json" },
            }),
          },
          {
            partType: "subtask",
            ordinal: 3,
            metadata: JSON.stringify({
              originalRole: "assistant",
              rawType: "subtask",
              raw: { title: "verify deployment report", taskId: "task_123" },
            }),
          },
        ],
      );
      expect(result).not.toBeNull();
      expect(result!.source).toBe("scanner");
      expect(result!.extractor).toBe("structured_guidance_parts");
      expect(result!.metadata).toMatchObject({
        structuredPartTypes: ["file", "snapshot", "subtask"],
        pathHints: expect.arrayContaining(["docs/evidence/run.txt", "run.txt", "docs/evidence/snapshot.json"]),
        labels: expect.arrayContaining([
          "docs/evidence/run.txt",
          "run.txt",
          "worker crash snapshot",
          "docs/evidence/snapshot.json",
          "verify deployment report",
        ]),
        partOrdinals: [1, 2, 3],
      });
    });

    it("detects multiple evidence signals from one assistant message when both are present", () => {
      const { harvester } = setup();
      const results = harvester.detectLabels(
        "assistant",
        "Successfully deployed to production.\n\nRunbook:\n1. Inspect CI logs\n2. Retry deployment with the known-safe flag",
      );
      expect(results).toHaveLength(2);
      expect(results.map((result) => result.source).sort()).toEqual(["scanner", "self"]);
      expect(results.map((result) => result.extractor).sort()).toEqual(["scanner_heuristic", "self_pattern"]);
    });
  });

  describe("harvestFromMessage", () => {
    it("prefers an exact episode id over the latest conversation episode", async () => {
      const { store, harvester } = setup();
      store.insertEpisode(makeEpisode({
        id: "ep_older",
        conversationId: 7,
        createdAt: Date.now() - 10_000,
      }));
      store.insertEpisode(makeEpisode({
        id: "ep_newer",
        conversationId: 7,
        createdAt: Date.now(),
      }));

      await harvester.harvestFromMessage({
        conversationId: 7,
        episodeId: "ep_older",
        role: "tool",
        content: "Successfully deployed to production",
      });

      const evidence = store.getPendingEvidence();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.episodeId).toBe("ep_older");
      expect(evidence[0]?.kind).toBe("self_result");
      expect(evidence[0]?.metadata?.attributionMode).toBe("explicit");
      expect(evidence[0]?.metadata?.explicitEpisodeId).toBe("ep_older");
    });

    it("prefers the resolver-provided pending episode for the conversation", async () => {
      const { store, harvester } = setup((conversationId) => conversationId === 7 ? "ep_pending" : null);
      store.insertEpisode(makeEpisode({
        id: "ep_pending",
        conversationId: 7,
        createdAt: Date.now() - 10_000,
      }));
      store.insertEpisode(makeEpisode({
        id: "ep_newer",
        conversationId: 7,
        createdAt: Date.now(),
      }));

      await harvester.harvestFromMessage({
        conversationId: 7,
        role: "user",
        content: "Perfect, that's exactly right!",
      });

      const evidence = store.getPendingEvidence();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.episodeId).toBe("ep_pending");
      expect(evidence[0]?.metadata?.resolvedEpisodeId).toBe("ep_pending");
      expect(evidence[0]?.metadata?.attributionMode).toBe("resolver");
    });

    it("falls back to the most recent episode in the same conversation", async () => {
      const { store, harvester } = setup();
      store.insertEpisode(makeEpisode({
        id: "ep_other",
        conversationId: 99,
        createdAt: Date.now() - 10_000,
      }));
      store.insertEpisode(makeEpisode({
        id: "ep_target",
        conversationId: 7,
        createdAt: Date.now(),
      }));

      await harvester.harvestFromMessage({
        conversationId: 7,
        episodeId: "ep_other",
        role: "user",
        content: "Perfect, that's exactly right!",
      });

      const evidence = store.getPendingEvidence();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.episodeId).toBe("ep_target");
      expect(evidence[0]?.kind).toBe("human_feedback");
      expect(evidence[0]?.metadata?.attributionMode).toBe("recent_conversation_fallback");
      expect(evidence[0]?.metadata?.explicitEpisodeId).toBe("ep_other");
      expect(evidence[0]?.metadata?.matchedEpisodeId).toBe("ep_target");
    });

    it("persists multiple raw evidence rows from one harvested assistant message", async () => {
      const { store, harvester } = setup((conversationId) => conversationId === 7 ? "ep_pending" : null);
      store.insertEpisode(makeEpisode({
        id: "ep_pending",
        conversationId: 7,
        createdAt: Date.now(),
      }));

      await harvester.harvestFromMessage({
        conversationId: 7,
        role: "assistant",
        content: "Successfully deployed to production.\n\n## Deployment checklist\n- Inspect docs/DEPLOY.md\n- Run `pnpm test`\n- Retry deployment only after verification",
      });

      const evidence = store.getPendingEvidence();
      expect(evidence).toHaveLength(2);
      expect(evidence.every((row) => row.episodeId === "ep_pending")).toBe(true);
      expect(evidence.map((row) => row.source).sort()).toEqual(["scanner", "self"]);
      expect(evidence.map((row) => row.metadata?.extractor).sort()).toEqual(["scanner_heuristic", "self_pattern"]);
      expect(evidence.map((row) => row.metadata?.evidenceCount)).toEqual([2, 2]);
      expect(evidence.map((row) => row.metadata?.attributionMode)).toEqual(["resolver", "resolver"]);
    });

    it("persists structured scanner metadata when guidance is backed by explicit tool-call parts", async () => {
      const { store, harvester } = setup((conversationId) => conversationId === 7 ? "ep_pending" : null);
      store.insertEpisode(makeEpisode({
        id: "ep_pending",
        conversationId: 7,
        createdAt: Date.now(),
      }));

      await harvester.harvestFromMessage({
        conversationId: 7,
        messageId: 77,
        role: "assistant",
        content: "## Release runbook\n1. Inspect the deploy report\n2. Compare the current branch state before promotion",
        messageParts: [
          {
            partType: "tool",
            ordinal: 1,
            toolCallId: "call_1",
            toolName: "bash",
            toolInput: JSON.stringify({ command: "pnpm test" }),
            toolOutput: JSON.stringify({ artifactPath: "docs/evidence/run.txt" }),
            metadata: JSON.stringify({ originalRole: "assistant", rawType: "function_call" }),
          },
          {
            partType: "tool",
            ordinal: 2,
            toolCallId: "call_2",
            toolName: "gh",
            toolInput: JSON.stringify({ command: "gh pr view 12" }),
            metadata: JSON.stringify({ originalRole: "assistant", rawType: "function_call" }),
          },
        ],
      });

      const evidence = store.getPendingEvidence();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.source).toBe("scanner");
      expect(evidence[0]?.metadata).toMatchObject({
        messageId: 77,
        extractor: "structured_tool_chain",
        toolNames: ["bash", "gh"],
        commands: ["pnpm test", "gh pr view 12"],
        fileHints: ["docs/evidence/run.txt"],
        partOrdinals: [1, 2],
      });
    });

    it("persists structured scanner metadata when guidance is backed by explicit non-tool parts", async () => {
      const { store, harvester } = setup((conversationId) => conversationId === 7 ? "ep_pending" : null);
      store.insertEpisode(makeEpisode({
        id: "ep_pending",
        conversationId: 7,
        createdAt: Date.now(),
      }));

      await harvester.harvestFromMessage({
        conversationId: 7,
        messageId: 78,
        role: "assistant",
        content: "## Incident workflow\n1. Open the captured snapshot\n2. Compare the rollout notes before retrying",
        messageParts: [
          {
            partType: "file",
            ordinal: 1,
            textContent: "docs/evidence/run.txt",
            metadata: JSON.stringify({
              originalRole: "assistant",
              rawType: "file",
              raw: { path: "docs/evidence/run.txt", fileName: "run.txt" },
            }),
          },
          {
            partType: "snapshot",
            ordinal: 2,
            metadata: JSON.stringify({
              originalRole: "assistant",
              rawType: "snapshot",
              raw: { title: "worker crash snapshot", path: "docs/evidence/snapshot.json" },
            }),
          },
          {
            partType: "subtask",
            ordinal: 3,
            metadata: JSON.stringify({
              originalRole: "assistant",
              rawType: "subtask",
              raw: { title: "verify deployment report", taskId: "task_123" },
            }),
          },
        ],
      });

      const evidence = store.getPendingEvidence();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.source).toBe("scanner");
      expect(evidence[0]?.metadata).toMatchObject({
        messageId: 78,
        extractor: "structured_guidance_parts",
        structuredPartTypes: ["file", "snapshot", "subtask"],
        pathHints: expect.arrayContaining(["docs/evidence/run.txt", "run.txt", "docs/evidence/snapshot.json"]),
        labels: expect.arrayContaining([
          "docs/evidence/run.txt",
          "run.txt",
          "worker crash snapshot",
          "docs/evidence/snapshot.json",
          "verify deployment report",
        ]),
        partOrdinals: [1, 2, 3],
      });
    });

    it("persists structured tool-result evidence even when flattened text is empty", async () => {
      const { store, harvester } = setup((conversationId) => conversationId === 7 ? "ep_pending" : null);
      store.insertEpisode(makeEpisode({
        id: "ep_pending",
        conversationId: 7,
        createdAt: Date.now(),
      }));

      await harvester.harvestFromMessage({
        conversationId: 7,
        messageId: 42,
        role: "tool",
        content: "",
        messageParts: [
          {
            partType: "tool",
            ordinal: 1,
            toolCallId: "call_1",
            toolName: "bash",
            toolInput: JSON.stringify({ command: "pnpm test" }),
            toolOutput: JSON.stringify({ ok: false, code: "ENOENT", exitCode: 2, filesTouched: ["docs/EVIDENCE.md"], artifactPath: "docs/evidence/run.txt" }),
            metadata: JSON.stringify({ originalRole: "toolResult", rawType: "tool_result" }),
          },
        ],
      });

      const evidence = store.getPendingEvidence();
      expect(evidence).toHaveLength(1);
      expect(evidence[0]?.episodeId).toBe("ep_pending");
      expect(evidence[0]?.source).toBe("self");
      expect(evidence[0]?.metadata).toMatchObject({
        messageId: 42,
        extractor: "structured_tool_result",
        messagePartCount: 1,
        partOrdinal: 1,
        toolCallId: "call_1",
        toolName: "bash",
        command: "pnpm test",
        exitCode: 2,
        filesTouched: ["docs/EVIDENCE.md"],
        artifactPath: "docs/evidence/run.txt",
      });
    });
  });
});
