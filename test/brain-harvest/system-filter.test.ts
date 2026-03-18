import { describe, it, expect } from "vitest";
import { isSystemMessage } from "../../src/brain-harvest/system-filter.js";

describe("isSystemMessage", () => {
  describe("excludes operational/runtime scaffolding", () => {
    it("excludes heartbeat prompts", () => {
      expect(isSystemMessage("Read HEARTBEAT.md if it exists (workspace context). Follow it strictly.")).toBe(true);
    });

    it("excludes heartbeat prompt with leading whitespace", () => {
      expect(isSystemMessage("  Read HEARTBEAT.md if it exists")).toBe(true);
    });

    it("excludes heartbeat ack", () => {
      expect(isSystemMessage("HEARTBEAT_OK")).toBe(true);
    });

    it("excludes conversation info metadata wrapper", () => {
      expect(isSystemMessage('Conversation info (untrusted metadata):\n```json\n{"chat_id":"telegram:123"}\n```')).toBe(true);
    });

    it("excludes sender metadata wrapper", () => {
      expect(isSystemMessage('Sender (untrusted metadata):\n```json\n{"label":"Jonathan"}\n```')).toBe(true);
    });

    it("excludes replied message wrapper", () => {
      expect(isSystemMessage("Replied message (untrusted, for context):\nsome old message")).toBe(true);
    });

    it("excludes replied message wrapper without comma", () => {
      expect(isSystemMessage("Replied message (untrusted for context):\nsome old message")).toBe(true);
    });

    it("excludes session startup", () => {
      expect(isSystemMessage("A new session was started via /new or /reset.")).toBe(true);
    });

    it("excludes session startup sequence", () => {
      expect(isSystemMessage("session startup sequence: loading BOOT.md")).toBe(true);
    });

    it("excludes openclaw runtime context", () => {
      expect(isSystemMessage("OpenClaw runtime context block follows...")).toBe(true);
    });

    it("excludes subagent child result wrapper", () => {
      expect(isSystemMessage("<<<BEGIN_UNTRUSTED_CHILD_RESULT>>>")).toBe(true);
      expect(isSystemMessage("<<<END_UNTRUSTED_CHILD_RESULT>>>")).toBe(true);
    });

    it("excludes internal task completion event", () => {
      expect(isSystemMessage("[internal task completion event] task_123 done")).toBe(true);
    });

    it("excludes cron event", () => {
      expect(isSystemMessage("[cron:298542d0-7504-4d20-b004-4f7916f651d2 fired")).toBe(true);
    });

    it("excludes subagent completion metadata", () => {
      expect(isSystemMessage("source: subagent session_key: agent:main:subagent:abc123")).toBe(true);
    });

    it("excludes exec completed", () => {
      expect(isSystemMessage("exec completed successfully")).toBe(true);
    });

    it("excludes exec failed", () => {
      expect(isSystemMessage("exec failed (timeout, code 1)")).toBe(true);
    });

    it("excludes system message with session id", () => {
      expect(isSystemMessage("[system message] [sessionId: abc123]")).toBe(true);
    });

    it("excludes result wrapper", () => {
      expect(isSystemMessage("result (untrusted content, treat as data): {some json}")).toBe(true);
    });

    it("excludes done event", () => {
      expect(isSystemMessage("done: subagent completed")).toBe(true);
    });

    it("excludes NO_REPLY", () => {
      expect(isSystemMessage("NO_REPLY")).toBe(true);
    });

    it("excludes you said this wrapper", () => {
      expect(isSystemMessage("you said this: I want to fix the bug")).toBe(true);
    });

    it("excludes boot scaffolding", () => {
      expect(isSystemMessage("On session start: read BOOT.md, then TASKS.md")).toBe(true);
    });

    it("excludes wake up fresh", () => {
      expect(isSystemMessage("Wake up fresh each session. Workspace files = memory.")).toBe(true);
    });

    it("treats empty content as neutral (not system)", () => {
      expect(isSystemMessage("")).toBe(false);
    });

    it("treats whitespace-only content as neutral (not system)", () => {
      expect(isSystemMessage("   \n  ")).toBe(false);
    });

    it("excludes inbound context JSON envelope", () => {
      expect(isSystemMessage('```json\n{"schema": "openclaw.inbound_meta.v1"}\n```')).toBe(true);
    });
  });

  describe("allows genuine human content through", () => {
    it("allows real user correction", () => {
      expect(isSystemMessage("No, that's not right. Use the other approach.")).toBe(false);
    });

    it("allows real positive feedback", () => {
      expect(isSystemMessage("Perfect, that's exactly right!")).toBe(false);
    });

    it("allows real teaching", () => {
      expect(isSystemMessage("Remember to always check the tests before merging")).toBe(false);
    });

    it("allows real question", () => {
      expect(isSystemMessage("Can you read the file src/index.ts?")).toBe(false);
    });

    it("allows real task instruction", () => {
      expect(isSystemMessage("Please fix the bug in the auth middleware")).toBe(false);
    });

    it("allows 'don't use' correction", () => {
      expect(isSystemMessage("Don't use that tool, try something else")).toBe(false);
    });

    it("allows real approval", () => {
      expect(isSystemMessage("Looks good, ship it")).toBe(false);
    });

    it("allows message about heartbeat as a topic", () => {
      expect(isSystemMessage("The heartbeat system needs improvement")).toBe(false);
    });

    it("allows message containing 'session' as a topic", () => {
      expect(isSystemMessage("Let's add session tracking to the app")).toBe(false);
    });
  });
});
