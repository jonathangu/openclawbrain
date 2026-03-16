import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OpenClawPluginApi, OpenClawPluginToolContext } from "openclaw/plugin-sdk";
import lcmPlugin from "../index.js";
import { closeLcmConnection } from "../src/db/connection.js";

type RegisteredEngineFactory = (() => unknown) | undefined;

type ToolFactory = Parameters<OpenClawPluginApi["registerTool"]>[0];

function buildApi(params: { dbPath: string; storePath: string }): {
  api: OpenClawPluginApi;
  getFactory: () => RegisteredEngineFactory;
  registerTool: ReturnType<typeof vi.fn>;
} {
  let factory: RegisteredEngineFactory;
  const registerTool = vi.fn();

  const api = {
    id: "openclawbrain",
    name: "OpenClawBrain",
    source: "/tmp/openclawbrain",
    config: {},
    pluginConfig: {
      enabled: true,
      dbPath: params.dbPath,
      brainEnabled: true,
    },
    runtime: {
      subagent: {
        run: vi.fn(),
        waitForRun: vi.fn(),
        getSession: vi.fn(),
        deleteSession: vi.fn(),
      },
      config: {
        loadConfig: vi.fn(() => ({})),
      },
      channel: {
        session: {
          resolveStorePath: vi.fn(() => params.storePath),
        },
      },
      modelAuth: {
        getApiKeyForModel: vi.fn(async () => undefined),
        resolveApiKeyForProvider: vi.fn(async () => undefined),
      },
    },
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
    registerContextEngine: vi.fn((_id: string, nextFactory: () => unknown) => {
      factory = nextFactory;
    }),
    registerTool,
    registerHook: vi.fn(),
    registerHttpHandler: vi.fn(),
    registerHttpRoute: vi.fn(),
    registerChannel: vi.fn(),
    registerGatewayMethod: vi.fn(),
    registerCli: vi.fn(),
    registerService: vi.fn(),
    registerProvider: vi.fn(),
    registerCommand: vi.fn(),
    resolvePath: vi.fn(() => "/tmp/fake-agent"),
    on: vi.fn(),
  } as unknown as OpenClawPluginApi;

  return {
    api,
    getFactory: () => factory,
    registerTool,
  };
}

function findBrainTeachToolFactory(registerTool: ReturnType<typeof vi.fn>): (ctx: OpenClawPluginToolContext) => {
  name: string;
  execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
} {
  for (const [candidate] of registerTool.mock.calls as Array<[ToolFactory]>) {
    if (typeof candidate !== "function") {
      continue;
    }
    const tool = candidate({ sessionKey: "agent:main:test:seed" } as OpenClawPluginToolContext);
    if (tool && !Array.isArray(tool) && tool.name === "brain_teach") {
      return candidate as (ctx: OpenClawPluginToolContext) => {
        name: string;
        execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
      };
    }
  }
  throw new Error("Expected brain_teach tool factory to be registered.");
}

describe("brain_teach plugin session binding", () => {
  const tempDirs = new Set<string>();
  const dbPaths = new Set<string>();

  afterEach(() => {
    vi.restoreAllMocks();
    for (const dbPath of dbPaths) {
      closeLcmConnection(dbPath);
    }
    dbPaths.clear();
    for (const dir of tempDirs) {
      rmSync(dir, { recursive: true, force: true });
    }
    tempDirs.clear();
  });

  it("binds brain_teach to the conversation resolved from ctx.sessionKey", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "openclawbrain-brain-teach-binding-"));
    tempDirs.add(tempDir);
    const dbPath = join(tempDir, "lcm.db");
    dbPaths.add(dbPath);
    const storePath = join(tempDir, "session-store.json");
    const sessionKey = "agent:main:heartbeat:dm:+15550001111";
    const sessionId = "session-brain-teach-binding";

    writeFileSync(
      storePath,
      JSON.stringify({
        [sessionKey]: { sessionId },
      }),
      "utf8",
    );

    const { api, getFactory, registerTool } = buildApi({ dbPath, storePath });
    lcmPlugin.register(api);

    const factory = getFactory();
    if (!factory) {
      throw new Error("Expected context engine factory to be registered.");
    }

    const engine = factory() as {
      ingest: (params: { sessionId: string; message: AgentMessage }) => Promise<void>;
      getConversationStore: () => {
        getConversationBySessionId: (id: string) => Promise<{ conversationId: number } | null>;
      };
      getBrainService: () => { teach: (params: Record<string, unknown>) => Promise<unknown> } | null;
    };

    await engine.ingest({
      sessionId,
      message: {
        role: "user",
        content: "Deployment failed again.",
      } as AgentMessage,
    });

    const conversation = await engine.getConversationStore().getConversationBySessionId(sessionId);
    expect(conversation).not.toBeNull();

    const brain = engine.getBrainService();
    expect(brain).not.toBeNull();
    const teachSpy = vi.spyOn(brain!, "teach").mockResolvedValue({
      nodeId: "bn_test_binding",
      packVersion: 2,
    });

    const brainTeachFactory = findBrainTeachToolFactory(registerTool);
    const brainTeachTool = brainTeachFactory({ sessionKey } as OpenClawPluginToolContext);

    await brainTeachTool.execute("call_brain_teach", {
      instruction: "For deployment failures, inspect CI logs before retrying.",
      kind: "correction",
      tags: ["deploy", "ci"],
    });

    expect(teachSpy).toHaveBeenCalledWith({
      instruction: "For deployment failures, inspect CI logs before retrying.",
      conversationId: conversation?.conversationId,
      kind: "correction",
      tags: ["deploy", "ci"],
    });
  });
});
