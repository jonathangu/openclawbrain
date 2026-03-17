import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { AgentMessage, OpenClawPluginToolContext } from "../src/openclaw-sdk-compat.js";
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

function findToolFactory(
  registerTool: ReturnType<typeof vi.fn>,
  toolName: string,
): (ctx: OpenClawPluginToolContext) => {
  name: string;
  execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
} {
  for (const [candidate] of registerTool.mock.calls as Array<[ToolFactory]>) {
    if (typeof candidate !== "function") {
      continue;
    }
    const tool = candidate({ sessionKey: "agent:main:test:seed" } as OpenClawPluginToolContext);
    if (tool && !Array.isArray(tool) && tool.name === toolName) {
      return candidate as (ctx: OpenClawPluginToolContext) => {
        name: string;
        execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
      };
    }
  }
  throw new Error(`Expected ${toolName} tool factory to be registered.`);
}

describe("brain_teach_user_correction plugin session binding", () => {
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

  it("binds the explicit user-correction tool to the conversation resolved from ctx.sessionKey", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "openclawbrain-user-correction-binding-"));
    tempDirs.add(tempDir);
    const dbPath = join(tempDir, "lcm.db");
    dbPaths.add(dbPath);
    const storePath = join(tempDir, "session-store.json");
    const sessionKey = "agent:main:demo:dm:+15550001111";
    const sessionId = "session-user-correction-binding";

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
      getBrainService: () => {
        teachUserCorrection: (params: Record<string, unknown>) => Promise<unknown>;
      } | null;
    };

    await engine.ingest({
      sessionId,
      message: {
        role: "user",
        content: "wrong, it changed to giraffe",
      } as AgentMessage,
    });

    const conversation = await engine.getConversationStore().getConversationBySessionId(sessionId);
    expect(conversation).not.toBeNull();

    const brain = engine.getBrainService();
    expect(brain).not.toBeNull();

    const teachSpy = vi.spyOn(brain!, "teachUserCorrection").mockResolvedValue({
      nodeId: "bn_codeword",
      packVersion: 2,
    });

    const toolFactory = findToolFactory(registerTool, "brain_teach_user_correction");
    const tool = toolFactory({ sessionKey } as OpenClawPluginToolContext);

    await tool.execute("call_brain_teach_user_correction", {
      canonicalInstruction: "The codeword is giraffe.",
      sourceQuote: "wrong, it changed to giraffe",
      tags: ["demo", "codeword"],
    });

    expect(teachSpy).toHaveBeenCalledWith({
      canonicalInstruction: "The codeword is giraffe.",
      sourceQuote: "wrong, it changed to giraffe",
      conversationId: conversation?.conversationId,
      tags: ["demo", "codeword"],
      via: "brain_teach_user_correction",
    });
  });
});
