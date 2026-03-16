import { execFileSync } from "node:child_process";
import { DatabaseSync } from "node:sqlite";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { OpenClawPluginApi, OpenClawPluginToolContext } from "openclaw/plugin-sdk";
import lcmPlugin from "../index.js";
import { closeLcmConnection } from "../src/db/connection.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");

type RegisteredEngineFactory = (() => unknown) | undefined;
type ToolFactory = Parameters<OpenClawPluginApi["registerTool"]>[0];

type RunSummary = {
  iteration: number;
  sessionKey: string;
  sessionId: string;
  conversationId: number;
  warmupEpisodeId: string;
  taughtNodeId: string;
  teachPackVersion: number | null;
  assembleMode: string | null;
  assembledBrainHeaderVisible: boolean;
  assembledCorrectionVisible: boolean;
  traceIncludesTaughtNode: boolean;
  traceId: string | null;
  currentPackVersion: number | null;
  evidenceEpisodeId: string | null;
  evidenceCorrectedEpisodeId: string | null;
  evidenceExtractor: string | null;
  evidenceVia: string | null;
  fingerprint: Record<string, unknown>;
};

function parseArgs(argv: string[]) {
  const args = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key?.startsWith("--") && value) {
      args.set(key.slice(2), value);
      index += 1;
    }
  }

  const validationStateDir = resolve(
    args.get("state-dir")
      ?? process.env.OPENCLAWBRAIN_VALIDATION_ROOT
      ?? process.env.OPENCLAWBRAIN_VALIDATION_STATE_DIR
      ?? join(process.env.HOME ?? homedir(), ".openclaw-ocbphase1"),
  );
  const workspaceRoot = resolve(
    args.get("workspace")
      ?? process.env.OPENCLAWBRAIN_VALIDATION_WORKSPACE
      ?? join(process.env.HOME ?? homedir(), ".openclaw", "workspace-ocbphase1"),
  );
  const repetitions = Number.parseInt(args.get("repetitions") ?? "20", 10);
  const gitSha = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  const artifactDate = new Date().toISOString().slice(0, 10);
  const artifactDir = resolve(
    args.get("artifact-dir")
      ?? process.env.OPENCLAWBRAIN_VALIDATION_ARTIFACT_DIR
      ?? join(
        repoRoot,
        "docs",
        "evidence",
        artifactDate,
        gitSha,
        "brain-teach-session-bound",
      ),
  );
  return {
    validationStateDir,
    workspaceRoot,
    repetitions,
    artifactDir,
    gitSha,
  };
}

function embed(text: string): Float32Array {
  const normalized = text.toLowerCase();
  if (normalized.includes("pull request") || normalized.includes("gh pr create")) {
    return new Float32Array([1, 0, 0]);
  }
  if (normalized.includes("deployment") || normalized.includes("ci")) {
    return new Float32Array([0, 1, 0]);
  }
  return new Float32Array([0.5, 0.5, 0]);
}

function installFetchStub() {
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as { input?: string };
    const vector = Array.from(embed(body.input ?? ""));
    return {
      ok: true,
      json: async () => ({ data: [{ embedding: vector }] }),
      text: async () => JSON.stringify({ data: [{ embedding: vector }] }),
      status: 200,
    } as Response;
  }) as typeof fetch;
}

function buildApi(params: {
  dbPath: string;
  storePath: string;
  brainRoot: string;
  workspaceRoot: string;
}): {
  api: OpenClawPluginApi;
  getFactory: () => RegisteredEngineFactory;
  toolFactories: ToolFactory[];
} {
  let factory: RegisteredEngineFactory;
  const toolFactories: ToolFactory[] = [];

  const api = {
    id: "openclawbrain",
    name: "OpenClawBrain",
    source: params.workspaceRoot,
    config: {
      agents: {
        defaults: {
          workspace: params.workspaceRoot,
        },
      },
    },
    pluginConfig: {
      enabled: true,
      dbPath: params.dbPath,
      brainEnabled: true,
      brainRoot: params.brainRoot,
      brainWorkerMode: "in_process",
      brainTeacherEnabled: false,
      brainEmbeddingProvider: "openai",
      brainEmbeddingModel: "text-embedding-3-small",
      brainEmbeddingBaseUrl: "https://example.invalid/v1",
    },
    runtime: {
      subagent: {
        run: async () => undefined,
        waitForRun: async () => undefined,
        getSession: async () => undefined,
        deleteSession: async () => undefined,
      },
      config: {
        loadConfig: () => ({
          agents: {
            defaults: {
              workspace: params.workspaceRoot,
            },
          },
          session: {
            store: params.storePath,
          },
        }),
      },
      channel: {
        session: {
          resolveStorePath: () => params.storePath,
        },
      },
      modelAuth: {
        getApiKeyForModel: async () => ({ apiKey: "test-key" }),
        resolveApiKeyForProvider: async () => ({ apiKey: "test-key" }),
      },
    },
    logger: {
      info: () => {},
      warn: () => {},
      error: () => {},
      debug: () => {},
    },
    registerContextEngine: (_id: string, nextFactory: () => unknown) => {
      factory = nextFactory;
    },
    registerTool: (toolFactory: ToolFactory) => {
      toolFactories.push(toolFactory);
    },
    registerHook: () => {},
    registerHttpHandler: () => {},
    registerHttpRoute: () => {},
    registerChannel: () => {},
    registerGatewayMethod: () => {},
    registerCli: () => {},
    registerService: () => {},
    registerProvider: () => {},
    registerCommand: () => {},
    resolvePath: () => params.workspaceRoot,
    on: () => {},
  } as unknown as OpenClawPluginApi;

  return {
    api,
    getFactory: () => factory,
    toolFactories,
  };
}

function findBrainTeachToolFactory(toolFactories: ToolFactory[]): (ctx: OpenClawPluginToolContext) => {
  name: string;
  execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
} {
  for (const candidate of toolFactories) {
    if (typeof candidate !== "function") {
      continue;
    }
    const tool = candidate({ sessionKey: "agent:main:validation:seed" } as OpenClawPluginToolContext);
    if (tool && !Array.isArray(tool) && tool.name === "brain_teach") {
      return candidate as (ctx: OpenClawPluginToolContext) => {
        name: string;
        execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
      };
    }
  }
  throw new Error("Expected brain_teach tool factory to be registered.");
}

function extractText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .map((part) => {
      if (typeof part === "string") {
        return part;
      }
      if (part && typeof part === "object" && "text" in part && typeof (part as { text?: unknown }).text === "string") {
        return (part as { text: string }).text;
      }
      return "";
    })
    .join("\n");
}

function parseToolPayload(result: unknown): { nodeId: string; packVersion: number | null } {
  if (result && typeof result === "object" && "details" in result) {
    const details = (result as { details?: unknown }).details;
    if (details && typeof details === "object") {
      return {
        nodeId: String((details as { nodeId?: unknown }).nodeId ?? ""),
        packVersion:
          typeof (details as { packVersion?: unknown }).packVersion === "number"
            ? ((details as { packVersion: number }).packVersion)
            : null,
      };
    }
  }
  throw new Error(`Unexpected brain_teach tool result: ${JSON.stringify(result)}`);
}

function readLatestTeachEvidence(brainRoot: string): {
  episodeId: string | null;
  correctedEpisodeId: string | null;
  extractor: string | null;
  via: string | null;
} {
  const db = new DatabaseSync(join(brainRoot, "state.db"));
  try {
    const row = db.prepare(`
      SELECT episode_id AS episodeId, metadata
      FROM brain_evidence
      WHERE kind = 'teach_correction'
      ORDER BY created_at DESC
      LIMIT 1
    `).get() as { episodeId?: string; metadata?: string } | undefined;
    if (!row) {
      return {
        episodeId: null,
        correctedEpisodeId: null,
        extractor: null,
        via: null,
      };
    }
    const metadata = row.metadata ? JSON.parse(row.metadata) as Record<string, unknown> : {};
    return {
      episodeId: row.episodeId ?? null,
      correctedEpisodeId: typeof metadata.correctedEpisodeId === "string" ? metadata.correctedEpisodeId : null,
      extractor: typeof metadata.extractor === "string" ? metadata.extractor : null,
      via: typeof metadata.via === "string" ? metadata.via : null,
    };
  } finally {
    db.close();
  }
}

async function runIteration(params: {
  iteration: number;
  validationStateDir: string;
  workspaceRoot: string;
}): Promise<{ run: RunSummary; trace: Record<string, unknown> | null; status: Record<string, unknown> }> {
  const iterationName = `run-${String(params.iteration).padStart(2, "0")}`;
  const iterationRoot = join(params.validationStateDir, "brain-teach-session-bound", iterationName);
  rmSync(iterationRoot, { recursive: true, force: true });
  mkdirSync(iterationRoot, { recursive: true });

  const dbPath = join(iterationRoot, "lcm.db");
  const brainRoot = join(iterationRoot, "openclawbrain");
  const storePath = join(iterationRoot, "session-store.json");
  const sessionKey = "agent:main:validation:brain-teach-session-bound";
  const sessionId = "session-brain-teach-session-bound";

  writeFileSync(
    storePath,
    JSON.stringify({
      [sessionKey]: { sessionId },
    }),
    "utf8",
  );

  const { api, getFactory, toolFactories } = buildApi({
    dbPath,
    storePath,
    brainRoot,
    workspaceRoot: params.workspaceRoot,
  });

  lcmPlugin.register(api);
  const factory = getFactory();
  if (!factory) {
    throw new Error("Expected context engine factory to be registered.");
  }

  const engine = factory() as {
    ingest: (params: { sessionId: string; message: AgentMessage }) => Promise<void>;
    assemble: (params: {
      sessionId: string;
      messages: AgentMessage[];
      tokenBudget: number;
    }) => Promise<{ messages: AgentMessage[] }>;
    getConversationStore: () => {
      getConversationBySessionId: (id: string) => Promise<{ conversationId: number } | null>;
    };
    getBrainService: () => {
      init: (params: { workspaceRoot: string; embedFn?: (text: string) => Promise<Float32Array> }) => Promise<string>;
      query: (params: {
        conversationId: number;
        queryText: string;
        budgetChars: number;
        queryEmbedding: Float32Array;
      }) => Promise<{ episode: { id: string } }>;
      status: () => Promise<Record<string, unknown>>;
      getTrace: (traceId?: string) => Promise<Record<string, unknown> | null>;
    } | null;
  };

  const brainService = engine.getBrainService();
  if (!brainService) {
    throw new Error("Expected brain runtime service to be available.");
  }

  await brainService.init({
    workspaceRoot: params.workspaceRoot,
    embedFn: async (text) => embed(text),
  });

  await engine.ingest({
    sessionId,
    message: {
      role: "user",
      content: "Deployment failed again.",
    } as AgentMessage,
  });

  const conversation = await engine.getConversationStore().getConversationBySessionId(sessionId);
  if (!conversation) {
    throw new Error("Expected conversation to exist after ingest.");
  }

  const warmup = await brainService.query({
    conversationId: conversation.conversationId,
    queryText: "deployment failed",
    budgetChars: 4_000,
    queryEmbedding: embed("deployment ci"),
  });

  const brainTeachFactory = findBrainTeachToolFactory(toolFactories);
  const brainTeachTool = brainTeachFactory({ sessionKey } as OpenClawPluginToolContext);
  const toolResult = await brainTeachTool.execute(`call_${iterationName}`, {
    instruction: "For deployment failures, inspect CI logs before retrying.",
    kind: "correction",
    tags: ["deploy", "ci"],
  });
  const teachPayload = parseToolPayload(toolResult);

  const assembled = await engine.assemble({
    sessionId,
    messages: [
      {
        role: "user",
        content: "deployment failed again",
      } as AgentMessage,
    ],
    tokenBudget: 10_000,
  });

  const assembledText = assembled.messages.map((message) => extractText(message.content)).join("\n\n");
  const trace = await brainService.getTrace();
  const status = await brainService.status();
  const evidence = readLatestTeachEvidence(brainRoot);

  const run: RunSummary = {
    iteration: params.iteration,
    sessionKey,
    sessionId,
    conversationId: conversation.conversationId,
    warmupEpisodeId: warmup.episode.id,
    taughtNodeId: teachPayload.nodeId,
    teachPackVersion: teachPayload.packVersion,
    assembleMode:
      typeof (status.lastAssemblyDecision as { mode?: unknown } | undefined)?.mode === "string"
        ? ((status.lastAssemblyDecision as { mode: string }).mode)
        : null,
    assembledBrainHeaderVisible: assembledText.includes("OpenClawBrain retrieved context"),
    assembledCorrectionVisible: assembledText.includes("inspect CI logs before retrying"),
    traceIncludesTaughtNode: Array.isArray((trace as { firedNodes?: unknown[] } | null)?.firedNodes)
      ? ((trace as { firedNodes: unknown[] }).firedNodes.includes(teachPayload.nodeId))
      : false,
    traceId: typeof (trace as { traceId?: unknown } | null)?.traceId === "string"
      ? ((trace as { traceId: string }).traceId)
      : null,
    currentPackVersion: typeof status.currentPackVersion === "number" ? (status.currentPackVersion as number) : null,
    evidenceEpisodeId: evidence.episodeId,
    evidenceCorrectedEpisodeId: evidence.correctedEpisodeId,
    evidenceExtractor: evidence.extractor,
    evidenceVia: evidence.via,
    fingerprint: {
      conversationId: conversation.conversationId,
      teachPackVersion: teachPayload.packVersion,
      currentPackVersion: typeof status.currentPackVersion === "number" ? status.currentPackVersion : null,
      assembleMode:
        typeof (status.lastAssemblyDecision as { mode?: unknown } | undefined)?.mode === "string"
          ? ((status.lastAssemblyDecision as { mode: string }).mode)
          : null,
      assembledBrainHeaderVisible: assembledText.includes("OpenClawBrain retrieved context"),
      assembledCorrectionVisible: assembledText.includes("inspect CI logs before retrying"),
      traceIncludesTaughtNode: Array.isArray((trace as { firedNodes?: unknown[] } | null)?.firedNodes)
        ? ((trace as { firedNodes: unknown[] }).firedNodes.includes(teachPayload.nodeId))
        : false,
      evidenceEpisodeMatchesWarmup: evidence.episodeId === warmup.episode.id,
      evidenceCorrectedEpisodeMatchesWarmup: evidence.correctedEpisodeId === warmup.episode.id,
      evidenceExtractor: evidence.extractor,
      evidenceVia: evidence.via,
    },
  };

  if (
    run.assembleMode !== "use_brain"
    || !run.assembledBrainHeaderVisible
    || !run.assembledCorrectionVisible
    || !run.traceIncludesTaughtNode
    || run.evidenceEpisodeId !== warmup.episode.id
    || run.evidenceCorrectedEpisodeId !== warmup.episode.id
    || run.evidenceExtractor !== "brain_teach"
    || run.evidenceVia !== "brain_teach"
  ) {
    throw new Error(`Session-bound brain_teach assertion failed: ${JSON.stringify(run, null, 2)}`);
  }

  closeLcmConnection(dbPath);
  return { run, trace, status };
}

function writeArtifacts(params: {
  artifactDir: string;
  gitSha: string;
  workspaceRoot: string;
  validationStateDir: string;
  repetitions: number;
  runs: RunSummary[];
  lastTrace: Record<string, unknown> | null;
  lastStatus: Record<string, unknown> | null;
  error?: string;
}) {
  mkdirSync(params.artifactDir, { recursive: true });

  const uniqueFingerprints = Array.from(new Set(params.runs.map((run) => JSON.stringify(run.fingerprint))));
  const report = {
    ok: !params.error,
    gitSha: params.gitSha,
    workspaceRoot: params.workspaceRoot,
    validationStateDir: params.validationStateDir,
    repetitionsRequested: params.repetitions,
    repetitionsCompleted: params.runs.length,
    identicalPassFingerprintCount: uniqueFingerprints.length,
    identicalPasses: uniqueFingerprints.length === 1,
    runs: params.runs,
    ...(params.error ? { error: params.error } : {}),
  };

  writeFileSync(join(params.artifactDir, "validation-report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  writeFileSync(join(params.artifactDir, "trace.json"), `${JSON.stringify(params.lastTrace, null, 2)}\n`, "utf8");
  writeFileSync(join(params.artifactDir, "status.json"), `${JSON.stringify(params.lastStatus, null, 2)}\n`, "utf8");

  const summary = [
    "# Session-bound brain_teach validation summary",
    "",
    `- commit: \`${params.gitSha}\``,
    `- workspace: \`${params.workspaceRoot}\``,
    `- validation state dir: \`${params.validationStateDir}\``,
    `- repetitions requested: ${params.repetitions}`,
    `- repetitions completed: ${params.runs.length}`,
    `- identical pass fingerprints: ${uniqueFingerprints.length}`,
    `- acceptance: ${!params.error && uniqueFingerprints.length === 1 && params.runs.length === params.repetitions ? "PASS" : "FAIL"}`,
    "",
    "## Required proof",
    "",
    "- session-bound `brain_teach` tool resolves `ctx.sessionKey` to the correct conversation",
    "- teach action records `brain_teach` evidence against the warmup episode",
    "- follow-up runtime assembly uses brain retrieval and surfaces the taught correction",
    "- repeated runs are semantically identical at the asserted boundary",
  ];

  if (params.error) {
    summary.push("", "## Failure", "", `- ${params.error}`);
  }

  writeFileSync(join(params.artifactDir, "summary.md"), `${summary.join("\n")}\n`, "utf8");
}

async function main() {
  const { validationStateDir, workspaceRoot, repetitions, artifactDir, gitSha } = parseArgs(process.argv.slice(2));
  mkdirSync(validationStateDir, { recursive: true });
  mkdirSync(workspaceRoot, { recursive: true });
  mkdirSync(dirname(artifactDir), { recursive: true });

  writeFileSync(
    join(workspaceRoot, "DEPLOY.md"),
    "# Deploy\n\nCheck CI logs before retrying a deployment.\n",
    "utf8",
  );

  installFetchStub();

  const runs: RunSummary[] = [];
  let lastTrace: Record<string, unknown> | null = null;
  let lastStatus: Record<string, unknown> | null = null;

  try {
    for (let iteration = 1; iteration <= repetitions; iteration += 1) {
      const result = await runIteration({
        iteration,
        validationStateDir,
        workspaceRoot,
      });
      runs.push(result.run);
      lastTrace = result.trace;
      lastStatus = result.status;
    }

    const uniqueFingerprints = Array.from(new Set(runs.map((run) => JSON.stringify(run.fingerprint))));
    if (uniqueFingerprints.length !== 1) {
      throw new Error(`Expected identical pass fingerprints across ${repetitions} runs, saw ${uniqueFingerprints.length}.`);
    }

    writeArtifacts({
      artifactDir,
      gitSha,
      workspaceRoot,
      validationStateDir,
      repetitions,
      runs,
      lastTrace,
      lastStatus,
    });

    process.stdout.write(`${JSON.stringify({
      ok: true,
      gitSha,
      artifactDir,
      repetitions,
      identicalPasses: true,
      runs,
    }, null, 2)}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeArtifacts({
      artifactDir,
      gitSha,
      workspaceRoot,
      validationStateDir,
      repetitions,
      runs,
      lastTrace,
      lastStatus,
      error: message,
    });
    process.stderr.write(`${JSON.stringify({
      ok: false,
      error: message,
      artifactDir,
      runs,
    }, null, 2)}\n`);
    process.exit(1);
  }
}

void main();
