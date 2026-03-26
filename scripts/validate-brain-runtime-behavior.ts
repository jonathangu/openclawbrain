import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { BrainService } from "../src/brain-runtime/service.js";
import type { LcmDependencies } from "../src/types.js";

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
  return {
    workspaceRoot: resolve(args.get("workspace") ?? process.cwd()),
    brainRoot: resolve(args.get("brain-root") ?? join(process.cwd(), ".brain-runtime-validation")),
    lcmDbPath: resolve(args.get("lcm-db") ?? join(process.cwd(), ".brain-runtime-validation", "lcm.db")),
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

function createDeps(
  brainRoot: string,
  lcmDbPath: string,
  overrides?: Partial<NonNullable<LcmDependencies["config"]["brain"]>>,
): LcmDependencies {
  return {
    config: {
      enabled: true,
      databasePath: lcmDbPath,
      contextThreshold: 0.75,
      freshTailCount: 8,
      leafMinFanout: 8,
      condensedMinFanout: 4,
      condensedMinFanoutHard: 2,
      incrementalMaxDepth: 0,
      leafChunkTokens: 20000,
      leafTargetTokens: 1200,
      condensedTargetTokens: 2000,
      maxExpandTokens: 4000,
      largeFileTokenThreshold: 25000,
      largeFileSummaryProvider: "",
      largeFileSummaryModel: "",
      autocompactDisabled: false,
      timezone: "America/Los_Angeles",
      pruneHeartbeatOk: false,
      brain: {
        enabled: true,
        root: brainRoot,
        budgetFraction: 0.3,
        maxHops: 8,
        maxFanoutPerNode: 4,
        maxFrontierSize: 32,
        maxSeeds: 10,
        semanticThreshold: 0.1,
        servingTemperature: 0.1,
        learningTemperature: 1,
        learningRate: 0.01,
        baselineAlpha: 0.1,
        decayRate: 0.995,
        trainerIntervalMs: 10_000,
        workerMode: "in_process",
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 100,
        teacherEnabled: false,
        teacherProvider: "",
        teacherModel: "",
        mutationsEnabled: true,
        replayEpisodeCount: 100,
        minFiredPerQuery: 1,
        maxDormantPercent: 0.3,
        maxOrphanCount: 10,
        shadowMode: false,
        embeddingProvider: "openai",
        embeddingModel: "text-embedding-3-small",
        embeddingBaseUrl: "https://example.invalid/v1",
        ...overrides,
      },
    },
    complete: async () => ({ content: [{ type: "text", text: "{}" }] }),
    callGateway: async () => ({}),
    resolveModel: () => ({ provider: "openai", model: "gpt-5.4-mini" }),
    getApiKey: async () => "test-key",
    requireApiKey: async () => "test-key",
    parseAgentSessionKey: () => null,
    isSubagentSessionKey: () => false,
    normalizeAgentId: (id?: string) => id ?? "main",
    buildSubagentSystemPrompt: () => "",
    readLatestAssistantReply: () => undefined,
    resolveAgentDir: () => brainRoot,
    resolveSessionIdFromSessionKey: async () => undefined,
    agentLaneSubagent: "subagent",
    log: {
      info: () => {},
      warn: () => {},
      error: () => {},
      debug: () => {},
    },
  };
}

async function waitFor(predicate: () => Promise<boolean> | boolean, timeoutMs = 3_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`Condition not met within ${timeoutMs}ms`);
}

async function main() {
  const { workspaceRoot, brainRoot, lcmDbPath } = parseArgs(process.argv.slice(2));
  const dbDir = dirname(lcmDbPath);
  const fixtureWorkspace = join(workspaceRoot, ".openclawbrain-runtime-proof");
  const teachBrainRoot = join(brainRoot, "teach-runtime");
  const failOpenBrainRoot = join(brainRoot, "fail-open-runtime");
  const teachLcmDbPath = join(dbDir, "teach-runtime.lcm.db");
  const failOpenLcmDbPath = join(dbDir, "fail-open-runtime.lcm.db");

  mkdirSync(fixtureWorkspace, { recursive: true });

  writeFileSync(
    join(fixtureWorkspace, "DEPLOY.md"),
    "# Deploy\n\nCheck CI logs before retrying a deployment.\n",
    "utf8",
  );
  writeFileSync(
    join(fixtureWorkspace, "PLAYBOOK.md"),
    "# Pull Requests\n\nUse `gh pr create` for pull request workflows.\nIf the branch is not pushed yet, push it first and then open the PR.\n",
    "utf8",
  );

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

  const runtimeService = new BrainService({
    deps: createDeps(teachBrainRoot, teachLcmDbPath),
  });
  await runtimeService.init({
    workspaceRoot: fixtureWorkspace,
    embedFn: async (text) => embed(text),
  });

  await runtimeService.query({
    conversationId: 7,
    queryText: "deployment failed",
    budgetChars: 4000,
    queryEmbedding: embed("deployment ci"),
  });

  const taught = await runtimeService.teach({
    instruction: "For deployment errors, inspect CI logs before retrying.",
    conversationId: 7,
    kind: "correction",
  });
  const retrieved = await runtimeService.query({
    conversationId: 7,
    queryText: "deployment failed again",
    budgetChars: 4000,
    queryEmbedding: embed("deployment ci"),
  });
  const retrieveTrace = await runtimeService.getTrace();

  const failOpenService = new BrainService({
    deps: createDeps(failOpenBrainRoot, failOpenLcmDbPath, {
      workerMode: "child",
      trainerIntervalMs: 200,
      workerHeartbeatTimeoutMs: 150,
      workerRestartDelayMs: 5_000,
    }),
  });
  await failOpenService.init({
    workspaceRoot: fixtureWorkspace,
    embedFn: async (text) => embed(text),
  });

  failOpenService.startWorker();
  await waitFor(async () => {
    const status = await failOpenService.status();
    return Boolean(status.workerPid) && status.workerHealthy === true;
  });
  const beforeCrash = await failOpenService.query({
    conversationId: 42,
    queryText: "how do I open a pull request?",
    budgetChars: 4000,
    queryEmbedding: embed("gh pr create pull request"),
  });
  const childPid = (await failOpenService.status()).workerPid as number;
  process.kill(childPid, "SIGKILL");
  await waitFor(async () => Boolean((await failOpenService.status()).workerLastExit), 5_000);
  await new Promise((resolve) => setTimeout(resolve, 250));

  const statusAfterCrash = await failOpenService.status();
  const failOpenQuery = await failOpenService.query({
    conversationId: 42,
    queryText: "how do I open a pull request again?",
    budgetChars: 4000,
    queryEmbedding: embed("gh pr create pull request"),
  });

  failOpenService.stopWorker();
  await new Promise((resolve) => setTimeout(resolve, 50));

  const payload = {
    ok: true,
    teachRetrieval: {
      taughtNodeId: taught.nodeId,
      packVersion: taught.packVersion ?? null,
      retrievedCorrectionVisible:
        retrieved?.fired.some((node) => node.kind === "correction" && node.content.includes("inspect CI logs before retrying"))
        ?? false,
      traceIncludesTaughtNode: retrieveTrace?.firedNodes.includes(taught.nodeId) ?? false,
      retrievedPackVersion: retrieved?.episode.packVersion ?? null,
    },
    workerDownFailOpen: {
      servedBeforeCrash: beforeCrash !== null,
      servedPullRequestGuidanceBeforeCrash:
        beforeCrash?.fired.some((node) => node.content.includes("gh pr create")) ?? false,
      workerHealthyAfterCrash: statusAfterCrash.workerHealthy ?? null,
      workerLastExit: statusAfterCrash.workerLastExit ?? null,
      currentPackVersion: statusAfterCrash.currentPackVersion ?? null,
      servedAfterCrash: failOpenQuery !== null,
      servedPackVersion: failOpenQuery?.episode.packVersion ?? null,
      servedPullRequestGuidance:
        failOpenQuery?.fired.some((node) => node.content.includes("gh pr create")) ?? false,
    },
  };

  if (!payload.teachRetrieval.retrievedCorrectionVisible || !payload.teachRetrieval.traceIncludesTaughtNode) {
    throw new Error(`Teach retrieval assertion failed: ${JSON.stringify(payload.teachRetrieval)}`);
  }
  if (
    !payload.workerDownFailOpen.servedBeforeCrash
    || !payload.workerDownFailOpen.servedPullRequestGuidanceBeforeCrash
    || payload.workerDownFailOpen.workerHealthyAfterCrash !== false
    || !payload.workerDownFailOpen.workerLastExit
    || !payload.workerDownFailOpen.servedAfterCrash
    || !payload.workerDownFailOpen.servedPullRequestGuidance
  ) {
    throw new Error(`Worker-down fail-open assertion failed: ${JSON.stringify(payload.workerDownFailOpen)}`);
  }

  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

void main().catch((error) => {
  process.stderr.write(`${JSON.stringify({ ok: false, error: (error as Error).message }, null, 2)}\n`);
  process.exit(1);
});
