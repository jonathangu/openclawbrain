import { mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { BrainAssemblerExtension } from "../../../src/brain-runtime/assembler-extension.js";
import { BrainService } from "../../../src/brain-runtime/service.js";
import { proposeUserCorrectionFast } from "../../../src/brain-runtime/user-memory-proposals.js";
import type { LcmDependencies } from "../../../src/types.js";
import type { ChatTurn } from "../src/agent/client.js";
import type { RetrievedItem } from "../src/types.js";

const CONVERSATION_ID = 1;
const DEFAULT_BUDGET_CHARS = 4_000;
const EMBED_DIMS = 96;

function readBoolEnv(...names: string[]): boolean {
  for (const name of names) {
    const raw = process.env[name]?.trim().toLowerCase();
    if (!raw) {
      continue;
    }
    if (raw === "1" || raw === "true" || raw === "yes" || raw === "on") {
      return true;
    }
    if (raw === "0" || raw === "false" || raw === "no" || raw === "off") {
      return false;
    }
  }
  return false;
}

function hashedEmbedding(text: string, dims = EMBED_DIMS): Float32Array {
  const vector = new Float32Array(dims);
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  if (tokens.length === 0) {
    return vector;
  }

  for (const token of tokens) {
    let hash = 2166136261;
    for (let i = 0; i < token.length; i += 1) {
      hash ^= token.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    const index = Math.abs(hash) % dims;
    const sign = (hash & 1) === 0 ? 1 : -1;
    vector[index] = (vector[index] ?? 0) + sign;
  }

  let norm = 0;
  for (const value of vector) {
    norm += value * value;
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vector.length; i += 1) {
      vector[i] = (vector[i] ?? 0) / norm;
    }
  }
  return vector;
}

function createDeps(brainRoot: string): LcmDependencies {
  const directAnswerNoFire = readBoolEnv(
    "OCB_BRAIN_DIRECT_ANSWER_NO_FIRE",
    "OPENCLAWBRAIN_DIRECT_ANSWER_NO_FIRE",
  );
  const suppressSyntheticWorkspaceSentinel = readBoolEnv(
    "OCB_BRAIN_SUPPRESS_SYNTHETIC_WORKSPACE_SENTINEL",
    "OPENCLAWBRAIN_SUPPRESS_SYNTHETIC_WORKSPACE_SENTINEL",
  );

  return {
    config: {
      enabled: true,
      databasePath: join(brainRoot, "lcm.db"),
      contextThreshold: 0.75,
      freshTailCount: 8,
      leafMinFanout: 8,
      condensedMinFanout: 4,
      condensedMinFanoutHard: 2,
      incrementalMaxDepth: 0,
      leafChunkTokens: 20_000,
      leafTargetTokens: 1_200,
      condensedTargetTokens: 2_000,
      maxExpandTokens: 4_000,
      largeFileTokenThreshold: 25_000,
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
        persistRawSurfaces: false,
        directAnswerNoFire,
        suppressSyntheticWorkspaceSentinel,
        teacherProvider: "",
        teacherModel: "",
        autoUserCorrectionsEnabled: false,
        autoUserCorrectionsProvider: "",
        autoUserCorrectionsModel: "",
        autoUserCorrectionsMinConfidence: 0.8,
        mutationsEnabled: true,
        replayEpisodeCount: 100,
        minFiredPerQuery: 1,
        maxDormantPercent: 0.3,
        maxOrphanCount: 10,
        shadowMode: false,
        embeddingProvider: "openai",
        embeddingModel: "text-embedding-3-small",
        embeddingBaseUrl: "https://example.invalid/v1",
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

function recentMessages(history: ChatTurn[], endExclusive: number): Array<{ role: "user" | "assistant"; content: string }> {
  return history
    .slice(Math.max(0, endExclusive - 6), endExclusive)
    .filter((turn): turn is ChatTurn & { role: "user" | "assistant" } => turn.role === "user" || turn.role === "assistant")
    .map((turn) => ({ role: turn.role, content: turn.content }));
}

function extractCorrections(history: ChatTurn[]) {
  const found = new Map<string, { instruction: string; sourceQuote: string; messageId: number; ageSeconds: number }>();

  for (let i = 0; i < history.length; i += 1) {
    const turn = history[i];
    if (!turn || turn.role !== "user") {
      continue;
    }

    const proposal = proposeUserCorrectionFast({
      conversationId: CONVERSATION_ID,
      messageId: i + 1,
      userText: turn.content,
      recentMessages: recentMessages(history, i),
      recentSummaries: [],
    });

    if (proposal.kind !== "explicit_correction") {
      continue;
    }

    found.set(proposal.canonicalInstruction, {
      instruction: proposal.canonicalInstruction,
      sourceQuote: turn.content,
      messageId: i + 1,
      ageSeconds: (history.length - i) * 60,
    });
  }

  return [...found.values()];
}

export async function createOcbAdapter(): Promise<{
  route(
    history: ChatTurn[],
    query: string,
  ): Promise<{
    fire: boolean;
    retrieved: RetrievedItem[];
    injected_text: string;
    prompt_turns?: ChatTurn[];
    gate_score: number | null;
    gate_threshold: number | null;
  }>;
}> {
  const embed = async (text: string) => hashedEmbedding(text);

  return {
    async route(history: ChatTurn[], query: string) {
      const corrections = extractCorrections(history);
      if (corrections.length === 0) {
        return {
          fire: false,
          retrieved: [],
          injected_text: "",
          gate_score: null,
          gate_threshold: null,
        };
      }

      const root = await mkHarnessRoot();
      try {
        const workspaceRoot = join(root, "workspace");
        const brainRoot = join(root, "brain-state");
        await mkdir(workspaceRoot, { recursive: true });
        await mkdir(brainRoot, { recursive: true });
        await writeFile(
          join(workspaceRoot, "HARNESS.md"),
          "# Synthetic eval workspace\n\nThis workspace exists only for correction-persistence harness runs.\n",
          "utf8",
        );

        const service = new BrainService({ deps: createDeps(brainRoot) });
        const extension = new BrainAssemblerExtension(service);
        await service.init({ workspaceRoot, embedFn: embed });
        (service as unknown as { embeddingClient: (text: string) => Promise<Float32Array> }).embeddingClient = embed;

        for (const correction of corrections) {
          await service.teachUserCorrection({
            canonicalInstruction: correction.instruction,
            sourceQuote: correction.sourceQuote,
            conversationId: CONVERSATION_ID,
            sourceMessageId: correction.messageId,
            tags: ["eval", "correction-persistence", "synthetic"],
            via: "correction_persistence_ablation_ocb_adapter",
          });
        }

        const result = await service.query({
          conversationId: CONVERSATION_ID,
          queryText: query,
          budgetChars: Number.parseInt(process.env.OCB_BUDGET_CHARS ?? "", 10) || DEFAULT_BUDGET_CHARS,
          queryEmbedding: await embed(query),
        });

        const retrieved: RetrievedItem[] = (result?.fired ?? []).map((node, index) => ({
          source_id: node.nodeId,
          content: node.content,
          score: Number((1 - (index * 0.05)).toFixed(4)),
          age_seconds: corrections.find((correction) => correction.instruction === node.content)?.ageSeconds ?? 0,
        }));

        const assembled = await extension.augmentAssembly({
          conversationId: CONVERSATION_ID,
          tokenBudget: 4096,
          maxContextChars: 4000,
          assembled: {
            messages: [{ role: "user", content: query }],
            estimatedTokens: 0,
            stats: {
              rawMessageCount: 1,
              summaryCount: 0,
              totalContextItems: 1,
            },
          },
          liveMessages: [{ role: "user", content: query }],
        });

        const candidateInjectedText = typeof assembled.messages[0]?.content === "string"
          ? assembled.messages[0].content
          : "";
        const injected_text = assembled.brainDecision?.mode === "use_brain" && assembled.messages.length > 1
          ? candidateInjectedText
          : (retrieved.length > 0
            ? "The user previously corrected or stated a preference:\n" + retrieved.map((item) => `- ${item.content}`).join("\n")
            : "");

        const prompt_turns = assembled.brainDecision?.mode === "use_brain"
          ? assembled.messages
              .filter((message): message is { role: "system" | "user" | "assistant"; content: string } =>
                (message.role === "system" || message.role === "user" || message.role === "assistant") && typeof message.content === "string",
              )
              .map((message) => ({ role: message.role, content: message.content }))
          : undefined;

        return {
          fire: retrieved.length > 0,
          retrieved,
          injected_text,
          prompt_turns,
          gate_score: null,
          gate_threshold: null,
        };
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    },
  };
}

async function mkHarnessRoot(): Promise<string> {
  const { mkdtemp } = await import("node:fs/promises");
  return mkdtemp(join(tmpdir(), "ocb-correction-persistence-"));
}
