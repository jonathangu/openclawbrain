#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");
const args = new Set(process.argv.slice(2));
const keepTemp = args.has("--keep-temp");
const setupOnly = args.has("--setup-only");

const tempRoot = mkdtempSync(join(tmpdir(), "openclawbrain-validate-"));
const tempHome = join(tempRoot, "home");
const fixtureWorkspace = join(tempRoot, "workspace-fixture");
const lcmDbPath = join(tempHome, ".openclaw", "lcm.db");
const brainRoot = join(tempHome, ".openclaw", "openclawbrain");
const configPath = join(tempHome, ".openclaw", "openclaw.json");

mkdirSync(tempHome, { recursive: true });
mkdirSync(fixtureWorkspace, { recursive: true });

function cleanEnv(extra = {}) {
  const env = { ...process.env };
  for (const key of Object.keys(env)) {
    if (key.startsWith("OPENCLAW_")) {
      delete env[key];
    }
  }
  env.HOME = tempHome;
  env.LCM_DATABASE_PATH = lcmDbPath;
  env.OPENCLAWBRAIN_ROOT = brainRoot;
  env.OPENCLAWBRAIN_EMBEDDING_PROVIDER =
    process.env.OPENCLAWBRAIN_VALIDATION_EMBEDDING_PROVIDER
    ?? process.env.OPENCLAWBRAIN_EMBEDDING_PROVIDER
    ?? "openai";
  env.OPENCLAWBRAIN_EMBEDDING_MODEL =
    process.env.OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL
    ?? process.env.OPENCLAWBRAIN_EMBEDDING_MODEL
    ?? "";
  env.OPENCLAWBRAIN_EMBEDDING_BASE_URL =
    process.env.OPENCLAWBRAIN_VALIDATION_EMBEDDING_BASE_URL
    ?? process.env.OPENCLAWBRAIN_EMBEDDING_BASE_URL
    ?? "";
  return { ...env, ...extra };
}

function run(command, commandArgs, options = {}) {
  const env = cleanEnv(options.env);
  return execFileSync(command, commandArgs, {
    cwd: options.cwd ?? repoRoot,
    env,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...(typeof options.timeoutMs === "number" ? { timeout: options.timeoutMs } : {}),
  });
}

function runPassthrough(command, commandArgs, options = {}) {
  const env = cleanEnv(options.env);
  return execFileSync(command, commandArgs, {
    cwd: options.cwd ?? repoRoot,
    env,
    encoding: "utf8",
    stdio: ["ignore", "inherit", "inherit"],
  });
}

function extractJson(text) {
  try {
    return JSON.parse(text);
  } catch {}

  for (let index = text.lastIndexOf("{"); index >= 0; index = text.lastIndexOf("{", index - 1)) {
    const candidate = text.slice(index).trim();
    try {
      return JSON.parse(candidate);
    } catch {}
  }

  throw new Error(`Unable to parse JSON from command output:\n${text}`);
}

function writeFixtureWorkspace() {
  writeFileSync(
    join(fixtureWorkspace, "PLAYBOOK.md"),
    [
      "# Pull Requests",
      "",
      "Use `gh pr create` for pull request workflows.",
      "If the branch is not pushed yet, push it first and then open the PR.",
      "",
      "# Deployments",
      "",
      "Check CI logs before retrying a deployment.",
    ].join("\n"),
    "utf8",
  );

  writeFileSync(
    join(fixtureWorkspace, "RUNBOOK.md"),
    [
      "# Recovery",
      "",
      "If a deployment fails, inspect the CI logs before retrying.",
      "Prefer the most recent successful workflow when reconstructing the sequence.",
    ].join("\n"),
    "utf8",
  );
}

function updateConfig(options = {}) {
  const config = JSON.parse(readFileSync(configPath, "utf8"));
  config.plugins ??= {};
  config.plugins.slots ??= {};
  config.plugins.entries ??= {};
  config.plugins.entries.openclawbrain ??= {};
  config.plugins.entries.openclawbrain.enabled = true;
  config.plugins.entries.openclawbrain.config = {
    enabled: true,
  };
  config.plugins.slots.contextEngine = "openclawbrain";

  const validationModel = process.env.OPENCLAWBRAIN_VALIDATION_MODEL?.trim();
  if (validationModel) {
    config.agents ??= {};
    config.agents.defaults ??= {};
    config.agents.defaults.model = { primary: validationModel };

    const [providerId, ...modelParts] = validationModel.split("/");
    const modelId = modelParts.join("/").trim();
    if (providerId?.trim() === "ollama" && modelId) {
      config.models ??= {};
      config.models.providers ??= {};
      const providerKey = config.models.providers.ollama ? "ollama" : "ollama";
      const existing = config.models.providers[providerKey] ?? {};
      config.models.providers[providerKey] = {
        ...existing,
        api: "ollama",
        baseUrl: process.env.OPENCLAWBRAIN_VALIDATION_MODEL_BASE_URL?.trim() || "http://127.0.0.1:11434",
        apiKey: (existing.apiKey && String(existing.apiKey).trim()) || "ollama-local",
        models: Array.isArray(existing.models) && existing.models.length > 0
          ? existing.models
          : [
              {
                id: modelId,
                name: modelId,
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: 262144,
                maxTokens: 16384,
              },
            ],
      };
    }
  }

  writeFileSync(configPath, `${JSON.stringify(config, null, 2)}\n`, "utf8");
}

function collectStrings(value, out = []) {
  if (typeof value === "string") {
    out.push(value);
    return out;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectStrings(item, out);
    }
    return out;
  }
  if (value && typeof value === "object") {
    for (const entry of Object.values(value)) {
      collectStrings(entry, out);
    }
  }
  return out;
}

function maybeRunAgentChecks(report) {
  const validationModel = process.env.OPENCLAWBRAIN_VALIDATION_MODEL?.trim();
  const embeddingModel = cleanEnv().OPENCLAWBRAIN_EMBEDDING_MODEL;
  const agentTimeoutMs = Number.parseInt(process.env.OPENCLAWBRAIN_VALIDATION_AGENT_TIMEOUT_MS?.trim() || "120000", 10);
  const agentTimeoutSeconds = Math.max(1, Math.ceil(agentTimeoutMs / 1000));
  const agentCommandTimeoutMs = agentTimeoutMs + 15_000;
  if (!validationModel || !embeddingModel) {
    report.skipped.push({
      phase: "agent-routing",
      reason: !validationModel
        ? "OPENCLAWBRAIN_VALIDATION_MODEL is unset, so host-app local agent checks were not attempted."
        : "OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL is unset, so host-app local agent checks were not attempted.",
    });
    return;
  }

  function runAgentCheck({ to, message, extraEnv }) {
    return extractJson(run("openclaw", [
      "agent",
      "--local",
      "--to",
      to,
      "--message",
      message,
      "--json",
      "--timeout",
      String(agentTimeoutSeconds),
    ], {
      timeoutMs: agentCommandTimeoutMs,
      env: extraEnv,
    }));
  }

  function assertAgentCompleted(label, result) {
    if (result?.meta?.aborted) {
      throw new Error(`Validation harness expected ${label} to complete within ${agentTimeoutSeconds}s, but the embedded OpenClaw agent aborted/timed out.`);
    }
  }

  report.agent.recurrentQuery = runAgentCheck({
    to: "+15550001111",
    message: "How do I open a pull request again?",
  });
  assertAgentCompleted("recurrent host-agent query", report.agent.recurrentQuery);

  const recurrentStatus = extractJson(run("node", ["bin/openclawbrain.js", "status"]));
  const recurrentTrace = extractJson(run("node", ["bin/openclawbrain.js", "trace"]));
  report.assertions.recurrentQuery = {
    lastAssemblyMode: recurrentStatus.lastAssemblyDecision?.mode ?? null,
    traceId: recurrentTrace?.trace?.id ?? null,
    episodeId: recurrentTrace?.trace?.episodeId ?? null,
    workerMode: recurrentStatus.workerMode ?? null,
    workerPid: recurrentStatus.workerPid ?? null,
    workerHealthy: recurrentStatus.workerHealthy ?? null,
    workerLastHeartbeatAt: recurrentStatus.workerLastHeartbeatAt ?? null,
    aborted: report.agent.recurrentQuery?.meta?.aborted ?? null,
  };

  if ((recurrentStatus.workerMode ?? null) === "child") {
    if (!recurrentStatus.workerPid) {
      throw new Error("Validation harness expected a child worker PID after recurrent host-agent query, but none was reported.");
    }
    if (recurrentStatus.workerHealthy !== true) {
      throw new Error("Validation harness expected the child worker to report healthy after recurrent host-agent query.");
    }
  }

  report.agent.shortLookup = runAgentCheck({
    to: "+15550002222",
    message: "open PLAYBOOK.md",
  });
  assertAgentCompleted("short lookup host-agent query", report.agent.shortLookup);
  const shortStatus = extractJson(run("node", ["bin/openclawbrain.js", "status"]));
  report.assertions.shortLookup = {
    lastAssemblyMode: shortStatus.lastAssemblyDecision?.mode ?? null,
    aborted: report.agent.shortLookup?.meta?.aborted ?? null,
  };

  report.agent.shadowQuery = runAgentCheck({
    to: "+15550003333",
    message: "How do I open a pull request again?",
    extraEnv: { OPENCLAWBRAIN_SHADOW_MODE: "true" },
  });
  assertAgentCompleted("shadow-mode host-agent query", report.agent.shadowQuery);
  const shadowStatus = extractJson(run("node", ["bin/openclawbrain.js", "status"], { env: { OPENCLAWBRAIN_SHADOW_MODE: "true" } }));
  const shadowTrace = extractJson(run("node", ["bin/openclawbrain.js", "trace"], { env: { OPENCLAWBRAIN_SHADOW_MODE: "true" } }));
  const shadowVisibleText = collectStrings(report.agent.shadowQuery).join("\n");
  const injectedContextVisible =
    shadowVisibleText.includes("OpenClawBrain retrieved context.")
    || shadowVisibleText.includes("## Correction Cards")
    || shadowVisibleText.includes("Use gh pr create for pull requests.");
  report.assertions.shadowMode = {
    shadowMode: shadowStatus.shadowMode ?? null,
    lastAssemblyMode: shadowStatus.lastAssemblyDecision?.mode ?? null,
    traceId: shadowTrace?.trace?.id ?? null,
    episodeId: shadowTrace?.trace?.episodeId ?? null,
    injectedContextVisible,
    aborted: report.agent.shadowQuery?.meta?.aborted ?? null,
  };

  if (shadowStatus.lastAssemblyDecision?.mode !== "shadow") {
    throw new Error(`Validation harness expected shadow mode decision, got ${shadowStatus.lastAssemblyDecision?.mode ?? "null"}.`);
  }
  if (!shadowTrace?.trace?.id || !shadowTrace?.trace?.episodeId) {
    throw new Error("Validation harness expected shadow mode to record a trace and episode id.");
  }
  if (injectedContextVisible) {
    throw new Error("Validation harness expected shadow mode to avoid visible brain-context injection in the host response.");
  }

  report.skipped.push(
    {
      phase: "brain-teach",
      reason: "Phase-1 harness scaffold still needs a deterministic host-surface path for brain_teach assertion wiring.",
    },
    {
      phase: "worker-down",
      reason: "Phase-1 harness scaffold still needs an explicit worker-stop assertion against last promoted pack serving.",
    },
  );
}

const report = {
  tempRoot,
  tempHome,
  fixtureWorkspace,
  configPath,
  setup: {},
  init: null,
  doctor: null,
  status: null,
  runtime: null,
  agent: {},
  assertions: {},
  skipped: [],
};

try {
  writeFixtureWorkspace();

  runPassthrough("openclaw", ["plugins", "install", "--link", repoRoot]);
  updateConfig();

  report.setup = {
    linkedPlugin: repoRoot,
    lcmDbPath,
    brainRoot,
    contextEngineSlot: "openclawbrain",
    validationModel: process.env.OPENCLAWBRAIN_VALIDATION_MODEL?.trim() || null,
    workerMode: process.env.OPENCLAWBRAIN_VALIDATION_WORKER_MODE?.trim() || "child",
    embeddingProvider: cleanEnv().OPENCLAWBRAIN_EMBEDDING_PROVIDER,
    embeddingModel: cleanEnv().OPENCLAWBRAIN_EMBEDDING_MODEL || null,
    agentTimeoutMs: Number.parseInt(process.env.OPENCLAWBRAIN_VALIDATION_AGENT_TIMEOUT_MS?.trim() || "120000", 10),
  };

  if (setupOnly) {
    report.skipped.push({
      phase: "init-and-agent-checks",
      reason: "--setup-only was requested.",
    });
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    process.exit(0);
  }

  report.runtime = extractJson(run("pnpm", [
    "exec",
    "tsx",
    "scripts/validate-brain-runtime-behavior.ts",
    "--workspace",
    fixtureWorkspace,
    "--brain-root",
    brainRoot,
    "--lcm-db",
    lcmDbPath,
  ]));
  report.assertions.teachRetrieval = report.runtime.teachRetrieval;
  report.assertions.workerDownFailOpen = report.runtime.workerDownFailOpen;

  if (cleanEnv().OPENCLAWBRAIN_EMBEDDING_MODEL) {
    report.init = extractJson(run("node", ["bin/openclawbrain.js", "init", fixtureWorkspace]));
    report.status = extractJson(run("node", ["bin/openclawbrain.js", "status"]));
    report.doctor = extractJson(run("node", ["bin/openclawbrain.js", "doctor"]));
  } else {
    report.skipped.push({
      phase: "cli-init",
      reason: "OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL is unset, so CLI init/status/doctor checks were not attempted in the disposable harness.",
    });
  }

  maybeRunAgentChecks(report);

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} catch (error) {
  const payload = {
    ok: false,
    error: (error instanceof Error ? error.message : String(error)),
    report,
  };
  process.stderr.write(`${JSON.stringify(payload, null, 2)}\n`);
  process.exitCode = 1;
} finally {
  if (!keepTemp) {
    rmSync(tempRoot, { recursive: true, force: true });
  }
}
