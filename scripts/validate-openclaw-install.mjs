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
const validationRecordFile = join(tempRoot, "validation-assemble.jsonl");

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
  env.OPENCLAWBRAIN_VALIDATION_RECORD_FILE = validationRecordFile;
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

function readValidationRecords() {
  if (!readFileSync || !validationRecordFile) {
    return [];
  }
  try {
    const text = readFileSync(validationRecordFile, "utf8").trim();
    if (!text) {
      return [];
    }
    return text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch {
    return [];
  }
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

  function runStatus(extraEnv) {
    return extractJson(run("node", ["bin/openclawbrain.js", "status"], { env: extraEnv }));
  }

  function runTrace(extraEnv) {
    return extractJson(run("node", ["bin/openclawbrain.js", "trace"], { env: extraEnv }));
  }

  function sleepMs(ms) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
  }

  function normalizeValidationText(value) {
    return typeof value === "string" ? value.trim().toLowerCase() : null;
  }

  function readValidationRecordAfter(startIndex, expectedQueryText) {
    const records = readValidationRecords();
    const normalizedExpected = normalizeValidationText(expectedQueryText);
    const newRecords = records.slice(startIndex);
    const record = normalizedExpected
      ? [...newRecords].reverse().find((entry) => normalizeValidationText(entry?.queryText) === normalizedExpected) ?? null
      : (newRecords.at(-1) ?? null);
    return {
      records,
      newRecords,
      record,
    };
  }

  function runPrimedAgentCheck({ to, primerMessage = "hi", message, extraEnv }) {
    const primer = runAgentCheck({ to, message: primerMessage, extraEnv });
    assertAgentCompleted(`primer host-agent query (${to})`, primer);
    const result = runAgentCheck({ to, message, extraEnv });
    return {
      primer,
      result,
    };
  }

  function assertAgentCompleted(label, result) {
    if (result?.meta?.aborted) {
      throw new Error(`Validation harness expected ${label} to complete within ${agentTimeoutSeconds}s, but the embedded OpenClaw agent aborted/timed out.`);
    }
  }

  const baselineStatus = runStatus();
  const recurrentRecordStart = readValidationRecords().length;
  const recurrentCheck = runPrimedAgentCheck({
    to: "+15550001111",
    primerMessage: "How do I open a pull request again?",
    message: "Please answer my previous question directly.",
  });
  report.agent.recurrentPrimer = recurrentCheck.primer;
  report.agent.recurrentQuery = recurrentCheck.result;
  assertAgentCompleted("recurrent host-agent query", report.agent.recurrentQuery);

  const recurrentStatus = runStatus();
  const recurrentValidation = readValidationRecordAfter(
    recurrentRecordStart,
    "How do I open a pull request again?",
  );
  report.assertions.recurrentQuery = {
    validationRecordCountBefore: recurrentRecordStart,
    validationRecordCountAfter: recurrentValidation.records.length,
    mode: recurrentValidation.record?.mode ?? null,
    traceId: recurrentValidation.record?.traceId ?? null,
    episodeId: recurrentValidation.record?.episodeId ?? null,
    traceQueryText: recurrentValidation.record?.queryText ?? null,
    workerMode: recurrentStatus.workerMode ?? null,
    workerPid: recurrentStatus.workerPid ?? null,
    workerHealthy: recurrentStatus.workerHealthy ?? null,
    workerLastHeartbeatAt: recurrentStatus.workerLastHeartbeatAt ?? null,
    currentPackVersion: recurrentStatus.currentPackVersion ?? null,
    aborted: report.agent.recurrentQuery?.meta?.aborted ?? null,
  };

  if (!recurrentValidation.record) {
    throw new Error("Validation harness expected recurrent host-agent query to emit a host-surface validation record, but none was captured.");
  }
  if (recurrentValidation.record?.mode !== "use_brain") {
    throw new Error(`Validation harness expected recurrent host-agent query to record use_brain, got ${recurrentValidation.record?.mode ?? "null"}.`);
  }
  if (normalizeValidationText(recurrentValidation.record?.queryText) !== "how do i open a pull request again?") {
    throw new Error(`Validation harness expected recurrent host-agent validation query text to match the host prompt, got ${JSON.stringify(recurrentValidation.record?.queryText ?? null)}.`);
  }

  if ((recurrentStatus.workerMode ?? null) === "child") {
    if (!recurrentStatus.workerPid) {
      throw new Error("Validation harness expected a child worker PID after recurrent host-agent query, but none was reported.");
    }
    // Note: child worker exits after each agent run completes - this is expected behavior.
    // We verify pack promotion via currentPackVersion and brain usage via validation records.
  }

  const shortLookupRecordStart = readValidationRecords().length;
  const shortLookupCheck = runPrimedAgentCheck({
    to: "+15550002222",
    primerMessage: "open PLAYBOOK.md",
    message: "Please answer my previous question directly.",
  });
  report.agent.shortLookupPrimer = shortLookupCheck.primer;
  report.agent.shortLookup = shortLookupCheck.result;
  assertAgentCompleted("short lookup host-agent query", report.agent.shortLookup);
  const shortStatus = runStatus();
  const shortValidation = readValidationRecordAfter(shortLookupRecordStart, "open PLAYBOOK.md");
  report.assertions.shortLookup = {
    validationRecordCountBefore: shortLookupRecordStart,
    validationRecordCountAfter: shortValidation.records.length,
    mode: shortValidation.record?.mode ?? null,
    traceId: shortValidation.record?.traceId ?? null,
    lastAssemblyMode: shortStatus.lastAssemblyDecision?.mode ?? null,
    aborted: report.agent.shortLookup?.meta?.aborted ?? null,
  };

  if (!shortValidation.record) {
    throw new Error("Validation harness expected short lookup host-agent query to emit a host-surface validation record, but none was captured.");
  }
  if (shortValidation.record?.mode !== "skip_short_static_lookup") {
    throw new Error(`Validation harness expected short lookup host-agent query to bypass with skip_short_static_lookup, got ${shortValidation.record?.mode ?? "null"}.`);
  }
  if (shortValidation.record?.traceId) {
    throw new Error(`Validation harness expected short lookup host-agent query to bypass brain retrieval without creating a trace, got ${JSON.stringify(shortValidation.record?.traceId)}.`);
  }

  const shadowRecordStart = readValidationRecords().length;
  const shadowCheck = runPrimedAgentCheck({
    to: "+15550003333",
    primerMessage: "How do I open a pull request again?",
    message: "Please answer my previous question directly.",
    extraEnv: { OPENCLAWBRAIN_SHADOW_MODE: "true" },
  });
  report.agent.shadowPrimer = shadowCheck.primer;
  report.agent.shadowQuery = shadowCheck.result;
  assertAgentCompleted("shadow-mode host-agent query", report.agent.shadowQuery);
  const shadowStatus = runStatus({ OPENCLAWBRAIN_SHADOW_MODE: "true" });
  const shadowValidation = readValidationRecordAfter(
    shadowRecordStart,
    "How do I open a pull request again?",
  );
  const shadowVisibleText = collectStrings(report.agent.shadowQuery).join("\n");
  const injectedContextVisible =
    shadowVisibleText.includes("OpenClawBrain retrieved context.")
    || shadowVisibleText.includes("## Correction Cards")
    || shadowVisibleText.includes("Use gh pr create for pull requests.");
  report.assertions.shadowMode = {
    shadowMode: shadowStatus.shadowMode ?? null,
    validationRecordCountBefore: shadowRecordStart,
    validationRecordCountAfter: shadowValidation.records.length,
    mode: shadowValidation.record?.mode ?? null,
    traceId: shadowValidation.record?.traceId ?? null,
    episodeId: shadowValidation.record?.episodeId ?? null,
    traceQueryText: shadowValidation.record?.queryText ?? null,
    injectedContextVisible,
    aborted: report.agent.shadowQuery?.meta?.aborted ?? null,
  };

  if (shadowStatus.shadowMode !== true) {
    throw new Error(`Validation harness expected shadow mode to be enabled for the host-agent query, got ${shadowStatus.shadowMode ?? "null"}.`);
  }
  if (!shadowValidation.record) {
    throw new Error("Validation harness expected shadow-mode host-agent query to emit a host-surface validation record, but none was captured.");
  }
  if (shadowValidation.record?.mode !== "shadow") {
    throw new Error(`Validation harness expected shadow-mode host-agent query to record shadow, got ${shadowValidation.record?.mode ?? "null"}.`);
  }
  if (!shadowValidation.record?.traceId || !shadowValidation.record?.episodeId) {
    throw new Error("Validation harness expected shadow mode to record a trace and episode id.");
  }
  if (normalizeValidationText(shadowValidation.record?.queryText) !== "how do i open a pull request again?") {
    throw new Error(`Validation harness expected shadow-mode host-agent validation query text to match the host prompt, got ${JSON.stringify(shadowValidation.record?.queryText ?? null)}.`);
  }
  if (injectedContextVisible) {
    throw new Error("Validation harness expected shadow mode to avoid visible brain-context injection in the host response.");
  }

  const noEmbeddingRecordStart = readValidationRecords().length;
  const noEmbeddingCheck = runPrimedAgentCheck({
    to: "+15550004444",
    primerMessage: "How do I open a pull request again?",
    message: "Please answer my previous question directly.",
    extraEnv: { OPENCLAWBRAIN_EMBEDDING_MODEL: "" },
  });
  report.agent.noEmbeddingPrimer = noEmbeddingCheck.primer;
  report.agent.noEmbeddingQuery = noEmbeddingCheck.result;
  assertAgentCompleted("no-embedding host-agent query", report.agent.noEmbeddingQuery);
  const noEmbeddingStatus = runStatus({ OPENCLAWBRAIN_EMBEDDING_MODEL: "" });
  const noEmbeddingValidation = readValidationRecordAfter(
    noEmbeddingRecordStart,
    "How do I open a pull request again?",
  );
  report.assertions.noEmbedding = {
    validationRecordCountBefore: noEmbeddingRecordStart,
    validationRecordCountAfter: noEmbeddingValidation.records.length,
    mode: noEmbeddingValidation.record?.mode ?? null,
    traceId: noEmbeddingValidation.record?.traceId ?? null,
    lastAssemblyMode: noEmbeddingStatus.lastAssemblyDecision?.mode ?? null,
    aborted: report.agent.noEmbeddingQuery?.meta?.aborted ?? null,
  };

  if (!noEmbeddingValidation.record) {
    throw new Error("Validation harness expected no-embedding host-agent query to emit a host-surface validation record, but none was captured.");
  }
  if (noEmbeddingValidation.record?.mode !== "skip_no_embedding") {
    throw new Error(`Validation harness expected no-embedding host-agent query to bypass with skip_no_embedding, got ${noEmbeddingValidation.record?.mode ?? "null"}.`);
  }
  if (noEmbeddingValidation.record?.traceId) {
    throw new Error(`Validation harness expected no-embedding host-agent query to bypass brain retrieval without creating a trace, got ${JSON.stringify(noEmbeddingValidation.record?.traceId)}.`);
  }

  const uninitializedBrainRoot = join(tempHome, ".openclaw", "openclawbrain-uninitialized");
  mkdirSync(uninitializedBrainRoot, { recursive: true });
  const uninitializedRecordStart = readValidationRecords().length;
  const uninitializedCheck = runPrimedAgentCheck({
    to: "+15550005555",
    primerMessage: "How do I open a pull request again?",
    message: "Please answer my previous question directly.",
    extraEnv: { OPENCLAWBRAIN_ROOT: uninitializedBrainRoot },
  });
  report.agent.uninitializedPrimer = uninitializedCheck.primer;
  report.agent.uninitializedQuery = uninitializedCheck.result;
  assertAgentCompleted("uninitialized host-agent query", report.agent.uninitializedQuery);
  const uninitializedStatus = runStatus({ OPENCLAWBRAIN_ROOT: uninitializedBrainRoot });
  const uninitializedValidation = readValidationRecordAfter(
    uninitializedRecordStart,
    "How do I open a pull request again?",
  );
  report.assertions.uninitialized = {
    validationRecordCountBefore: uninitializedRecordStart,
    validationRecordCountAfter: uninitializedValidation.records.length,
    mode: uninitializedValidation.record?.mode ?? null,
    traceId: uninitializedValidation.record?.traceId ?? null,
    lastAssemblyMode: uninitializedStatus.lastAssemblyDecision?.mode ?? null,
    aborted: report.agent.uninitializedQuery?.meta?.aborted ?? null,
  };

  if (!uninitializedValidation.record) {
    throw new Error("Validation harness expected uninitialized host-agent query to emit a host-surface validation record, but none was captured.");
  }
  if (uninitializedValidation.record?.mode !== "skip_uninitialized") {
    throw new Error(`Validation harness expected uninitialized host-agent query to bypass with skip_uninitialized, got ${uninitializedValidation.record?.mode ?? "null"}.`);
  }
  if (uninitializedValidation.record?.traceId) {
    throw new Error(`Validation harness expected uninitialized host-agent query to bypass brain retrieval without creating a trace, got ${JSON.stringify(uninitializedValidation.record?.traceId)}.`);
  }

  if ((recurrentStatus.workerMode ?? null) === "child" && recurrentStatus.workerPid) {
    const workerDownRecordStart = readValidationRecords().length;
    const workerDownPrimer = runAgentCheck({
      to: "+15550006666",
      message: "How do I open a pull request again?",
    });
    report.agent.workerDownPrimer = workerDownPrimer;
    assertAgentCompleted("worker-down host-agent primer", workerDownPrimer);
    const workerDownPrimerStatus = runStatus();
    if (workerDownPrimerStatus.workerPid) {
      process.kill(workerDownPrimerStatus.workerPid, "SIGKILL");
      sleepMs(250);
    }
    const workerDownQuery = runAgentCheck({
      to: "+15550006666",
      message: "Please answer my previous question directly.",
    });
    report.agent.workerDownQuery = workerDownQuery;
    assertAgentCompleted("worker-down host-agent query", workerDownQuery);
    const workerDownStatus = runStatus();
    const workerDownValidation = readValidationRecordAfter(
      workerDownRecordStart,
      "How do I open a pull request again?",
    );
    const workerDownVisibleText = collectStrings(workerDownQuery).join("\n");
    const servedPullRequestGuidance =
      workerDownVisibleText.includes("gh pr create")
      || workerDownVisibleText.includes("pull request");
    report.assertions.workerDownHostFailOpen = {
      workerPidBeforeCrash: workerDownPrimerStatus.workerPid ?? null,
      currentPackVersionBeforeCrash: workerDownPrimerStatus.currentPackVersion ?? null,
      currentPackVersionAfterCrash: workerDownStatus.currentPackVersion ?? null,
      validationRecordCountBefore: workerDownRecordStart,
      validationRecordCountAfter: workerDownValidation.records.length,
      mode: workerDownValidation.record?.mode ?? null,
      traceId: workerDownValidation.record?.traceId ?? null,
      traceQueryText: workerDownValidation.record?.queryText ?? null,
      workerHealthyAfterCrash: workerDownStatus.workerHealthy ?? null,
      servedPullRequestGuidance,
      aborted: report.agent.workerDownQuery?.meta?.aborted ?? null,
    };

    if (!workerDownValidation.record) {
      throw new Error("Validation harness expected worker-down host-agent query to emit a host-surface validation record, but none was captured.");
    }
    if (workerDownValidation.record?.mode !== "use_brain") {
      throw new Error(`Validation harness expected worker-down host-agent query to keep routing through the last promoted pack, got ${workerDownValidation.record?.mode ?? "null"}.`);
    }
    if (!workerDownValidation.record?.traceId) {
      throw new Error("Validation harness expected worker-down host-agent query to still record a brain trace after killing the child worker.");
    }
    if (!servedPullRequestGuidance) {
      throw new Error("Validation harness expected worker-down host-agent query to keep serving last-promoted pull-request guidance after a child-worker crash.");
    }
    if (normalizeValidationText(workerDownValidation.record?.queryText) !== "how do i open a pull request again?") {
      throw new Error(`Validation harness expected worker-down host-agent validation query text to match the host prompt, got ${JSON.stringify(workerDownValidation.record?.queryText ?? null)}.`);
    }
    if ((workerDownPrimerStatus.currentPackVersion ?? null) !== (workerDownStatus.currentPackVersion ?? null)) {
      throw new Error(`Validation harness expected worker-down host-agent query to keep serving the last promoted pack version, got ${workerDownStatus.currentPackVersion ?? "null"} after ${workerDownPrimerStatus.currentPackVersion ?? "null"}.`);
    }
  } else {
    report.skipped.push({
      phase: "worker-down",
      reason: "Host-surface worker-down assertion requires child-worker mode with a live worker PID.",
    });
  }

  report.skipped.push({
    phase: "brain-teach",
    reason: "Phase-1 harness still needs a deterministic host-surface path for brain_teach assertion wiring; raw openclaw agent --local text prompting does not force tool use honestly.",
  });
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
