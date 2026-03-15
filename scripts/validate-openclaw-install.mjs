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

function updateConfig() {
  const config = JSON.parse(readFileSync(configPath, "utf8"));
  config.plugins ??= {};
  config.plugins.slots ??= {};
  config.plugins.entries ??= {};
  config.plugins.entries.openclawbrain ??= {};
  config.plugins.entries.openclawbrain.enabled = true;
  config.plugins.entries.openclawbrain.config = {
    enabled: true,
    dbPath: lcmDbPath,
    brainRoot,
    brainEnabled: true,
    brainEmbeddingProvider: cleanEnv().OPENCLAWBRAIN_EMBEDDING_PROVIDER,
    brainEmbeddingModel: cleanEnv().OPENCLAWBRAIN_EMBEDDING_MODEL,
    brainEmbeddingBaseUrl: cleanEnv().OPENCLAWBRAIN_EMBEDDING_BASE_URL,
  };
  config.plugins.slots.contextEngine = "openclawbrain";

  const validationModel = process.env.OPENCLAWBRAIN_VALIDATION_MODEL?.trim();
  if (validationModel) {
    config.agents ??= {};
    config.agents.defaults ??= {};
    config.agents.defaults.model = { primary: validationModel };
  }

  writeFileSync(configPath, `${JSON.stringify(config, null, 2)}\n`, "utf8");
}

function requireEmbeddingConfig() {
  if (!cleanEnv().OPENCLAWBRAIN_EMBEDDING_MODEL) {
    throw new Error(
      [
        "Embedding config is required for init.",
        "Set OPENCLAWBRAIN_VALIDATION_EMBEDDING_MODEL (and optionally provider/base URL) before running the harness.",
      ].join(" "),
    );
  }
}

function maybeRunAgentChecks(report) {
  const validationModel = process.env.OPENCLAWBRAIN_VALIDATION_MODEL?.trim();
  if (!validationModel) {
    report.skipped.push({
      phase: "agent-routing",
      reason: "OPENCLAWBRAIN_VALIDATION_MODEL is unset, so host-app local agent checks were not attempted.",
    });
    return;
  }

  const recurrentOutput = run("openclaw", [
    "agent",
    "--local",
    "--to",
    "+15550001111",
    "--message",
    "How do I open a pull request again?",
    "--json",
  ]);
  report.agent.recurrentQuery = extractJson(recurrentOutput);

  const recurrentStatus = extractJson(run("node", ["bin/openclawbrain.js", "status"]));
  const recurrentTrace = extractJson(run("node", ["bin/openclawbrain.js", "trace"]));
  report.assertions.recurrentQuery = {
    lastAssemblyMode: recurrentStatus.lastAssemblyDecision?.mode ?? null,
    traceId: recurrentTrace?.trace?.id ?? null,
    episodeId: recurrentTrace?.trace?.episodeId ?? null,
  };

  const shortLookupOutput = run("openclaw", [
    "agent",
    "--local",
    "--to",
    "+15550002222",
    "--message",
    "open PLAYBOOK.md",
    "--json",
  ]);
  report.agent.shortLookup = extractJson(shortLookupOutput);
  const shortStatus = extractJson(run("node", ["bin/openclawbrain.js", "status"]));
  report.assertions.shortLookup = {
    lastAssemblyMode: shortStatus.lastAssemblyDecision?.mode ?? null,
  };

  report.skipped.push(
    {
      phase: "brain-teach",
      reason: "Phase-1 harness scaffold still needs a deterministic host-surface path for brain_teach assertion wiring.",
    },
    {
      phase: "shadow-mode",
      reason: "Phase-1 harness scaffold still needs a dedicated shadow-mode config/run branch.",
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
    embeddingProvider: cleanEnv().OPENCLAWBRAIN_EMBEDDING_PROVIDER,
    embeddingModel: cleanEnv().OPENCLAWBRAIN_EMBEDDING_MODEL || null,
  };

  if (setupOnly) {
    report.skipped.push({
      phase: "init-and-agent-checks",
      reason: "--setup-only was requested.",
    });
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    process.exit(0);
  }

  requireEmbeddingConfig();
  report.init = extractJson(run("node", ["bin/openclawbrain.js", "init", fixtureWorkspace]));
  report.status = extractJson(run("node", ["bin/openclawbrain.js", "status"]));
  report.doctor = extractJson(run("node", ["bin/openclawbrain.js", "doctor"]));

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
