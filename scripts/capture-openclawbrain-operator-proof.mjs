#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import process from "node:process";

function usage() {
  process.stderr.write(
    [
      "Usage: node scripts/capture-openclawbrain-operator-proof.mjs --openclaw-home <path> [options]",
      "",
      "Options:",
      "  --cli-version <version>     Use npx @openclawbrain/cli@<version> (default: latest @openclawbrain/cli)",
      "  --activation-root <path>    Override activation root when status parsing is insufficient",
      "  --output-dir <path>         Bundle directory to write (default: ./artifacts/operator-proof-<timestamp>)",
      "  --skip-install              Do not run install step",
      "  --skip-restart              Do not run gateway restart step",
      "  --plugin-id <id>            Plugin id for inspect (default: openclawbrain)",
      "  --timeout-ms <ms>           Per-command timeout in ms (default: 120000)",
      "  --help                      Show this help",
      "",
      "This script captures one operator-proof bundle containing:",
      "- per-step stdout/stderr and exit results",
      "- OpenClaw gateway status",
      "- plugin inspect truth",
      "- detailed OpenClawBrain status",
      "- startup breadcrumb extraction",
      "- runtime-load-proof snapshot",
      "- summary.md, steps.json, and verdict.json",
    ].join("\n") + "\n",
  );
}

function parseArgs(argv) {
  const out = {
    openclawHome: null,
    cliVersion: null,
    activationRoot: null,
    outputDir: null,
    skipInstall: false,
    skipRestart: false,
    pluginId: "openclawbrain",
    timeoutMs: 120_000,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--openclaw-home":
        out.openclawHome = argv[++index] ?? null;
        break;
      case "--cli-version":
        out.cliVersion = argv[++index] ?? null;
        break;
      case "--activation-root":
        out.activationRoot = argv[++index] ?? null;
        break;
      case "--output-dir":
        out.outputDir = argv[++index] ?? null;
        break;
      case "--skip-install":
        out.skipInstall = true;
        break;
      case "--skip-restart":
        out.skipRestart = true;
        break;
      case "--plugin-id":
        out.pluginId = argv[++index] ?? out.pluginId;
        break;
      case "--timeout-ms":
        out.timeoutMs = Number.parseInt(argv[++index] ?? "120000", 10);
        break;
      case "--help":
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!out.openclawHome) {
    throw new Error("--openclaw-home is required");
  }

  if (!existsSync(out.openclawHome)) {
    throw new Error(`--openclaw-home directory does not exist: ${out.openclawHome}`);
  }

  return out;
}

function timestampToken(date = new Date()) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z").replace("T", "-");
}

function resolveOutputDir(options) {
  if (options.outputDir) {
    return path.resolve(options.outputDir);
  }
  return path.resolve(process.cwd(), "artifacts", `operator-proof-${timestampToken()}`);
}

function writeText(filePath, text) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, text, "utf8");
}

function writeJson(filePath, value) {
  writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function shellJoin(parts) {
  return parts
    .map((part) => {
      if (/^[A-Za-z0-9_./:@=-]+$/.test(part)) {
        return part;
      }
      return JSON.stringify(part);
    })
    .join(" ");
}

function runCapture(command, args, options = {}) {
  const startedAt = new Date();
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? process.cwd(),
    env: options.env ?? process.env,
    encoding: "utf8",
    timeout: options.timeoutMs,
  });
  const endedAt = new Date();
  return {
    label: options.label ?? command,
    command,
    argv: args,
    shellCommand: shellJoin([command, ...args]),
    startedAt: startedAt.toISOString(),
    endedAt: endedAt.toISOString(),
    durationMs: endedAt.getTime() - startedAt.getTime(),
    exitCode: typeof result.status === "number" ? result.status : null,
    signal: result.signal ?? null,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error ? String(result.error) : null,
  };
}

function summarizeCapture(step) {
  let resultClass = "unknown";
  if (step.exitCode === 0 && !step.error) {
    resultClass = "success";
  } else if (step.signal === "SIGTERM" || step.signal === "SIGKILL") {
    resultClass = "interrupted";
  } else if (step.error && /timed out/i.test(step.error)) {
    resultClass = "timed_out";
  } else if (step.exitCode !== 0 || step.error) {
    resultClass = "command_failed";
  }

  const captureState = step.exitCode === null && !step.stdout && !step.stderr
    ? "missing"
    : step.exitCode === null
      ? "partial"
      : "complete";

  return { resultClass, captureState };
}

function writeStepBundle(bundleDir, stepId, capture) {
  const stdoutName = `${stepId}.stdout.log`;
  const stderrName = `${stepId}.stderr.log`;
  writeText(path.join(bundleDir, stdoutName), capture.stdout);
  writeText(path.join(bundleDir, stderrName), capture.stderr);
  return { stdoutName, stderrName };
}

function extractGatewayLogPath(text) {
  const match = text.match(/^File logs:\s+(.+)$/m);
  return match ? match[1].trim() : null;
}

function extractActivationRoot(statusText, override) {
  if (override) {
    return path.resolve(override);
  }
  const targetMatch = statusText.match(/^target\s+activation=([^\s]+)\s+/m);
  if (targetMatch) {
    return targetMatch[1].trim();
  }
  const hostMatch = statusText.match(/^host\s+runtime=[^\s]+\s+activation=([^\s]+)$/m);
  if (hostMatch) {
    return hostMatch[1].trim();
  }
  return path.join(homedir(), ".openclawbrain", "activation");
}

function readJsonIfExists(filePath) {
  if (!filePath || !existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function readTextIfExists(filePath) {
  if (!filePath || !existsSync(filePath)) {
    return null;
  }
  return readFileSync(filePath, "utf8");
}

function extractStartupBreadcrumbs(logText, bundleStartedAtIso) {
  if (!logText) {
    return { all: [], afterBundleStart: [] };
  }
  const bundleStartMs = Date.parse(bundleStartedAtIso);
  const out = [];
  for (const line of logText.split(/\r?\n/)) {
    if (!line.includes("[openclawbrain] BRAIN LOADED") && !line.includes("[openclawbrain] BRAIN NOT YET LOADED")) {
      continue;
    }
    let parsed = null;
    try {
      parsed = JSON.parse(line);
    } catch {}
    const timestamp = parsed?._meta?.date ?? parsed?.time ?? null;
    out.push({
      line,
      timestamp,
      kind: line.includes("BRAIN LOADED") ? "loaded" : "not_yet_loaded",
    });
  }
  return {
    all: out,
    afterBundleStart: out.filter((entry) => entry.timestamp && Date.parse(entry.timestamp) >= bundleStartMs),
  };
}

function extractStatusSignals(statusText) {
  return {
    statusOk: /^STATUS ok$/m.test(statusText),
    loadProofReady: /loadProof=status_probe_ready/.test(statusText),
    runtimeProven: /attachTruth .*runtime=proven/.test(statusText),
    pluginInstalled: /hook\s+install=installed/.test(statusText),
    serveActivePack: /serve\s+state=serving_active_pack/.test(statusText),
    routeFnAvailable: /routeFn\s+available=yes/.test(statusText),
    proofPath: statusText.match(/proofPath=([^\s]+)/)?.[1] ?? null,
  };
}

function buildVerdict({ steps, gatewayStatus, pluginInspect, statusSignals, breadcrumbs, runtimeLoadProof, openclawHome }) {
  const failedStep = steps.find((step) => step.resultClass !== "success" && step.skipped !== true);
  if (failedStep) {
    return {
      verdict: "command_failed",
      severity: "blocking",
      why: `${failedStep.stepId} exited as ${failedStep.resultClass}`,
    };
  }

  const gatewayHealthy = /Runtime:\s+running/m.test(gatewayStatus) && /RPC probe:\s+ok/m.test(gatewayStatus);
  const pluginLoaded = /Status:\s+loaded/m.test(pluginInspect);
  const packagedHookPath = /Source:\s+.*openclawbrain\/dist\/extension\/index\.js/m.test(pluginInspect);
  const breadcrumbLoaded = breadcrumbs.afterBundleStart.some((entry) => entry.kind === "loaded");
  const runtimeProofMatched = Boolean(runtimeLoadProof?.profiles?.some((profile) => profile?.openclawHome === openclawHome));

  const missingProofs = [];
  if (!gatewayHealthy) missingProofs.push("gateway_health");
  if (!pluginLoaded) missingProofs.push("plugin_loaded");
  if (!packagedHookPath) missingProofs.push("packaged_hook_path");
  if (!statusSignals.statusOk) missingProofs.push("status_ok");
  if (!statusSignals.loadProofReady) missingProofs.push("load_proof");
  if (!statusSignals.runtimeProven) missingProofs.push("runtime_proven");
  if (!statusSignals.serveActivePack) missingProofs.push("serve_active_pack");
  if (!statusSignals.routeFnAvailable) missingProofs.push("route_fn");
  if (!breadcrumbLoaded) missingProofs.push("startup_breadcrumb");
  if (!runtimeProofMatched) missingProofs.push("runtime_load_proof_record");

  if (missingProofs.length === 0) {
    return {
      verdict: "success_and_proven",
      severity: "none",
      why: "install, restart, gateway health, plugin load, startup breadcrumb, runtime-load-proof record, and detailed status all aligned",
    };
  }

  const blocking = missingProofs.some((item) => [
    "gateway_health",
    "plugin_loaded",
    "packaged_hook_path",
    "status_ok",
    "load_proof",
    "runtime_proven",
    "serve_active_pack",
    "route_fn",
  ].includes(item));

  return {
    verdict: blocking ? "degraded_or_failed_proof" : "success_but_proof_incomplete",
    severity: blocking ? "blocking" : "degraded",
    why: `missing or conflicting proofs: ${missingProofs.join(", ")}`,
    missingProofs,
  };
}

function buildSummary({ options, steps, verdict, gatewayStatusText, pluginInspectText, statusSignals, breadcrumbs, runtimeLoadProofPath }) {
  const passed = [];
  const missing = [];

  if (steps.find((step) => step.stepId === "01-install")?.resultClass === "success") {
    passed.push("install command succeeded");
  }
  if (steps.find((step) => step.stepId === "02-restart")?.skipped === true || steps.find((step) => step.stepId === "02-restart")?.resultClass === "success") {
    passed.push("restart step completed or was intentionally skipped");
  }
  if (/Runtime:\s+running/m.test(gatewayStatusText) && /RPC probe:\s+ok/m.test(gatewayStatusText)) {
    passed.push("gateway status showed runtime running and RPC probe ok");
  }
  if (/Status:\s+loaded/m.test(pluginInspectText)) {
    passed.push("plugin inspect showed OpenClawBrain loaded");
  }
  if (statusSignals.statusOk) {
    passed.push("detailed status returned STATUS ok");
  }
  if (statusSignals.loadProofReady) {
    passed.push("detailed status reported loadProof=status_probe_ready");
  }
  if (statusSignals.serveActivePack) {
    passed.push("detailed status reported serve state=serving_active_pack");
  }
  if (statusSignals.routeFnAvailable) {
    passed.push("detailed status reported routeFn available=yes");
  }
  if (breadcrumbs.afterBundleStart.some((entry) => entry.kind === "loaded")) {
    passed.push("startup log contained a post-bundle [openclawbrain] BRAIN LOADED breadcrumb");
  }

  if (!statusSignals.loadProofReady) missing.push("detailed status did not prove hook load");
  if (!breadcrumbs.afterBundleStart.some((entry) => entry.kind === "loaded")) missing.push("no post-bundle startup breadcrumb was found");
  if (!runtimeLoadProofPath) missing.push("runtime-load-proof path could not be resolved");

  const lines = [
    "# OpenClawBrain operator proof summary",
    "",
    `- openclaw home: \`${options.openclawHome}\``,
    `- bundle verdict: **${verdict.verdict}**`,
    `- severity: **${verdict.severity}**`,
    `- why: ${verdict.why}`,
    "",
    "## Passed",
    ...passed.map((item) => `- ${item}`),
    "",
    "## Missing / incomplete",
    ...(missing.length === 0 ? ["- none"] : missing.map((item) => `- ${item}`)),
    "",
    "## Step ledger",
    ...steps.map((step) => `- ${step.stepId}: ${step.skipped ? "skipped" : `${step.resultClass} (${step.captureState})`} — ${step.summary}`),
  ];

  if (runtimeLoadProofPath) {
    lines.push("", "## Runtime proof file", `- ${runtimeLoadProofPath}`);
  }

  return `${lines.join("\n")}\n`;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const bundleStartedAt = new Date().toISOString();
  const bundleDir = resolveOutputDir(options);
  mkdirSync(bundleDir, { recursive: true });

  const cliPackage = options.cliVersion ? `@openclawbrain/cli@${options.cliVersion}` : "@openclawbrain/cli";
  const steps = [];

  function addStep(stepId, label, command, args, { skipped = false } = {}) {
    if (skipped) {
      steps.push({
        stepId,
        label,
        shellCommand: shellJoin([command, ...args]),
        skipped: true,
        captureState: "complete",
        resultClass: "success",
        summary: "step intentionally skipped",
        stdoutPath: null,
        stderrPath: null,
      });
      return { stdout: "", stderr: "", exitCode: 0, signal: null };
    }

    const capture = runCapture(command, args, {
      label,
      timeoutMs: options.timeoutMs,
    });
    const { stdoutName, stderrName } = writeStepBundle(bundleDir, stepId, capture);
    const summary = summarizeCapture(capture);
    steps.push({
      stepId,
      label,
      shellCommand: capture.shellCommand,
      startedAt: capture.startedAt,
      endedAt: capture.endedAt,
      durationMs: capture.durationMs,
      exitCode: capture.exitCode,
      signal: capture.signal,
      resultClass: summary.resultClass,
      captureState: summary.captureState,
      summary: summary.resultClass === "success"
        ? `${label} completed successfully`
        : `${label} ended as ${summary.resultClass}`,
      stdoutPath: stdoutName,
      stderrPath: stderrName,
    });
    return capture;
  }

  const installCapture = addStep(
    "01-install",
    "install",
    "npx",
    [cliPackage, "install", "--openclaw-home", options.openclawHome],
    { skipped: options.skipInstall },
  );

  const restartCapture = addStep(
    "02-restart",
    "gateway restart",
    "openclaw",
    ["gateway", "restart"],
    { skipped: options.skipRestart },
  );

  const gatewayStatusCapture = addStep(
    "03-gateway-status",
    "gateway status",
    "openclaw",
    ["gateway", "status"],
  );

  const pluginInspectCapture = addStep(
    "04-plugin-inspect",
    "plugin inspect",
    "openclaw",
    ["plugins", "inspect", options.pluginId],
  );

  const statusCapture = addStep(
    "05-detailed-status",
    "detailed status",
    "npx",
    [cliPackage, "status", "--openclaw-home", options.openclawHome, "--detailed"],
  );

  const gatewayLogPath = extractGatewayLogPath(gatewayStatusCapture.stdout);
  const activationRoot = extractActivationRoot(statusCapture.stdout, options.activationRoot);
  const runtimeLoadProofPath = path.join(activationRoot, "attachment-truth", "runtime-load-proofs.json");
  const runtimeLoadProof = readJsonIfExists(runtimeLoadProofPath);
  const gatewayLogText = readTextIfExists(gatewayLogPath);
  const breadcrumbs = extractStartupBreadcrumbs(gatewayLogText, bundleStartedAt);
  const statusSignals = extractStatusSignals(statusCapture.stdout);

  writeText(
    path.join(bundleDir, "extracted-startup-breadcrumbs.log"),
    breadcrumbs.all.length === 0
      ? "<no matching breadcrumbs found>\n"
      : breadcrumbs.all.map((entry) => entry.line).join("\n") + "\n",
  );
  writeJson(path.join(bundleDir, "runtime-load-proof.json"), {
    path: runtimeLoadProofPath,
    exists: runtimeLoadProof !== null,
    value: runtimeLoadProof,
  });

  const verdict = buildVerdict({
    steps,
    gatewayStatus: gatewayStatusCapture.stdout,
    pluginInspect: pluginInspectCapture.stdout,
    statusSignals,
    breadcrumbs,
    runtimeLoadProof,
    openclawHome: path.resolve(options.openclawHome),
  });

  writeJson(path.join(bundleDir, "steps.json"), {
    bundleStartedAt,
    openclawHome: path.resolve(options.openclawHome),
    cliPackage,
    gatewayLogPath,
    activationRoot,
    steps,
  });
  writeJson(path.join(bundleDir, "verdict.json"), {
    bundleStartedAt,
    verdict,
    statusSignals,
    breadcrumbs: {
      allCount: breadcrumbs.all.length,
      postBundleCount: breadcrumbs.afterBundleStart.length,
      postBundleKinds: breadcrumbs.afterBundleStart.map((entry) => entry.kind),
    },
    runtimeLoadProofPath,
  });
  writeText(
    path.join(bundleDir, "summary.md"),
    buildSummary({
      options,
      steps,
      verdict,
      gatewayStatusText: gatewayStatusCapture.stdout,
      pluginInspectText: pluginInspectCapture.stdout,
      statusSignals,
      breadcrumbs,
      runtimeLoadProofPath,
    }),
  );

  process.stdout.write(`${JSON.stringify({ ok: true, bundleDir, verdict }, null, 2)}\n`);
}

main();
