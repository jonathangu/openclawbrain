import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import process from "node:process";

export const DEFAULT_OPERATOR_PROOF_PLUGIN_ID = "openclawbrain";
export const DEFAULT_OPERATOR_PROOF_TIMEOUT_MS = 120_000;

function quoteShellArg(value) {
    return `'${value.replace(/'/g, `"'"'`)}'`;
}

function normalizeOptionalCliString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}

function normalizeReportedProofPath(filePath) {
    const normalizedPath = normalizeOptionalCliString(filePath);
    if (normalizedPath === null) {
        return null;
    }
    if (normalizedPath === "~") {
        return homedir();
    }
    if (normalizedPath.startsWith("~/")) {
        return path.join(homedir(), normalizedPath.slice(2));
    }
    return path.isAbsolute(normalizedPath)
        ? normalizedPath
        : path.resolve(normalizedPath);
}

function canonicalizeExistingProofPath(filePath) {
    const resolvedPath = path.resolve(filePath);
    try {
        return realpathSync(resolvedPath);
    }
    catch {
        return resolvedPath;
    }
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

function timestampToken(date = new Date()) {
    return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z").replace("T", "-");
}

function resolveProofOutputDir(options) {
    if (options.outputDir !== null) {
        return path.resolve(options.outputDir);
    }
    return path.resolve(options.cwd ?? process.cwd(), "artifacts", `operator-proof-${timestampToken()}`);
}

function writeText(filePath, text) {
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, text, "utf8");
}

function writeJson(filePath, value) {
    writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function buildCurrentCliInvocation(cliEntryPath = process.argv[1]) {
    const normalizedEntryPath = normalizeOptionalCliString(cliEntryPath);
    if (normalizedEntryPath === null) {
        return {
            command: "openclawbrain",
            args: []
        };
    }
    return {
        command: process.execPath,
        args: [path.resolve(normalizedEntryPath)]
    };
}

function defaultRunCapture(command, args, options = {}) {
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
    }
    else if (step.signal === "SIGTERM" || step.signal === "SIGKILL") {
        resultClass = "interrupted";
    }
    else if (step.error && /timed out/i.test(step.error)) {
        resultClass = "timed_out";
    }
    else if (step.exitCode !== 0 || step.error) {
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
    if (override !== null) {
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

function readTextIfExists(filePath) {
    if (filePath === null || !existsSync(filePath)) {
        return null;
    }
    return readFileSync(filePath, "utf8");
}

function readJsonSnapshot(filePath) {
    if (filePath === null) {
        return {
            path: null,
            exists: false,
            error: "proof path unresolved",
            value: null
        };
    }
    if (!existsSync(filePath)) {
        return {
            path: filePath,
            exists: false,
            error: null,
            value: null
        };
    }
    try {
        return {
            path: filePath,
            exists: true,
            error: null,
            value: JSON.parse(readFileSync(filePath, "utf8"))
        };
    }
    catch (error) {
        return {
            path: filePath,
            exists: true,
            error: error instanceof Error ? error.message : String(error),
            value: null
        };
    }
}

function describeStepWarning(step) {
    return step.captureState === "partial"
        ? `${step.stepId} ended as ${step.resultClass} with partial capture`
        : `${step.stepId} ended as ${step.resultClass}`;
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
        }
        catch {
            parsed = null;
        }
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
        proofError: statusText.match(/proofError=([^\s]+)/)?.[1] ?? null,
    };
}

function hasPackagedHookSource(pluginInspectText) {
    return /Source:\s+.*(?:@openclawbrain[\\/]+openclaw|openclawbrain)[\\/]+dist[\\/]+extension[\\/]+index\.js/m.test(pluginInspectText);
}

function buildVerdict({ steps, gatewayStatus, pluginInspect, statusSignals, breadcrumbs, runtimeLoadProofSnapshot, openclawHome }) {
    const failedSteps = steps.filter((step) => step.resultClass !== "success" && step.skipped !== true);
    const failedDetailedStatusStep = failedSteps.find((step) => step.stepId === "05-detailed-status");
    const gatewayHealthy = /Runtime:\s+running/m.test(gatewayStatus) && /RPC probe:\s+ok/m.test(gatewayStatus);
    const pluginLoaded = /Status:\s+loaded/m.test(pluginInspect);
    const packagedHookPath = hasPackagedHookSource(pluginInspect);
    const breadcrumbLoaded = breadcrumbs.afterBundleStart.some((entry) => entry.kind === "loaded");
    const runtimeProofMatched = Array.isArray(runtimeLoadProofSnapshot?.value?.profiles)
        && runtimeLoadProofSnapshot.value.profiles.some((profile) => canonicalizeExistingProofPath(profile?.openclawHome ?? "") === canonicalizeExistingProofPath(openclawHome));
    const runtimeTruthGaps = [];
    if (!statusSignals.statusOk)
        runtimeTruthGaps.push("status_ok");
    if (!statusSignals.loadProofReady)
        runtimeTruthGaps.push("load_proof");
    if (!statusSignals.runtimeProven)
        runtimeTruthGaps.push("runtime_proven");
    if (!statusSignals.serveActivePack)
        runtimeTruthGaps.push("serve_active_pack");
    if (!statusSignals.routeFnAvailable)
        runtimeTruthGaps.push("route_fn");
    const warningCodes = [];
    const warnings = [];
    if (!gatewayHealthy) {
        warningCodes.push("gateway_health");
        warnings.push("gateway status did not confirm runtime running and RPC probe ok");
    }
    if (!pluginLoaded) {
        warningCodes.push("plugin_loaded");
        warnings.push("plugin inspect did not report Status: loaded");
    }
    if (!packagedHookPath) {
        warningCodes.push("packaged_hook_path");
        warnings.push("plugin inspect did not confirm the packaged hook source");
    }
    if (!breadcrumbLoaded) {
        warningCodes.push("startup_breadcrumb");
        warnings.push("startup log did not contain a post-bundle [openclawbrain] BRAIN LOADED breadcrumb");
    }
    if (!runtimeProofMatched) {
        warningCodes.push("runtime_load_proof_record");
        warnings.push(runtimeLoadProofSnapshot.error !== null
            ? `runtime-load-proof snapshot was unreadable: ${runtimeLoadProofSnapshot.error}`
            : runtimeLoadProofSnapshot.exists
                ? "runtime-load-proof snapshot did not include the current openclaw home"
                : "runtime-load-proof snapshot was missing");
    }
    if (statusSignals.proofError !== null && statusSignals.proofError !== "none") {
        warningCodes.push(`proof_error:${statusSignals.proofError}`);
        warnings.push(`detailed status reported proofError=${statusSignals.proofError}`);
    }
    for (const step of failedSteps) {
        warningCodes.push(`step:${step.stepId}:${step.resultClass}:${step.captureState}`);
        warnings.push(describeStepWarning(step));
    }
    const uniqueWarningCodes = [...new Set(warningCodes)];
    const uniqueWarnings = [...new Set(warnings)];
    if (runtimeTruthGaps.length === 0 && uniqueWarningCodes.length === 0) {
        return {
            verdict: "success_and_proven",
            severity: "none",
            why: "install, restart, gateway health, plugin load, startup breadcrumb, runtime-load-proof record, and detailed status all aligned",
            missingProofs: [],
            warnings: [],
        };
    }
    if (runtimeTruthGaps.length === 0) {
        return {
            verdict: "success_but_proof_incomplete",
            severity: "degraded",
            why: `status/runtime evidence stayed healthy, but proof warnings remained: ${uniqueWarningCodes.join(", ")}`,
            missingProofs: uniqueWarningCodes,
            warnings: uniqueWarnings,
        };
    }
    const hasUsableStatusTruth = statusSignals.statusOk
        || statusSignals.loadProofReady
        || statusSignals.runtimeProven
        || statusSignals.serveActivePack
        || statusSignals.routeFnAvailable;
    if (failedDetailedStatusStep && !hasUsableStatusTruth) {
        return {
            verdict: "command_failed",
            severity: "blocking",
            why: `${failedDetailedStatusStep.stepId} ended as ${failedDetailedStatusStep.resultClass} before runtime truth could be established`,
            missingProofs: runtimeTruthGaps,
            warnings: uniqueWarnings,
        };
    }
    return {
        verdict: "degraded_or_failed_proof",
        severity: "blocking",
        why: `missing or conflicting runtime truths: ${runtimeTruthGaps.join(", ")}`,
        missingProofs: [...new Set([...runtimeTruthGaps, ...uniqueWarningCodes])],
        warnings: uniqueWarnings,
    };
}

function buildSummary({ options, steps, verdict, gatewayStatusText, pluginInspectText, statusSignals, breadcrumbs, runtimeLoadProofSnapshot }) {
    const passed = [];
    const missing = [];
    const warnings = Array.isArray(verdict.warnings) ? verdict.warnings : [];
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
    if (!statusSignals.loadProofReady)
        missing.push("detailed status did not prove hook load");
    if (!breadcrumbs.afterBundleStart.some((entry) => entry.kind === "loaded"))
        missing.push("no post-bundle startup breadcrumb was found");
    if (runtimeLoadProofSnapshot.path === null)
        missing.push("runtime-load-proof path could not be resolved");
    if (runtimeLoadProofSnapshot.error !== null)
        missing.push(`runtime-load-proof snapshot was unreadable: ${runtimeLoadProofSnapshot.error}`);
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
        "## Warnings",
        ...(warnings.length === 0 ? ["- none"] : warnings.map((item) => `- ${item}`)),
        "",
        "## Step ledger",
        ...steps.map((step) => `- ${step.stepId}: ${step.skipped ? "skipped" : `${step.resultClass} (${step.captureState})`} - ${step.summary}`),
    ];
    if (runtimeLoadProofSnapshot.path !== null) {
        lines.push("", "## Runtime proof file", `- ${runtimeLoadProofSnapshot.path}`);
    }
    return `${lines.join("\n")}\n`;
}

function readOpenClawProfileName(openclawHome) {
    try {
        const openclawJsonPath = path.join(openclawHome, "openclaw.json");
        if (!existsSync(openclawJsonPath)) {
            return null;
        }
        const parsed = JSON.parse(readFileSync(openclawJsonPath, "utf8"));
        return normalizeOptionalCliString(parsed?.profile);
    }
    catch {
        return null;
    }
}

function buildGatewayArgs(action, profileName) {
    return profileName === null
        ? ["gateway", action]
        : ["gateway", action, "--profile", profileName];
}

export function buildProofCommandForOpenClawHome(openclawHome) {
    return `openclawbrain proof --openclaw-home ${quoteShellArg(path.resolve(openclawHome))}`;
}

export function buildProofCommandHelpSection() {
    return {
        usage: "  openclawbrain proof --openclaw-home <path> [options]",
        optionLines: [
            "  --output-dir <path>         Bundle directory for proof artifacts (proof only). Defaults to ./artifacts/operator-proof-<timestamp>.",
            "  --skip-install              Capture proof without rerunning install first (proof only).",
            "  --skip-restart              Capture proof without restarting OpenClaw first (proof only).",
            `  --plugin-id <id>            Plugin id for \`openclaw plugins inspect\` (proof only; default: ${DEFAULT_OPERATOR_PROOF_PLUGIN_ID}).`,
            `  --timeout-ms <ms>           Per-step timeout in ms for proof capture (proof only; default: ${DEFAULT_OPERATOR_PROOF_TIMEOUT_MS}).`,
        ],
        lifecycle: "  5. proof              openclawbrain proof --openclaw-home <path> - capture one durable operator proof bundle after install/restart/status",
        advanced: "  proof        capture one durable operator proof bundle with step logs, startup breadcrumbs, and a runtime-load-proof snapshot",
    };
}

export function parseProofCliArgs(argv, options = {}) {
    const existsSyncImpl = options.existsSyncImpl ?? existsSync;
    let openclawHome = null;
    let activationRoot = null;
    let outputDir = null;
    let skipInstall = false;
    let skipRestart = false;
    let pluginId = DEFAULT_OPERATOR_PROOF_PLUGIN_ID;
    let timeoutMs = DEFAULT_OPERATOR_PROOF_TIMEOUT_MS;
    let json = false;
    let help = false;
    for (let index = 0; index < argv.length; index += 1) {
        const arg = argv[index];
        if (arg === "--help" || arg === "-h") {
            help = true;
            continue;
        }
        if (arg === "--json") {
            json = true;
            continue;
        }
        if (arg === "--skip-install") {
            skipInstall = true;
            continue;
        }
        if (arg === "--skip-restart") {
            skipRestart = true;
            continue;
        }
        if (arg === "--openclaw-home") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--openclaw-home requires a value");
            }
            openclawHome = next;
            index += 1;
            continue;
        }
        if (arg === "--activation-root") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--activation-root requires a value");
            }
            activationRoot = next;
            index += 1;
            continue;
        }
        if (arg === "--output-dir") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--output-dir requires a value");
            }
            outputDir = next;
            index += 1;
            continue;
        }
        if (arg === "--plugin-id") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--plugin-id requires a value");
            }
            pluginId = next;
            index += 1;
            continue;
        }
        if (arg === "--timeout-ms") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--timeout-ms requires a value");
            }
            const parsed = Number.parseInt(next, 10);
            if (!Number.isInteger(parsed) || parsed < 1) {
                throw new Error("--timeout-ms must be a positive integer");
            }
            timeoutMs = parsed;
            index += 1;
            continue;
        }
        throw new Error(`unknown argument for proof: ${arg}`);
    }
    if (help) {
        return {
            command: "proof",
            openclawHome: "",
            activationRoot: null,
            outputDir: null,
            skipInstall,
            skipRestart,
            pluginId,
            timeoutMs,
            json,
            help
        };
    }
    if (openclawHome === null) {
        throw new Error("proof requires --openclaw-home <path>");
    }
    const resolvedOpenClawHome = path.resolve(openclawHome);
    if (!existsSyncImpl(resolvedOpenClawHome)) {
        throw new Error(`--openclaw-home directory does not exist: ${resolvedOpenClawHome}`);
    }
    return {
        command: "proof",
        openclawHome: resolvedOpenClawHome,
        activationRoot: activationRoot === null ? null : path.resolve(activationRoot),
        outputDir: outputDir === null ? null : path.resolve(outputDir),
        skipInstall,
        skipRestart,
        pluginId,
        timeoutMs,
        json,
        help
    };
}

export function captureOperatorProofBundle(options) {
    const cliInvocation = options.cliInvocation ?? buildCurrentCliInvocation();
    const runCapture = options.runCapture ?? defaultRunCapture;
    const bundleStartedAt = new Date().toISOString();
    const bundleDir = resolveProofOutputDir(options);
    mkdirSync(bundleDir, { recursive: true });
    const steps = [];
    const gatewayProfile = readOpenClawProfileName(options.openclawHome);
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
            return { stdout: "", stderr: "", exitCode: 0, signal: null, error: null };
        }
        const capture = runCapture(command, args, {
            label,
            cwd: options.cwd ?? process.cwd(),
            env: options.env ?? process.env,
            timeoutMs: options.timeoutMs,
        });
        const { stdoutName, stderrName } = writeStepBundle(bundleDir, stepId, capture);
        const summary = summarizeCapture(capture);
        steps.push({
            stepId,
            label,
            shellCommand: capture.shellCommand ?? shellJoin([command, ...args]),
            startedAt: capture.startedAt ?? null,
            endedAt: capture.endedAt ?? null,
            durationMs: capture.durationMs ?? null,
            exitCode: capture.exitCode ?? null,
            signal: capture.signal ?? null,
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
    addStep("01-install", "install", cliInvocation.command, [...cliInvocation.args, "install", "--openclaw-home", options.openclawHome], { skipped: options.skipInstall === true });
    addStep("02-restart", "gateway restart", "openclaw", buildGatewayArgs("restart", gatewayProfile), { skipped: options.skipRestart === true });
    const gatewayStatusCapture = addStep("03-gateway-status", "gateway status", "openclaw", buildGatewayArgs("status", gatewayProfile));
    const pluginInspectCapture = addStep("04-plugin-inspect", "plugin inspect", "openclaw", ["plugins", "inspect", options.pluginId]);
    const statusCapture = addStep("05-detailed-status", "detailed status", cliInvocation.command, [...cliInvocation.args, "status", "--openclaw-home", options.openclawHome, "--detailed"]);
    const gatewayLogPath = extractGatewayLogPath(gatewayStatusCapture.stdout);
    const activationRoot = extractActivationRoot(statusCapture.stdout, options.activationRoot ?? null);
    const statusSignals = extractStatusSignals(statusCapture.stdout);
    const runtimeLoadProofPath = normalizeReportedProofPath(statusSignals.proofPath)
        ?? path.join(activationRoot, "attachment-truth", "runtime-load-proofs.json");
    const runtimeLoadProofSnapshot = readJsonSnapshot(runtimeLoadProofPath);
    const gatewayLogText = readTextIfExists(gatewayLogPath);
    const breadcrumbs = extractStartupBreadcrumbs(gatewayLogText, bundleStartedAt);
    writeText(path.join(bundleDir, "extracted-startup-breadcrumbs.log"), breadcrumbs.all.length === 0
        ? "<no matching breadcrumbs found>\n"
        : `${breadcrumbs.all.map((entry) => entry.line).join("\n")}\n`);
    writeJson(path.join(bundleDir, "runtime-load-proof.json"), runtimeLoadProofSnapshot);
    const verdict = buildVerdict({
        steps,
        gatewayStatus: gatewayStatusCapture.stdout,
        pluginInspect: pluginInspectCapture.stdout,
        statusSignals,
        breadcrumbs,
        runtimeLoadProofSnapshot,
        openclawHome: options.openclawHome,
    });
    writeJson(path.join(bundleDir, "steps.json"), {
        bundleStartedAt,
        openclawHome: canonicalizeExistingProofPath(options.openclawHome),
        activationRoot,
        gatewayProfile,
        gatewayLogPath,
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
        runtimeLoadProofError: runtimeLoadProofSnapshot.error,
    });
    writeText(path.join(bundleDir, "summary.md"), buildSummary({
        options,
        steps,
        verdict,
        gatewayStatusText: gatewayStatusCapture.stdout,
        pluginInspectText: pluginInspectCapture.stdout,
        statusSignals,
        breadcrumbs,
        runtimeLoadProofSnapshot,
    }));
    return {
        ok: true,
        bundleDir,
        bundleStartedAt,
        activationRoot,
        gatewayProfile,
        gatewayLogPath,
        runtimeLoadProofPath,
        runtimeLoadProofSnapshot,
        verdict,
        statusSignals,
        steps,
        summaryPath: path.join(bundleDir, "summary.md"),
        stepsPath: path.join(bundleDir, "steps.json"),
        verdictPath: path.join(bundleDir, "verdict.json"),
        breadcrumbPath: path.join(bundleDir, "extracted-startup-breadcrumbs.log"),
        runtimeLoadProofSnapshotPath: path.join(bundleDir, "runtime-load-proof.json"),
    };
}

export function formatOperatorProofResult(result) {
    const lines = [
        `PROOF ${result.verdict.verdict}`,
        `  Severity: ${result.verdict.severity}`,
        `  Why: ${result.verdict.why}`,
        `  Bundle: ${result.bundleDir}`,
        `  Summary: ${result.summaryPath}`,
        `  Steps: ${result.stepsPath}`,
        `  Verdict: ${result.verdictPath}`,
        `  Breadcrumbs: ${result.breadcrumbPath}`,
        `  Runtime proof: ${result.runtimeLoadProofSnapshotPath}`,
    ];
    return lines.join("\n");
}
