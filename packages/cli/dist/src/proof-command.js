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

function readJsonObject(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    return value;
}

function parseJsonObjectText(text) {
    if (typeof text !== "string" || text.trim().length === 0) {
        return null;
    }
    try {
        return readJsonObject(JSON.parse(text));
    }
    catch {
        return null;
    }
}

function normalizeOptionalBoolean(value) {
    return value === true ? true : value === false ? false : null;
}

function extractInstallRestartState(text) {
    const parsed = parseJsonObjectText(text);
    const restart = readJsonObject(parsed?.restart ?? null);
    if (restart === null) {
        return null;
    }
    return {
        required: normalizeOptionalBoolean(restart.required),
        performed: normalizeOptionalBoolean(restart.performed),
    };
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
    const surface = extractKeyValuePairs(extractDetailedStatusLine(statusText, "surface"));
    return {
        statusOk: /^STATUS ok$/m.test(statusText),
        loadProofReady: /loadProof=status_probe_ready/.test(statusText),
        runtimeProven: /attachTruth .*runtime=proven/.test(statusText),
        pluginInstalled: /hook\s+install=installed/.test(statusText),
        serveActivePack: /serve\s+state=serving_active_pack/.test(statusText),
        routeFnAvailable: /routeFn\s+available=yes/.test(statusText),
        proofPath: statusText.match(/proofPath=([^\s]+)/)?.[1] ?? null,
        proofError: statusText.match(/proofError=([^\s]+)/)?.[1] ?? null,
        surfaceBoundary: surface.boundary ?? null,
        surfaceSkew: surface.skew ?? null,
        surfaceConvergeState: surface.converge ?? null,
        surfaceDaemonSource: surface.daemonSource ?? null,
        surfaceSelectedHome: surface.selectedHome ?? null,
    };
}
function extractDetailedStatusLine(statusText, prefix) {
    const normalizedPrefix = `${prefix} `;
    return statusText.split(/\r?\n/).find((line) => line.startsWith(normalizedPrefix)) ?? null;
}
function extractKeyValuePairs(line) {
    if (typeof line !== "string") {
        return {};
    }
    const pairs = {};
    for (const match of line.matchAll(/([A-Za-z][A-Za-z0-9]*)=([^\s]+)/g)) {
        pairs[match[1]] = match[2];
    }
    return pairs;
}
function extractAttachedProfileCoverageEntries(line) {
    if (typeof line !== "string") {
        return [];
    }
    const normalized = line.replace(/^attachedSet\s+/, "");
    const proofPathIndex = normalized.indexOf(" proofPath=");
    const entriesText = (proofPathIndex === -1 ? normalized : normalized.slice(0, proofPathIndex)).trim();
    if (entriesText.length === 0 || entriesText === "none") {
        return [];
    }
    const entries = [];
    let index = 0;
    while (index < entriesText.length) {
        while (index < entriesText.length && entriesText[index] === " ") {
            index += 1;
        }
        if (index >= entriesText.length) {
            break;
        }
        const bracketStart = entriesText.indexOf("[", index);
        if (bracketStart === -1) {
            break;
        }
        const bracketEnd = entriesText.indexOf("]", bracketStart + 1);
        if (bracketEnd === -1) {
            break;
        }
        const rawLabel = entriesText.slice(index, bracketStart).trim();
        const fields = extractKeyValuePairs(entriesText.slice(bracketStart + 1, bracketEnd));
        entries.push({
            label: rawLabel.replace(/^\*/, "").trim(),
            current: rawLabel.startsWith("*"),
            hookFiles: fields.hook ?? "unknown",
            configLoad: fields.config ?? "unknown",
            runtimeLoad: fields.runtime ?? "unknown",
            loadedAt: fields.loadedAt ?? null,
            coverageState: fields.hook === "present" && fields.config === "allows_load" && fields.runtime === "proven"
                ? "covered"
                : "attention"
        });
        index = bracketEnd + 1;
    }
    return entries;
}
function buildCoverageSnapshot({ attachedSetLine, runtimeLoadProofSnapshot, openclawHome }) {
    const parsedEntries = extractAttachedProfileCoverageEntries(attachedSetLine);
    const proofProfiles = Array.isArray(runtimeLoadProofSnapshot?.value?.profiles)
        ? runtimeLoadProofSnapshot.value.profiles
        : [];
    const profiles = parsedEntries.length > 0
        ? parsedEntries
        : proofProfiles.map((profile) => ({
            label: `${profile?.profileId ?? "current_profile"}@${canonicalizeExistingProofPath(profile?.openclawHome ?? "")}`,
            current: canonicalizeExistingProofPath(profile?.openclawHome ?? "") === canonicalizeExistingProofPath(openclawHome),
            hookFiles: "unknown",
            configLoad: "unknown",
            runtimeLoad: "proven",
            loadedAt: profile?.loadedAt ?? null,
            coverageState: "covered"
        }));
    const runtimeProvenCount = profiles.filter((entry) => entry.runtimeLoad === "proven").length;
    return {
        contract: "openclaw_operator_profile_coverage_snapshot.v1",
        generatedAt: new Date().toISOString(),
        openclawHome: canonicalizeExistingProofPath(openclawHome),
        attachedProfileCount: profiles.length,
        runtimeProofProfileCount: proofProfiles.length,
        hookReadyCount: profiles.filter((entry) => entry.hookFiles === "present").length,
        configReadyCount: profiles.filter((entry) => entry.configLoad === "allows_load").length,
        runtimeProvenCount,
        coverageRate: profiles.length === 0 ? null : runtimeProvenCount / profiles.length,
        profiles
    };
}
function buildHardeningSnapshot({ attachTruthLine, serveLine, routeFnLine, surfaceLine, verdict, statusSignals }) {
    const attachTruth = extractKeyValuePairs(attachTruthLine);
    const serve = extractKeyValuePairs(serveLine);
    const routeFn = extractKeyValuePairs(routeFnLine);
    const surface = extractKeyValuePairs(surfaceLine);
    return {
        contract: "openclaw_operator_hardening_snapshot.v1",
        generatedAt: new Date().toISOString(),
        statusSignals: {
            statusOk: statusSignals.statusOk,
            loadProofReady: statusSignals.loadProofReady,
            runtimeProven: statusSignals.runtimeProven,
            serveActivePack: statusSignals.serveActivePack,
            routeFnAvailable: statusSignals.routeFnAvailable,
            surfaceConverged: statusSignals.surfaceConvergeState === "converged",
        },
        attachTruth: {
            current: attachTruth.current ?? null,
            hook: attachTruth.hook ?? null,
            config: attachTruth.config ?? null,
            runtime: attachTruth.runtime ?? null,
            watcher: attachTruth.watcher ?? null,
        },
        surface: {
            boundary: surface.boundary ?? null,
            skew: surface.skew ?? null,
            converge: surface.converge ?? null,
            daemonSource: surface.daemonSource ?? null,
            selectedHome: surface.selectedHome ?? null,
            daemon: surface.daemon ?? null,
            hook: surface.hook ?? null,
        },
        serve: {
            state: serve.state ?? null,
            failOpen: serve.failOpen ?? null,
            hardFail: serve.hardFail ?? null,
            usedRouteFn: serve.usedRouteFn ?? null,
            awaitingFirstExport: serve.awaitingFirstExport ?? null,
        },
        routeFn: {
            available: routeFn.available ?? null,
            freshness: routeFn.freshness ?? null,
        },
        verdict: {
            verdict: verdict.verdict,
            severity: verdict.severity,
            missingProofCount: Array.isArray(verdict.missingProofs) ? verdict.missingProofs.length : 0,
            warningCount: Array.isArray(verdict.warnings) ? verdict.warnings.length : 0,
        }
    };
}

function hasPackagedHookSource(pluginInspectText, openclawHome) {
    if (/Source:\s+.*(?:@openclawbrain[\\/]+openclaw|openclawbrain)[\\/]+dist[\\/]+extension[\\/]+index\.js/m.test(pluginInspectText)) {
        return true;
    }
    const generatedShadowHookPath = canonicalizeExistingProofPath(path.join(openclawHome, "extensions", "openclawbrain", "index.ts"));
    const sourceMatch = pluginInspectText.match(/^Source:\s+(.+)$/m);
    const reportedSourcePath = canonicalizeExistingProofPath(sourceMatch?.[1] ?? "");
    return reportedSourcePath === generatedShadowHookPath;
}

function buildVerdict({ steps, gatewayStatus, pluginInspect, statusSignals, breadcrumbs, runtimeLoadProofSnapshot, coverageSnapshot, openclawHome }) {
    const failedSteps = steps.filter((step) => step.resultClass !== "success" && step.skipped !== true);
    const failedDetailedStatusStep = failedSteps.find((step) => step.stepId === "05-detailed-status");
    const gatewayHealthy = /Runtime:\s+running/m.test(gatewayStatus) && /RPC probe:\s+ok/m.test(gatewayStatus);
    const pluginLoaded = /Status:\s+loaded/m.test(pluginInspect);
    const packagedHookPath = hasPackagedHookSource(pluginInspect, openclawHome);
    const breadcrumbLoaded = breadcrumbs.afterBundleStart.some((entry) => entry.kind === "loaded");
    const runtimeProofMatched = Array.isArray(runtimeLoadProofSnapshot?.value?.profiles)
        && runtimeLoadProofSnapshot.value.profiles.some((profile) => canonicalizeExistingProofPath(profile?.openclawHome ?? "") === canonicalizeExistingProofPath(openclawHome));
    const runtimeTruthGaps = [];
    const currentCoverageEntry = Array.isArray(coverageSnapshot?.profiles)
        ? coverageSnapshot.profiles.find((entry) => entry.current)
        : null;
    const currentProfileRuntimeCovered = currentCoverageEntry?.runtimeLoad === "proven";
    const crossProfileCoverageGapOnly = currentProfileRuntimeCovered
        && (coverageSnapshot?.attachedProfileCount ?? 0) > (coverageSnapshot?.runtimeProvenCount ?? 0)
        && (statusSignals.proofError === null || statusSignals.proofError === "none");
    const strongRuntimeTruth = statusSignals.loadProofReady
        && statusSignals.runtimeProven
        && statusSignals.serveActivePack
        && statusSignals.routeFnAvailable;
    const surfaceConverged = statusSignals.surfaceConvergeState === "converged";
    const surfaceHalfConverged = statusSignals.surfaceConvergeState === "half_converged";
    if (!statusSignals.loadProofReady)
        runtimeTruthGaps.push("load_proof");
    if (!statusSignals.runtimeProven)
        runtimeTruthGaps.push("runtime_proven");
    if (!statusSignals.serveActivePack)
        runtimeTruthGaps.push("serve_active_pack");
    if (!statusSignals.routeFnAvailable)
        runtimeTruthGaps.push("route_fn");
    if (surfaceHalfConverged) {
        runtimeTruthGaps.push("surface_converged");
    }
    const warningCodes = [];
    const warnings = [];
    if (!statusSignals.statusOk) {
        if (strongRuntimeTruth) {
            if (!crossProfileCoverageGapOnly) {
                warningCodes.push("status_warn");
                warnings.push("detailed status did not return STATUS ok, but loadProof/runtime/serve/routeFn proofs stayed healthy");
            }
        }
        else {
            runtimeTruthGaps.push("status_ok");
        }
    }
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
    if (surfaceHalfConverged) {
        warningCodes.push(`surface_half_converged:${statusSignals.surfaceSkew ?? "unknown"}`);
        warnings.push(`detailed status reported daemon-vs-installed-surface half-converged (${statusSignals.surfaceSkew ?? "unknown"})`);
    }
    else if (!surfaceConverged) {
        warningCodes.push(`surface_unverified:${statusSignals.surfaceConvergeState ?? "unknown"}`);
        warnings.push("detailed status did not fully prove daemon-vs-installed-surface convergence");
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

function buildSummary({ options, steps, verdict, gatewayStatusText, pluginInspectText, statusSignals, breadcrumbs, runtimeLoadProofSnapshot, surfaceLine, surfacesLine, surfaceNoteLine, hotfixLine, guardLine, feedbackLine, attributionLine, attributionCoverageLine, learningPathLine, learningFlowLine, learningHealthLine, coverageSnapshot, hardeningSnapshot }) {
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
    if (statusSignals.surfaceConvergeState === "converged") {
        passed.push("detailed status reported daemon-vs-installed-surface convergence");
    }
    if (breadcrumbs.afterBundleStart.some((entry) => entry.kind === "loaded")) {
        passed.push("startup log contained a post-bundle [openclawbrain] BRAIN LOADED breadcrumb");
    }
    if (!statusSignals.loadProofReady)
        missing.push("detailed status did not prove hook load");
    if (statusSignals.surfaceConvergeState === "half_converged") {
        missing.push(`detailed status reported daemon-vs-installed-surface half-converged (${statusSignals.surfaceSkew ?? "unknown"})`);
    }
    else if (statusSignals.surfaceConvergeState !== "converged") {
        missing.push("detailed status did not fully prove daemon-vs-installed-surface convergence");
    }
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
        "## Hotfix Boundary",
        ...(surfaceLine === null
            ? ["- hotfix boundary line not reported by detailed status"]
            : [`- ${surfaceLine}`]),
        ...(surfacesLine === null
            ? ["- hotfix paths line not reported by detailed status"]
            : [`- ${surfacesLine}`]),
        ...(surfaceNoteLine === null
            ? ["- hotfix detail line not reported by detailed status"]
            : [`- ${surfaceNoteLine}`]),
        ...(hotfixLine === null
            ? ["- hotfix guidance line not reported by detailed status"]
            : [`- ${hotfixLine}`]),
        "",
        "## Runtime Guard",
        ...(guardLine === null
            ? ["- runtime guard line not reported by detailed status"]
            : [`- ${guardLine}`]),
        "",
        "## Learning Flow",
        ...(learningPathLine === null
            ? ["- learning path line not reported by detailed status"]
            : [`- ${learningPathLine}`]),
        ...(learningFlowLine === null
            ? ["- learning flow line not reported by detailed status"]
            : [`- ${learningFlowLine}`]),
        ...(learningHealthLine === null
            ? ["- learning health line not reported by detailed status"]
            : [`- ${learningHealthLine}`]),
        "",
        "## Learning Attribution",
        ...(feedbackLine === null
            ? ["- feedback line not reported by detailed status"]
            : [`- ${feedbackLine}`]),
        ...(attributionLine === null
            ? ["- attribution line not reported by detailed status"]
            : [`- ${attributionLine}`]),
        ...(attributionCoverageLine === null
            ? ["- attribution coverage line not reported by detailed status"]
            : [`- ${attributionCoverageLine}`]),
        "",
        "## Coverage snapshot",
        `- attached profiles: ${coverageSnapshot.attachedProfileCount}`,
        `- runtime-proven profiles: ${coverageSnapshot.runtimeProvenCount}/${coverageSnapshot.attachedProfileCount}`,
        `- coverage rate: ${coverageSnapshot.coverageRate === null ? "none" : coverageSnapshot.coverageRate.toFixed(3)}`,
        ...(coverageSnapshot.profiles.length === 0
            ? ["- per-profile: none"]
            : coverageSnapshot.profiles.map((entry) => `- ${entry.current ? "*" : ""}${entry.label} coverage=${entry.coverageState} hook=${entry.hookFiles} config=${entry.configLoad} runtime=${entry.runtimeLoad} loadedAt=${entry.loadedAt ?? "none"}`)),
        "",
        "## Hardening snapshot",
        `- status signals: statusOk=${hardeningSnapshot.statusSignals.statusOk} loadProofReady=${hardeningSnapshot.statusSignals.loadProofReady} runtimeProven=${hardeningSnapshot.statusSignals.runtimeProven} serveActivePack=${hardeningSnapshot.statusSignals.serveActivePack} routeFnAvailable=${hardeningSnapshot.statusSignals.routeFnAvailable} surfaceConverged=${hardeningSnapshot.statusSignals.surfaceConverged}`,
        `- surface: boundary=${hardeningSnapshot.surface.boundary ?? "none"} skew=${hardeningSnapshot.surface.skew ?? "none"} converge=${hardeningSnapshot.surface.converge ?? "none"} daemonSource=${hardeningSnapshot.surface.daemonSource ?? "none"} daemon=${hardeningSnapshot.surface.daemon ?? "none"} hook=${hardeningSnapshot.surface.hook ?? "none"} selectedHome=${hardeningSnapshot.surface.selectedHome ?? "none"}`,
        `- serve: state=${hardeningSnapshot.serve.state ?? "none"} failOpen=${hardeningSnapshot.serve.failOpen ?? "none"} hardFail=${hardeningSnapshot.serve.hardFail ?? "none"} usedRouteFn=${hardeningSnapshot.serve.usedRouteFn ?? "none"}`,
        `- attachTruth: current=${hardeningSnapshot.attachTruth.current ?? "none"} hook=${hardeningSnapshot.attachTruth.hook ?? "none"} config=${hardeningSnapshot.attachTruth.config ?? "none"} runtime=${hardeningSnapshot.attachTruth.runtime ?? "none"}`,
        `- proof verdict: ${hardeningSnapshot.verdict.verdict} severity=${hardeningSnapshot.verdict.severity} warnings=${hardeningSnapshot.verdict.warningCount}`,
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

function buildGatewayStatusArgs(profileName, gatewayUrl, gatewayToken) {
    const args = buildGatewayArgs("status", profileName);
    if (gatewayUrl !== null) {
        args.push("--url", gatewayUrl);
    }
    if (gatewayToken !== null) {
        args.push("--token", gatewayToken);
    }
    return args;
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
            "  --gateway-url <url>         Override the gateway-status probe target for proof capture (proof only).",
            "  --gateway-token <token>     Gateway token to use with --gateway-url or other non-default proof probes.",
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
    let gatewayUrl = null;
    let gatewayToken = null;
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
        if (arg === "--gateway-url") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--gateway-url requires a value");
            }
            gatewayUrl = next;
            index += 1;
            continue;
        }
        if (arg === "--gateway-token") {
            const next = argv[index + 1];
            if (next === undefined) {
                throw new Error("--gateway-token requires a value");
            }
            gatewayToken = next;
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
            gatewayUrl,
            gatewayToken,
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
        gatewayUrl,
        gatewayToken,
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
    function addStep(stepId, label, command, args, { skipped = false, skipSummary = "step intentionally skipped" } = {}) {
        if (skipped) {
            steps.push({
                stepId,
                label,
                shellCommand: shellJoin([command, ...args]),
                skipped: true,
                captureState: "complete",
                resultClass: "success",
                summary: skipSummary,
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
    const installCapture = addStep("01-install", "install", cliInvocation.command, [...cliInvocation.args, "install", "--openclaw-home", options.openclawHome, "--json"], { skipped: options.skipInstall === true });
    const installRestartState = options.skipInstall === true || installCapture.exitCode !== 0 || installCapture.error
        ? null
        : extractInstallRestartState(installCapture.stdout);
    let restartSkipSummary = null;
    if (options.skipRestart === true) {
        restartSkipSummary = "step intentionally skipped";
    }
    else if (installRestartState?.performed === true) {
        restartSkipSummary = "step intentionally skipped because install already performed the gateway restart";
    }
    else if (installRestartState?.required === false) {
        restartSkipSummary = "step intentionally skipped because install reported no gateway restart was required";
    }
    else if (gatewayProfile === null) {
        restartSkipSummary = "skipped because exact OpenClaw profile token could not be inferred; avoiding shared-gateway self-interrupt during proof capture";
    }
    addStep("02-restart", "gateway restart", "openclaw", buildGatewayArgs("restart", gatewayProfile), {
        skipped: restartSkipSummary !== null,
        skipSummary: restartSkipSummary ?? undefined
    });
    const gatewayStatusCapture = addStep("03-gateway-status", "gateway status", "openclaw", buildGatewayStatusArgs(
        gatewayProfile,
        normalizeOptionalCliString(options.gatewayUrl ?? null),
        normalizeOptionalCliString(options.gatewayToken ?? null),
    ));
    const pluginInspectCapture = addStep("04-plugin-inspect", "plugin inspect", "openclaw", ["plugins", "inspect", options.pluginId]);
    const statusCapture = addStep("05-detailed-status", "detailed status", cliInvocation.command, [...cliInvocation.args, "status", "--openclaw-home", options.openclawHome, "--detailed"]);
    const gatewayLogPath = extractGatewayLogPath(gatewayStatusCapture.stdout);
    const activationRoot = extractActivationRoot(statusCapture.stdout, options.activationRoot ?? null);
    const statusSignals = extractStatusSignals(statusCapture.stdout);
    const surfaceLine = extractDetailedStatusLine(statusCapture.stdout, "surface");
    const surfacesLine = extractDetailedStatusLine(statusCapture.stdout, "surfaces");
    const surfaceNoteLine = extractDetailedStatusLine(statusCapture.stdout, "surfaceNote");
    const hotfixLine = extractDetailedStatusLine(statusCapture.stdout, "hotfix");
    const attachTruthLine = extractDetailedStatusLine(statusCapture.stdout, "attachTruth");
    const attachedSetLine = extractDetailedStatusLine(statusCapture.stdout, "attachedSet");
    const serveLine = extractDetailedStatusLine(statusCapture.stdout, "serve");
    const routeFnLine = extractDetailedStatusLine(statusCapture.stdout, "routeFn");
    const guardLine = extractDetailedStatusLine(statusCapture.stdout, "guard");
    const learningFlowLine = extractDetailedStatusLine(statusCapture.stdout, "learnFlow");
    const learningHealthLine = extractDetailedStatusLine(statusCapture.stdout, "health");
    const feedbackLine = extractDetailedStatusLine(statusCapture.stdout, "feedback");
    const attributionLine = extractDetailedStatusLine(statusCapture.stdout, "attribution");
    const attributionCoverageLine = extractDetailedStatusLine(statusCapture.stdout, "attrCover");
    const learningPathLine = extractDetailedStatusLine(statusCapture.stdout, "path");
    const runtimeLoadProofPath = normalizeReportedProofPath(statusSignals.proofPath)
        ?? path.join(activationRoot, "attachment-truth", "runtime-load-proofs.json");
    const runtimeLoadProofSnapshot = readJsonSnapshot(runtimeLoadProofPath);
    const gatewayLogText = readTextIfExists(gatewayLogPath);
    const breadcrumbs = extractStartupBreadcrumbs(gatewayLogText, bundleStartedAt);
    const coverageSnapshot = buildCoverageSnapshot({
        attachedSetLine,
        runtimeLoadProofSnapshot,
        openclawHome: options.openclawHome,
    });
    writeText(path.join(bundleDir, "extracted-startup-breadcrumbs.log"), breadcrumbs.all.length === 0
        ? "<no matching breadcrumbs found>\n"
        : `${breadcrumbs.all.map((entry) => entry.line).join("\n")}\n`);
    writeJson(path.join(bundleDir, "runtime-load-proof.json"), runtimeLoadProofSnapshot);
    writeJson(path.join(bundleDir, "coverage-snapshot.json"), coverageSnapshot);
    const verdict = buildVerdict({
        steps,
        gatewayStatus: gatewayStatusCapture.stdout,
        pluginInspect: pluginInspectCapture.stdout,
        statusSignals,
        breadcrumbs,
        runtimeLoadProofSnapshot,
        coverageSnapshot,
        openclawHome: options.openclawHome,
    });
    const hardeningSnapshot = buildHardeningSnapshot({
        attachTruthLine,
        serveLine,
        routeFnLine,
        surfaceLine,
        verdict,
        statusSignals,
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
        coverageSnapshot,
        hardeningSnapshot,
        breadcrumbs: {
            allCount: breadcrumbs.all.length,
            postBundleCount: breadcrumbs.afterBundleStart.length,
            postBundleKinds: breadcrumbs.afterBundleStart.map((entry) => entry.kind),
        },
        runtimeLoadProofPath,
        runtimeLoadProofError: runtimeLoadProofSnapshot.error,
        surfaceLine,
        surfacesLine,
        surfaceNoteLine,
        hotfixLine,
        guardLine,
        learningFlowLine,
        learningHealthLine,
        feedbackLine,
        attributionLine,
        attributionCoverageLine,
        learningPathLine,
    });
    writeJson(path.join(bundleDir, "hardening-snapshot.json"), hardeningSnapshot);
    writeText(path.join(bundleDir, "summary.md"), buildSummary({
        options,
        steps,
        verdict,
        gatewayStatusText: gatewayStatusCapture.stdout,
        pluginInspectText: pluginInspectCapture.stdout,
        statusSignals,
        breadcrumbs,
        runtimeLoadProofSnapshot,
        surfaceLine,
        surfacesLine,
        surfaceNoteLine,
        hotfixLine,
        guardLine,
        learningFlowLine,
        learningHealthLine,
        feedbackLine,
        attributionLine,
        attributionCoverageLine,
        learningPathLine,
        coverageSnapshot,
        hardeningSnapshot,
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
        coverageSnapshot,
        hardeningSnapshot,
        verdict,
        statusSignals,
        surfaceLine,
        surfacesLine,
        surfaceNoteLine,
        hotfixLine,
        guardLine,
        learningFlowLine,
        learningHealthLine,
        feedbackLine,
        attributionLine,
        attributionCoverageLine,
        learningPathLine,
        steps,
        summaryPath: path.join(bundleDir, "summary.md"),
        stepsPath: path.join(bundleDir, "steps.json"),
        verdictPath: path.join(bundleDir, "verdict.json"),
        breadcrumbPath: path.join(bundleDir, "extracted-startup-breadcrumbs.log"),
        runtimeLoadProofSnapshotPath: path.join(bundleDir, "runtime-load-proof.json"),
        coverageSnapshotPath: path.join(bundleDir, "coverage-snapshot.json"),
        hardeningSnapshotPath: path.join(bundleDir, "hardening-snapshot.json"),
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
        `  Coverage snapshot: ${result.coverageSnapshotPath}`,
        `  Hardening snapshot: ${result.hardeningSnapshotPath}`,
    ];
    return lines.join("\n");
}
