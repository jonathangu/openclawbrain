/**
 * macOS launchd daemon management for OpenClawBrain.
 *
 * Manages macOS launchd user agents that run `openclawbrain watch` in the background.
 * Service identity is derived per activation root so one profile/service boundary
 * does not collide with another.
 *
 * Commands:
 *   daemon start  — generate and load a launchd plist
 *   daemon stop   — unload the plist
 *   daemon status — show running/stopped + PID + launch command + last log lines
 *   daemon logs   — tail the daemon log file
 */
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, unlinkSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadTeacherSurface, resolveWatchSessionTailCursorPath, resolveWatchStateRoot, resolveWatchTeacherSnapshotPath } from "./index.js";
const LABEL_PREFIX = "com.openclawbrain.daemon";
const LOG_ROOT_DIRNAME = "daemon";
const DEFAULT_SCAN_ROOT_DIRNAME = "event-exports";
const BASELINE_STATE_BASENAME = "baseline-state.json";
const SCANNER_CHECKPOINT_BASENAME = ".openclawbrain-scanner-checkpoint.json";
const CLI_PACKAGE_NAME = "@openclawbrain/cli";
const CLI_BIN_NAME = "openclawbrain";
const DEFAULT_DAEMON_COMMAND_RUNNER = (command) => execSync(command, {
    encoding: "utf8",
    stdio: "pipe",
});
let daemonCommandRunner = DEFAULT_DAEMON_COMMAND_RUNNER;
function getHomeDir() {
    return process.env.HOME ?? process.env.USERPROFILE ?? "~";
}
function canonicalizeActivationRoot(activationRoot) {
    const resolvedActivationRoot = path.resolve(activationRoot);
    return existsSync(resolvedActivationRoot) ? safeRealpath(resolvedActivationRoot) : resolvedActivationRoot;
}
function sanitizeActivationRootSlug(value) {
    const sanitized = value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
    return sanitized.length > 0 ? sanitized.slice(0, 32) : "activation-root";
}
export function buildDaemonServiceIdentity(activationRoot) {
    const requestedActivationRoot = path.resolve(activationRoot);
    const canonicalActivationRoot = canonicalizeActivationRoot(requestedActivationRoot);
    const activationRootHash = createHash("sha256").update(canonicalActivationRoot).digest("hex").slice(0, 12);
    const activationRootSlug = sanitizeActivationRootSlug(path.basename(canonicalActivationRoot));
    const label = `${LABEL_PREFIX}.${activationRootSlug}.${activationRootHash}`;
    const plistFilename = `${label}.plist`;
    return {
        requestedActivationRoot,
        canonicalActivationRoot,
        activationRootHash,
        activationRootSlug,
        label,
        plistFilename,
        plistPath: path.join(getHomeDir(), "Library", "LaunchAgents", plistFilename),
        logPath: path.join(getHomeDir(), ".openclawbrain", LOG_ROOT_DIRNAME, `${activationRootSlug}-${activationRootHash}.log`)
    };
}
export function setDaemonCommandRunnerForTesting(runner) {
    daemonCommandRunner = runner ?? DEFAULT_DAEMON_COMMAND_RUNNER;
}
function readPackageMetadata(packageRoot) {
    if (packageRoot === null) {
        return null;
    }
    const packageJsonPath = path.join(packageRoot, "package.json");
    if (!existsSync(packageJsonPath)) {
        return null;
    }
    try {
        const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
        const name = typeof packageJson.name === "string" && packageJson.name.trim().length > 0
            ? packageJson.name.trim()
            : null;
        const version = typeof packageJson.version === "string" && packageJson.version.trim().length > 0
            ? packageJson.version.trim()
            : null;
        if (name === null) {
            return null;
        }
        return { name, version };
    }
    catch {
        return null;
    }
}
function isCliScriptPath(filePath) {
    const basename = path.basename(filePath);
    return basename === "cli.js" || basename === "cli.cjs" || basename === "cli.mjs";
}
function isNodeExecutablePath(filePath) {
    return /^node(?:\.exe)?$/i.test(path.basename(filePath));
}
function isNpxCachePath(filePath) {
    const resolvedPath = safeRealpath(path.resolve(filePath));
    return resolvedPath.split(path.sep).includes("_npx");
}
function formatCommandArgument(value) {
    return /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : JSON.stringify(value);
}
function formatCommand(programArguments) {
    return programArguments.map((argument) => formatCommandArgument(argument)).join(" ");
}
function escapePlistString(value) {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");
}
function unescapePlistString(value) {
    return value
        .replace(/&apos;/g, "'")
        .replace(/&quot;/g, "\"")
        .replace(/&gt;/g, ">")
        .replace(/&lt;/g, "<")
        .replace(/&amp;/g, "&");
}
function getCommandPaths(commandName) {
    try {
        return daemonCommandRunner(`which -a ${commandName}`)
            .split("\n")
            .map((entry) => entry.trim())
            .filter((entry) => entry.length > 0);
    }
    catch {
        try {
            const resolved = daemonCommandRunner(`command -v ${commandName}`).trim();
            return resolved.length > 0 ? [resolved] : [];
        }
        catch {
            return [];
        }
    }
}
function safeRealpath(filePath) {
    try {
        return realpathSync(filePath);
    }
    catch {
        return filePath;
    }
}
function resolvePackageRoot(startDir) {
    let currentDir = path.resolve(startDir);
    while (true) {
        const packageJsonPath = path.join(currentDir, "package.json");
        if (existsSync(packageJsonPath)) {
            try {
                const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
                if (packageJson.name === "@openclawbrain/openclaw" || packageJson.name === "@openclawbrain/cli") {
                    return currentDir;
                }
            }
            catch {
                // Ignore malformed package.json while searching upward for the real package root.
            }
        }
        const parentDir = path.dirname(currentDir);
        if (parentDir === currentDir) {
            return null;
        }
        currentDir = parentDir;
    }
}
function resolveCliScriptCandidate(candidatePath) {
    if (typeof candidatePath !== "string" || candidatePath.trim().length === 0) {
        return null;
    }
    const absoluteCandidate = path.resolve(candidatePath);
    if (!existsSync(absoluteCandidate)) {
        return null;
    }
    const resolvedCandidate = safeRealpath(absoluteCandidate);
    if (!isCliScriptPath(resolvedCandidate)) {
        return null;
    }
    return resolvedCandidate;
}
function resolveCliPackageRoot(startDir) {
    const packageRoot = resolvePackageRoot(startDir);
    const packageMetadata = readPackageMetadata(packageRoot);
    if (packageMetadata?.name === CLI_PACKAGE_NAME) {
        return packageRoot;
    }
    if (packageRoot !== null) {
        const siblingCliRoot = path.join(path.dirname(packageRoot), "cli");
        const siblingMetadata = readPackageMetadata(siblingCliRoot);
        if (siblingMetadata?.name === CLI_PACKAGE_NAME) {
            return siblingCliRoot;
        }
    }
    return null;
}
function readNearestPackageMetadataForPath(filePath) {
    if (typeof filePath !== "string" || filePath.trim().length === 0) {
        return null;
    }
    let currentDir = path.dirname(path.resolve(filePath));
    while (true) {
        const packageMetadata = readPackageMetadata(currentDir);
        if (packageMetadata !== null) {
            return {
                root: currentDir,
                ...packageMetadata,
            };
        }
        const parentDir = path.dirname(currentDir);
        if (parentDir === currentDir) {
            return null;
        }
        currentDir = parentDir;
    }
}
function resolveDaemonPackageManagerLaunchSpec(moduleDir) {
    const cliPackageRoot = resolveCliPackageRoot(moduleDir);
    const cliPackageMetadata = readPackageMetadata(cliPackageRoot);
    if (cliPackageMetadata === null) {
        return null;
    }
    const npmPath = getCommandPaths("npm").find((candidate) => !isNpxCachePath(candidate)) ?? null;
    if (npmPath === null) {
        return null;
    }
    const packageSpec = cliPackageMetadata.version === null
        ? CLI_PACKAGE_NAME
        : `${CLI_PACKAGE_NAME}@${cliPackageMetadata.version}`;
    return {
        programArguments: [npmPath, "exec", "--yes", `--package=${packageSpec}`, "--", CLI_BIN_NAME],
        runtimePath: npmPath,
        runtimePackageSpec: packageSpec,
    };
}
function getOpenclawbrainCliScriptPathCandidates() {
    const moduleFilePath = fileURLToPath(import.meta.url);
    const moduleDir = path.dirname(moduleFilePath);
    const packageRoot = resolvePackageRoot(moduleDir);
    return [
        process.argv[1],
        path.join(moduleDir, "cli.js"),
        packageRoot === null ? null : path.join(packageRoot, "dist", "src", "cli.js")
    ];
}
function buildDaemonLaunchProgramArguments(serviceIdentity, programArguments) {
    return [...programArguments, "watch", "--activation-root", serviceIdentity.requestedActivationRoot];
}
function describeDaemonProgramArguments(programArguments) {
    if (programArguments === null || programArguments.length === 0) {
        return {
            configuredProgramArguments: null,
            configuredCommand: null,
            configuredRuntimePath: null,
            configuredRuntimePackageSpec: null,
            configuredRuntimePackageName: null,
            configuredRuntimePackageVersion: null,
            configuredRuntimeLooksEphemeral: null
        };
    }
    const runtimePath = programArguments.length >= 2 && isNodeExecutablePath(programArguments[0]) && isCliScriptPath(programArguments[1])
        ? programArguments[1]
        : programArguments[0];
    const runtimePackageSpec = programArguments.find((argument) => argument.startsWith("--package="))?.slice("--package=".length) ?? null;
    const runtimePackageMetadata = runtimePath === null ? null : readNearestPackageMetadataForPath(runtimePath);
    return {
        configuredProgramArguments: programArguments,
        configuredCommand: formatCommand(programArguments),
        configuredRuntimePath: runtimePath,
        configuredRuntimePackageSpec: runtimePackageSpec,
        configuredRuntimePackageName: runtimePackageMetadata?.name ?? null,
        configuredRuntimePackageVersion: runtimePackageMetadata?.version ?? null,
        configuredRuntimeLooksEphemeral: runtimePath === null ? null : isNpxCachePath(runtimePath)
    };
}
function buildDaemonHotfixBoundary(inspection) {
    return {
        surface: "daemon_runtime",
        separateFromInstalledHookSurface: true,
        runtimePath: inspection.configuredRuntimePath ?? null,
        guidance: "Patch this daemon runtime path for background watch/learner fixes. Use `openclawbrain status --openclaw-home <path> --detailed` to inspect the separate installed hook/runtime-guard surface before patching OpenClaw load behavior.",
        detail: inspection.configuredRuntimePath === null
            ? "daemon status is only reporting the background watch surface; no configured runtime path is visible yet."
            : `daemon status is reporting the background watch runtime at ${inspection.configuredRuntimePath}; installed hook/runtime-guard paths live on the OpenClaw profile side.`
    };
}
function resolveDaemonProgramArguments() {
    for (const candidate of getOpenclawbrainCliScriptPathCandidates()) {
        const cliScriptPath = resolveCliScriptCandidate(candidate);
        if (cliScriptPath !== null && !isNpxCachePath(cliScriptPath)) {
            return {
                programArguments: [process.execPath, cliScriptPath],
                runtimePath: cliScriptPath,
                runtimePackageSpec: null,
            };
        }
    }
    const durableBinPath = getCommandPaths(CLI_BIN_NAME).find((candidate) => !isNpxCachePath(candidate)) ?? null;
    if (durableBinPath !== null) {
        return {
            programArguments: [durableBinPath],
            runtimePath: durableBinPath,
            runtimePackageSpec: null,
        };
    }
    const moduleFilePath = fileURLToPath(import.meta.url);
    const moduleDir = path.dirname(moduleFilePath);
    const packageManagerLaunchSpec = resolveDaemonPackageManagerLaunchSpec(moduleDir);
    if (packageManagerLaunchSpec !== null) {
        return packageManagerLaunchSpec;
    }
    return null;
}
function buildPlistXml(serviceIdentity, programArguments) {
    const logPath = serviceIdentity.logPath;
    const homeDir = getHomeDir();
    const daemonProgramArguments = buildDaemonLaunchProgramArguments(serviceIdentity, programArguments)
        .map((argument) => `    <string>${escapePlistString(argument)}</string>`)
        .join("\n");
    return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${escapePlistString(serviceIdentity.label)}</string>
  <key>ProgramArguments</key>
  <array>
${daemonProgramArguments}
  </array>
  <key>WorkingDirectory</key>
  <string>${escapePlistString(serviceIdentity.requestedActivationRoot)}</string>
  <key>StandardOutPath</key>
  <string>${escapePlistString(logPath)}</string>
  <key>StandardErrorPath</key>
  <string>${escapePlistString(logPath)}</string>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>${escapePlistString(homeDir)}</string>
    <key>PATH</key>
    <string>${escapePlistString("/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin")}</string>
  </dict>
</dict>
</plist>
`;
}
function ensureLogDir(logPath) {
    const logDir = path.dirname(logPath);
    if (!existsSync(logDir)) {
        mkdirSync(logDir, { recursive: true });
    }
}
function hasLaunchctl() {
    try {
        return daemonCommandRunner("command -v launchctl").trim().length > 0;
    }
    catch {
        return false;
    }
}
function launchctlLoad(plistPath) {
    try {
        daemonCommandRunner(`launchctl load -w ${JSON.stringify(plistPath)}`);
        return { ok: true, message: "Daemon started." };
    }
    catch (err) {
        const message = err instanceof Error ? err.stderr?.toString() ?? err.message : String(err);
        return { ok: false, message: `Failed to load plist: ${message}` };
    }
}
function launchctlUnload(plistPath) {
    try {
        daemonCommandRunner(`launchctl unload ${JSON.stringify(plistPath)}`);
        return { ok: true, message: "Daemon stopped." };
    }
    catch (err) {
        const message = err instanceof Error ? err.stderr?.toString() ?? err.message : String(err);
        return { ok: false, message: `Failed to unload plist: ${message}` };
    }
}
function getLaunchctlInfo(label) {
    try {
        const output = daemonCommandRunner("launchctl list");
        for (const line of output.split("\n")) {
            if (line.includes(label)) {
                const parts = line.trim().split(/\s+/);
                const pidStr = parts[0];
                const pid = pidStr && pidStr !== "-" ? parseInt(pidStr, 10) : null;
                return { running: pid !== null && !isNaN(pid), pid: pid !== null && !isNaN(pid) ? pid : null };
            }
        }
    }
    catch {
        // launchctl list failed — treat as not running
    }
    return { running: false, pid: null };
}
function inspectManagedLearnerServiceInternal(activationRoot) {
    const serviceIdentity = buildDaemonServiceIdentity(activationRoot);
    const configuredActivationRoot = readDaemonActivationRoot(serviceIdentity.plistPath);
    const configuredProgramArguments = readDaemonProgramArguments(serviceIdentity.plistPath);
    const info = getLaunchctlInfo(serviceIdentity.label);
    return {
        requestedActivationRoot: serviceIdentity.requestedActivationRoot,
        canonicalActivationRoot: serviceIdentity.canonicalActivationRoot,
        serviceLabel: serviceIdentity.label,
        plistPath: serviceIdentity.plistPath,
        logPath: serviceIdentity.logPath,
        installed: existsSync(serviceIdentity.plistPath),
        running: info.running,
        pid: info.pid,
        configuredActivationRoot,
        ...describeDaemonProgramArguments(configuredProgramArguments),
        matchesRequestedActivationRoot: configuredActivationRoot === null
            ? null
            : canonicalizeActivationRoot(configuredActivationRoot) === serviceIdentity.canonicalActivationRoot,
        launchctlAvailable: hasLaunchctl()
    };
}
function startManagedLearnerService(activationRoot) {
    const inspectionBeforeStart = inspectManagedLearnerServiceInternal(activationRoot);
    const serviceIdentity = buildDaemonServiceIdentity(activationRoot);
    if (!inspectionBeforeStart.launchctlAvailable) {
        return {
            ok: false,
            message: "launchctl is unavailable on this host",
            inspection: inspectionBeforeStart
        };
    }
    const launchSpec = resolveDaemonProgramArguments();
    if (launchSpec === null) {
        return {
            ok: false,
            message: "Failed to resolve an OpenClawBrain CLI launch command without pinning an npx cache path.",
            inspection: inspectionBeforeStart
        };
    }
    const launchAgentsDir = path.dirname(inspectionBeforeStart.plistPath);
    if (!existsSync(launchAgentsDir)) {
        mkdirSync(launchAgentsDir, { recursive: true });
    }
    ensureLogDir(serviceIdentity.logPath);
    const plistContent = buildPlistXml(serviceIdentity, launchSpec.programArguments);
    writeFileSync(inspectionBeforeStart.plistPath, plistContent, "utf8");
    const result = launchctlLoad(inspectionBeforeStart.plistPath);
    if (!result.ok && !inspectionBeforeStart.installed) {
        try {
            unlinkSync(inspectionBeforeStart.plistPath);
        }
        catch {
            // Best effort cleanup for failed first-time auto-start attempts.
        }
    }
    return {
        ok: result.ok,
        message: result.message,
        inspection: inspectManagedLearnerServiceInternal(activationRoot)
    };
}
function stopManagedLearnerService(activationRoot) {
    const inspectionBeforeStop = inspectManagedLearnerServiceInternal(activationRoot);
    if (!inspectionBeforeStop.installed) {
        return {
            ok: true,
            message: "No daemon plist found.",
            inspection: inspectionBeforeStop
        };
    }
    if (!inspectionBeforeStop.launchctlAvailable) {
        return {
            ok: false,
            message: "launchctl is unavailable on this host",
            inspection: inspectionBeforeStop
        };
    }
    const result = launchctlUnload(inspectionBeforeStop.plistPath);
    if (result.ok) {
        try {
            unlinkSync(inspectionBeforeStop.plistPath);
        }
        catch {
            // best effort
        }
    }
    return {
        ok: result.ok,
        message: result.message,
        inspection: inspectManagedLearnerServiceInternal(activationRoot)
    };
}
export function inspectManagedLearnerService(activationRoot) {
    return inspectManagedLearnerServiceInternal(activationRoot);
}
export function ensureManagedLearnerServiceForActivationRoot(activationRoot) {
    const inspection = inspectManagedLearnerServiceInternal(activationRoot);
    if (inspection.matchesRequestedActivationRoot === true && inspection.running) {
        return {
            state: "ensured",
            reason: "already_running_exact_root",
            detail: `Learner auto-start already ensured for ${inspection.requestedActivationRoot}; the matching background learner service is running.`,
            inspection
        };
    }
    const startResult = startManagedLearnerService(activationRoot);
    if (startResult.ok) {
        return {
            state: "started",
            reason: "started_exact_root",
            detail: `Started the background learner service for ${startResult.inspection.requestedActivationRoot}; passive learning can begin for this attached profile now.`,
            inspection: startResult.inspection
        };
    }
    const reason = !inspection.launchctlAvailable
        ? "launchctl_unavailable"
        : startResult.message === "Failed to resolve an OpenClawBrain CLI launch command without pinning an npx cache path."
            ? "launch_command_unavailable"
            : "launch_failed";
    return {
        state: "deferred",
        reason,
        detail: `Learner auto-start deferred for ${inspection.requestedActivationRoot}: ${startResult.message}`,
        inspection: startResult.inspection
    };
}
export function removeManagedLearnerServiceForActivationRoot(activationRoot) {
    const inspection = inspectManagedLearnerServiceInternal(activationRoot);
    if (!inspection.installed) {
        return {
            state: "already_absent",
            reason: "not_installed",
            detail: `No background learner service is installed for ${inspection.requestedActivationRoot}.`,
            inspection
        };
    }
    if (inspection.matchesRequestedActivationRoot === false) {
        return {
            state: "preserved",
            reason: "configured_root_mismatch",
            detail: `Preserved the background learner service because ${inspection.plistPath} is configured for ${inspection.configuredActivationRoot}, ` +
                `not the requested exact root ${inspection.requestedActivationRoot}.`,
            inspection
        };
    }
    const stopResult = stopManagedLearnerService(activationRoot);
    if (stopResult.ok) {
        return {
            state: "removed",
            reason: "removed_exact_root",
            detail: `Removed the background learner service for ${stopResult.inspection.requestedActivationRoot}.`,
            inspection: stopResult.inspection
        };
    }
    return {
        state: "preserved",
        reason: inspection.launchctlAvailable ? "stop_failed" : "launchctl_unavailable",
        detail: `Preserved the background learner service for ${inspection.requestedActivationRoot}: ${stopResult.message}`,
        inspection: stopResult.inspection
    };
}
function readLastLines(filePath, count) {
    if (!existsSync(filePath))
        return [];
    try {
        const content = readFileSync(filePath, "utf8");
        const lines = content.split("\n");
        // Remove trailing empty line from split
        if (lines.length > 0 && lines[lines.length - 1] === "") {
            lines.pop();
        }
        return lines.slice(-count);
    }
    catch {
        return [];
    }
}
function readOptionalJsonFile(filePath) {
    if (!existsSync(filePath)) {
        return null;
    }
    try {
        return JSON.parse(readFileSync(filePath, "utf8"));
    }
    catch {
        return null;
    }
}
function readDaemonProgramArguments(plistPath) {
    if (!existsSync(plistPath)) {
        return null;
    }
    try {
        const plist = readFileSync(plistPath, "utf8");
        const sectionMatch = plist.match(/<key>ProgramArguments<\/key>\s*<array>([\s\S]*?)<\/array>/);
        if (sectionMatch === null) {
            return null;
        }
        const programArguments = [];
        const stringPattern = /<string>([\s\S]*?)<\/string>/g;
        let match = stringPattern.exec(sectionMatch[1]);
        while (match !== null) {
            programArguments.push(unescapePlistString(match[1]));
            match = stringPattern.exec(sectionMatch[1]);
        }
        return programArguments.length > 0 ? programArguments : null;
    }
    catch {
        return null;
    }
}
function readDaemonActivationRoot(plistPath) {
    const programArguments = readDaemonProgramArguments(plistPath);
    if (programArguments !== null) {
        const activationRootIndex = programArguments.indexOf("--activation-root");
        if (activationRootIndex !== -1) {
            return programArguments[activationRootIndex + 1] ?? null;
        }
    }
    return null;
}
function getWatchStatePaths(activationRoot) {
    if (activationRoot === null) {
        return {
            watchStateRoot: null,
            scanRoot: null,
            scannerCheckpointPath: null,
            sessionTailCursorPath: null,
            teacherSnapshotPath: null,
            baselineStatePath: null
        };
    }
    const scanRoot = path.join(activationRoot, DEFAULT_SCAN_ROOT_DIRNAME);
    return {
        watchStateRoot: resolveWatchStateRoot(activationRoot),
        scanRoot,
        scannerCheckpointPath: path.join(scanRoot, SCANNER_CHECKPOINT_BASENAME),
        sessionTailCursorPath: resolveWatchSessionTailCursorPath(activationRoot),
        teacherSnapshotPath: resolveWatchTeacherSnapshotPath(activationRoot),
        baselineStatePath: path.join(activationRoot, BASELINE_STATE_BASENAME)
    };
}
function readWatchStateSummary(activationRoot) {
    const paths = getWatchStatePaths(activationRoot);
    const cursorFile = paths.sessionTailCursorPath === null
        ? null
        : readOptionalJsonFile(paths.sessionTailCursorPath);
    const teacherSurface = paths.teacherSnapshotPath === null
        ? null
        : loadTeacherSurface(paths.teacherSnapshotPath);
    const teacherSnapshotFile = teacherSurface?.watchSnapshot ?? null;
    const teacherSnapshot = teacherSurface?.snapshot ?? null;
    const resolvedScanRoot = typeof teacherSnapshotFile?.scanRoot === "string" && teacherSnapshotFile.scanRoot.trim().length > 0
        ? teacherSnapshotFile.scanRoot
        : paths.scanRoot;
    const scannerCheckpointPath = typeof teacherSnapshotFile?.scannerCheckpointPath === "string" && teacherSnapshotFile.scannerCheckpointPath.trim().length > 0
        ? teacherSnapshotFile.scannerCheckpointPath
        : resolvedScanRoot === null ? null : path.join(resolvedScanRoot, SCANNER_CHECKPOINT_BASENAME);
    const scannerCheckpointFile = teacherSnapshotFile?.scannerCheckpoint ??
        (scannerCheckpointPath === null
            ? null
            : readOptionalJsonFile(scannerCheckpointPath));
    const baselineFile = paths.baselineStatePath === null ? null : readOptionalJsonFile(paths.baselineStatePath);
    const lastMaterializationPackId = teacherSnapshot?.learner.lastMaterialization?.candidate.summary.packId ?? null;
    const teacherSummary = teacherSnapshotFile?.teacher;
    const learningSummary = teacherSnapshotFile?.learning;
    const teacherDiagnostics = teacherSnapshot?.diagnostics;
    const teacherState = teacherSnapshot?.state;
    const teacherLearnerState = teacherSnapshot?.learner.state;
    return {
        watchStateRoot: paths.watchStateRoot,
        scanRoot: resolvedScanRoot,
        scannerCheckpoint: {
            path: scannerCheckpointPath,
            exists: scannerCheckpointPath !== null && existsSync(scannerCheckpointPath),
            updatedAt: typeof scannerCheckpointFile?.updatedAt === "string" ? scannerCheckpointFile.updatedAt : null,
            processedExportDigestCount: Array.isArray(scannerCheckpointFile?.processedExportDigests)
                ? scannerCheckpointFile.processedExportDigests.length
                : null,
            scanPasses: typeof scannerCheckpointFile?.stats?.scanPasses === "number" ? scannerCheckpointFile.stats.scanPasses : null,
            liveBundlesScanned: typeof scannerCheckpointFile?.stats?.liveBundlesScanned === "number" ? scannerCheckpointFile.stats.liveBundlesScanned : null,
            backfillBundlesScanned: typeof scannerCheckpointFile?.stats?.backfillBundlesScanned === "number" ? scannerCheckpointFile.stats.backfillBundlesScanned : null,
        },
        sessionTailCursor: {
            path: paths.sessionTailCursorPath,
            exists: paths.sessionTailCursorPath !== null && existsSync(paths.sessionTailCursorPath),
            updatedAt: typeof teacherSnapshotFile?.sessionTailCursorUpdatedAt === "string"
                ? teacherSnapshotFile.sessionTailCursorUpdatedAt
                : typeof cursorFile?.updatedAt === "string"
                    ? cursorFile.updatedAt
                    : null,
            sessionCount: typeof teacherSnapshotFile?.sessionTailSessionsTracked === "number"
                ? teacherSnapshotFile.sessionTailSessionsTracked
                : Array.isArray(cursorFile?.cursor)
                    ? cursorFile.cursor.length
                    : null,
            bridgedEventCount: typeof teacherSnapshotFile?.sessionTailBridgedEventCount === "number"
                ? teacherSnapshotFile.sessionTailBridgedEventCount
                : null,
        },
        teacherSnapshot: {
            path: paths.teacherSnapshotPath,
            exists: paths.teacherSnapshotPath !== null && existsSync(paths.teacherSnapshotPath),
            updatedAt: typeof teacherSnapshotFile?.updatedAt === "string" ? teacherSnapshotFile.updatedAt : null,
            sourceKind: teacherSurface?.sourceKind ?? "missing",
            lastRunAt: typeof teacherSnapshotFile?.lastRunAt === "string" ? teacherSnapshotFile.lastRunAt : null,
            scanRoot: resolvedScanRoot,
            artifactCount: typeof teacherSnapshotFile?.teacher?.artifactCount === "number"
                ? teacherSnapshotFile.teacher.artifactCount
                : teacherSnapshot?.teacher.artifactCount ?? null,
            latestFreshness: typeof teacherSnapshotFile?.teacher?.latestFreshness === "string"
                ? teacherSnapshotFile.teacher.latestFreshness
                : teacherSnapshot?.teacher.latestFreshness ?? null,
            replayedBundleCount: typeof teacherSnapshotFile?.replayedBundleCount === "number" ? teacherSnapshotFile.replayedBundleCount : null,
            replayedEventCount: typeof teacherSnapshotFile?.replayedEventCount === "number" ? teacherSnapshotFile.replayedEventCount : null,
            exportedBundleCount: typeof teacherSnapshotFile?.exportedBundleCount === "number" ? teacherSnapshotFile.exportedBundleCount : null,
            exportedEventCount: typeof teacherSnapshotFile?.exportedEventCount === "number" ? teacherSnapshotFile.exportedEventCount : null,
            startupWarningCount: Array.isArray(teacherSnapshotFile?.startupWarnings) ? teacherSnapshotFile.startupWarnings.length : null,
            lastTeacherError: typeof teacherSnapshotFile?.lastTeacherError === "string" ? teacherSnapshotFile.lastTeacherError : null,
            localSessionTailNoopReason: typeof teacherSnapshotFile?.localSessionTailNoopReason === "string" ? teacherSnapshotFile.localSessionTailNoopReason : null,
            learningCadence: typeof teacherSnapshotFile?.labeling?.learningCadence === "string" ? teacherSnapshotFile.labeling.learningCadence : null,
            scanPolicy: typeof teacherSnapshotFile?.labeling?.scanPolicy === "string" ? teacherSnapshotFile.labeling.scanPolicy : null,
            liveSlicesPerCycle: typeof teacherSnapshotFile?.labeling?.liveSlicesPerCycle === "number"
                ? teacherSnapshotFile.labeling.liveSlicesPerCycle
                : null,
            backfillSlicesPerCycle: typeof teacherSnapshotFile?.labeling?.backfillSlicesPerCycle === "number"
                ? teacherSnapshotFile.labeling.backfillSlicesPerCycle
                : null,
            failureMode: typeof teacherSnapshotFile?.failure?.mode === "string" ? teacherSnapshotFile.failure.mode : null,
            failureDetail: typeof teacherSnapshotFile?.failure?.detail === "string" ? teacherSnapshotFile.failure.detail : null,
            lastHandledMaterializationPackId: typeof teacherSnapshotFile?.lastHandledMaterializationPackId === "string"
                ? teacherSnapshotFile.lastHandledMaterializationPackId
                : null,
            lastMaterializationPackId: typeof lastMaterializationPackId === "string" ? lastMaterializationPackId : null,
            cadence: {
                acceptedExportCount: typeof teacherSummary?.acceptedExportCount === "number"
                    ? teacherSummary.acceptedExportCount
                    : typeof teacherDiagnostics?.acceptedExportCount === "number"
                        ? teacherDiagnostics.acceptedExportCount
                        : null,
                processedExportCount: typeof teacherSummary?.processedExportCount === "number"
                    ? teacherSummary.processedExportCount
                    : typeof teacherDiagnostics?.processedExportCount === "number"
                        ? teacherDiagnostics.processedExportCount
                        : null,
                duplicateExportCount: typeof teacherSummary?.duplicateExportCount === "number"
                    ? teacherSummary.duplicateExportCount
                    : typeof teacherDiagnostics?.duplicateExportCount === "number"
                        ? teacherDiagnostics.duplicateExportCount
                        : null,
                droppedExportCount: typeof teacherSummary?.droppedExportCount === "number"
                    ? teacherSummary.droppedExportCount
                    : typeof teacherDiagnostics?.droppedExportCount === "number"
                        ? teacherDiagnostics.droppedExportCount
                        : null,
                emittedArtifactCount: typeof teacherSummary?.emittedArtifactCount === "number"
                    ? teacherSummary.emittedArtifactCount
                    : typeof teacherDiagnostics?.emittedArtifactCount === "number"
                        ? teacherDiagnostics.emittedArtifactCount
                        : null,
                dedupedArtifactCount: typeof teacherSummary?.dedupedArtifactCount === "number"
                    ? teacherSummary.dedupedArtifactCount
                    : typeof teacherDiagnostics?.dedupedArtifactCount === "number"
                        ? teacherDiagnostics.dedupedArtifactCount
                        : null,
                seenExportDigestCount: Array.isArray(teacherState?.seenExportDigests)
                    ? teacherState.seenExportDigests.length
                    : null,
                materializationCount: typeof learningSummary?.materializationCount === "number"
                    ? learningSummary.materializationCount
                    : typeof teacherLearnerState?.materializationCount === "number"
                        ? teacherLearnerState.materializationCount
                        : null,
                lastProcessedAt: typeof teacherSummary?.lastProcessedAt === "string"
                    ? teacherSummary.lastProcessedAt
                    : typeof teacherDiagnostics?.lastProcessedAt === "string"
                        ? teacherDiagnostics.lastProcessedAt
                        : null,
                lastMaterializedAt: typeof learningSummary?.lastMaterializedAt === "string"
                    ? learningSummary.lastMaterializedAt
                    : typeof teacherLearnerState?.lastMaterializedAt === "string"
                        ? teacherLearnerState.lastMaterializedAt
                        : null,
            }
        },
        baselineState: {
            path: paths.baselineStatePath,
            exists: paths.baselineStatePath !== null && existsSync(paths.baselineStatePath),
            lastUpdatedAt: typeof baselineFile?.lastUpdatedAt === "string" ? baselineFile.lastUpdatedAt : null,
            count: typeof baselineFile?.count === "number" ? baselineFile.count : null,
            movingAverage: typeof baselineFile?.movingAverage === "number" ? baselineFile.movingAverage : null,
            alpha: typeof baselineFile?.alpha === "number" ? baselineFile.alpha : null,
        }
    };
}
// ─── Subcommand implementations ─────────────────────────────────────────────
export function daemonStart(activationRoot, json) {
    const serviceIdentity = buildDaemonServiceIdentity(activationRoot);
    const plistPath = serviceIdentity.plistPath;
    const logPath = serviceIdentity.logPath;
    const launchSpec = resolveDaemonProgramArguments();
    if (launchSpec === null) {
        const message = "Failed to resolve an OpenClawBrain CLI launch command without pinning an npx cache path. Install/build @openclawbrain/cli or use a durable repo/runtime checkout.";
        if (json) {
            console.log(JSON.stringify({
                command: "daemon start",
                ok: false,
                plistPath,
                logPath,
                activationRoot: serviceIdentity.requestedActivationRoot,
                serviceLabel: serviceIdentity.label,
                message,
            }, null, 2));
        }
        else {
            console.error(`✗ ${message}`);
        }
        return 1;
    }
    const daemonProgramArguments = buildDaemonLaunchProgramArguments(serviceIdentity, launchSpec.programArguments);
    const daemonLaunchDescription = describeDaemonProgramArguments(daemonProgramArguments);
    // Ensure LaunchAgents dir exists
    const launchAgentsDir = path.dirname(plistPath);
    if (!existsSync(launchAgentsDir)) {
        mkdirSync(launchAgentsDir, { recursive: true });
    }
    ensureLogDir(logPath);
    // Write the plist
    const plistContent = buildPlistXml(serviceIdentity, launchSpec.programArguments);
    writeFileSync(plistPath, plistContent, "utf8");
    // Load it
    const result = launchctlLoad(plistPath);
    if (json) {
        console.log(JSON.stringify({
            command: "daemon start",
            ok: result.ok,
            plistPath,
            logPath,
            activationRoot: serviceIdentity.requestedActivationRoot,
            serviceLabel: serviceIdentity.label,
            ...daemonLaunchDescription,
            message: result.message,
        }, null, 2));
    }
    else {
        if (result.ok) {
            console.log(`✓ Daemon started`);
            console.log(`  Label: ${serviceIdentity.label}`);
            console.log(`  Plist: ${plistPath}`);
            console.log(`  Log:   ${logPath}`);
            console.log(`  Root:  ${serviceIdentity.requestedActivationRoot}`);
            if (daemonLaunchDescription.configuredRuntimePath !== null) {
                const runtimePackageSuffix = daemonLaunchDescription.configuredRuntimePackageSpec === null
                    ? ""
                    : ` (${daemonLaunchDescription.configuredRuntimePackageSpec})`;
                console.log(`  Runtime: ${daemonLaunchDescription.configuredRuntimePath}${runtimePackageSuffix}`);
            }
            if (daemonLaunchDescription.configuredCommand !== null) {
                console.log(`  Command: ${daemonLaunchDescription.configuredCommand}`);
            }
        }
        else {
            console.error(`✗ ${result.message}`);
        }
    }
    return result.ok ? 0 : 1;
}
export function daemonStop(activationRoot, json) {
    const serviceIdentity = buildDaemonServiceIdentity(activationRoot);
    const plistPath = serviceIdentity.plistPath;
    if (!existsSync(plistPath)) {
        const msg = "No daemon plist found. Daemon is not installed.";
        if (json) {
            console.log(JSON.stringify({
                command: "daemon stop",
                ok: false,
                activationRoot: serviceIdentity.requestedActivationRoot,
                serviceLabel: serviceIdentity.label,
                plistPath,
                message: msg
            }, null, 2));
        }
        else {
            console.log(msg);
        }
        return 1;
    }
    const result = launchctlUnload(plistPath);
    // Remove the plist file after unloading
    if (result.ok) {
        try {
            unlinkSync(plistPath);
        }
        catch {
            // best effort
        }
    }
    if (json) {
        console.log(JSON.stringify({
            command: "daemon stop",
            activationRoot: serviceIdentity.requestedActivationRoot,
            serviceLabel: serviceIdentity.label,
            ok: result.ok,
            plistPath,
            message: result.message,
        }, null, 2));
    }
    else {
        if (result.ok) {
            console.log(`✓ Daemon stopped and plist removed.`);
            console.log(`  Label: ${serviceIdentity.label}`);
        }
        else {
            console.error(`✗ ${result.message}`);
        }
    }
    return result.ok ? 0 : 1;
}
export function daemonStatus(activationRoot, json) {
    const serviceIdentity = buildDaemonServiceIdentity(activationRoot);
    const plistPath = serviceIdentity.plistPath;
    const logPath = serviceIdentity.logPath;
    const plistInstalled = existsSync(plistPath);
    const info = getLaunchctlInfo(serviceIdentity.label);
    const lastLogLines = readLastLines(logPath, 5);
    const configuredActivationRoot = readDaemonActivationRoot(plistPath);
    const configuredProgramArguments = readDaemonProgramArguments(plistPath);
    const requestedActivationRoot = serviceIdentity.requestedActivationRoot;
    const watchStatePaths = getWatchStatePaths(requestedActivationRoot);
    const watchState = readWatchStateSummary(requestedActivationRoot);
    const matchesRequestedActivationRoot = configuredActivationRoot === null
        ? null
        : canonicalizeActivationRoot(configuredActivationRoot) === serviceIdentity.canonicalActivationRoot;
    const daemonLaunchDescription = describeDaemonProgramArguments(configuredProgramArguments);
    const hotfixBoundary = buildDaemonHotfixBoundary({
        installed: plistInstalled,
        configuredProgramArguments,
        ...daemonLaunchDescription,
    });
    if (json) {
        console.log(JSON.stringify({
            command: "daemon status",
            installed: plistInstalled,
            running: info.running,
            pid: info.pid,
            serviceLabel: serviceIdentity.label,
            plistPath,
            logPath,
            activationRoot: requestedActivationRoot,
            configuredActivationRoot,
            ...daemonLaunchDescription,
            matchesRequestedActivationRoot,
            ...watchStatePaths,
            watchState,
            lastLogLines,
            hotfixBoundary,
        }, null, 2));
    }
    else {
        const stateIcon = info.running ? "●" : "○";
        const stateText = info.running ? "running" : plistInstalled ? "stopped" : "not installed";
        console.log(`${stateIcon} Daemon: ${stateText}`);
        if (info.pid !== null) {
            console.log(`  PID: ${info.pid}`);
        }
        if (plistInstalled) {
            console.log(`  Label: ${serviceIdentity.label}`);
            console.log(`  Plist: ${plistPath}`);
        }
        console.log(`  Requested root: ${requestedActivationRoot}`);
        if (configuredActivationRoot !== null) {
            console.log(`  Configured root: ${configuredActivationRoot}`);
        }
        if (matchesRequestedActivationRoot === false) {
            console.log("  Requested root does not match the installed daemon plist.");
        }
        if (daemonLaunchDescription.configuredRuntimePath !== null) {
            const runtimePackageSuffix = daemonLaunchDescription.configuredRuntimePackageSpec === null
                ? daemonLaunchDescription.configuredRuntimePackageName === null
                    ? ""
                    : ` (${daemonLaunchDescription.configuredRuntimePackageName}${daemonLaunchDescription.configuredRuntimePackageVersion === null ? "" : `@${daemonLaunchDescription.configuredRuntimePackageVersion}`})`
                : ` (${daemonLaunchDescription.configuredRuntimePackageSpec})`;
            const runtimeWarning = daemonLaunchDescription.configuredRuntimeLooksEphemeral ? " [ephemeral]" : "";
            console.log(`  Runtime: ${daemonLaunchDescription.configuredRuntimePath}${runtimePackageSuffix}${runtimeWarning}`);
        }
        console.log("  Runtime surface: daemon watch/learner runtime");
        console.log(`  Hotfix boundary: ${hotfixBoundary.guidance}`);
        if (configuredProgramArguments !== null && configuredProgramArguments.length > 0) {
            console.log(`  Program: ${configuredProgramArguments[0]}`);
            if (configuredProgramArguments.length > 1) {
                console.log(`  Args: ${configuredProgramArguments.slice(1).map((argument) => formatCommandArgument(argument)).join(" ")}`);
            }
        }
        if (daemonLaunchDescription.configuredCommand !== null) {
            console.log(`  Command: ${daemonLaunchDescription.configuredCommand}`);
        }
        console.log(`  Log: ${logPath}`);
        if (watchState.scanRoot !== null) {
            console.log(`  Scan root: ${watchState.scanRoot}`);
        }
        if (watchState.scannerCheckpoint.path !== null) {
            const checkpointSummary = watchState.scannerCheckpoint.exists
                ? `processed=${watchState.scannerCheckpoint.processedExportDigestCount ?? "?"} passes=${watchState.scannerCheckpoint.scanPasses ?? "?"}`
                : "missing";
            console.log(`  Scanner: ${watchState.scannerCheckpoint.path} (${checkpointSummary})`);
        }
        if (watchState.teacherSnapshot.path !== null) {
            const snapshotSummary = watchState.teacherSnapshot.exists
                ? `updated=${watchState.teacherSnapshot.updatedAt ?? "unknown"} lastRun=${watchState.teacherSnapshot.lastRunAt ?? "unknown"} artifacts=${watchState.teacherSnapshot.artifactCount ?? "?"} replayed=${watchState.teacherSnapshot.replayedBundleCount ?? "?"}/${watchState.teacherSnapshot.replayedEventCount ?? "?"} exported=${watchState.teacherSnapshot.exportedBundleCount ?? "?"}/${watchState.teacherSnapshot.exportedEventCount ?? "?"}`
                : "missing";
            console.log(`  Snapshot: ${watchState.teacherSnapshot.path} (${snapshotSummary})`);
        }
        if (watchState.teacherSnapshot.cadence.processedExportCount !== null) {
            console.log(`  Teacher cadence: processed=${watchState.teacherSnapshot.cadence.processedExportCount} materialized=${watchState.teacherSnapshot.cadence.materializationCount ?? "?"} seen=${watchState.teacherSnapshot.cadence.seenExportDigestCount ?? "?"}`);
        }
        if (watchState.sessionTailCursor.path !== null) {
            const cursorSummary = watchState.sessionTailCursor.exists
                ? `sessions=${watchState.sessionTailCursor.sessionCount ?? "?"} bridged=${watchState.sessionTailCursor.bridgedEventCount ?? "?"} updated=${watchState.sessionTailCursor.updatedAt ?? "unknown"}`
                : "missing";
            console.log(`  Cursor: ${watchState.sessionTailCursor.path} (${cursorSummary})`);
        }
        if (watchState.baselineState.path !== null) {
            const baselineSummary = watchState.baselineState.exists
                ? `count=${watchState.baselineState.count ?? "?"} avg=${watchState.baselineState.movingAverage ?? "?"} updated=${watchState.baselineState.lastUpdatedAt ?? "unknown"}`
                : "missing";
            console.log(`  Baseline: ${watchState.baselineState.path} (${baselineSummary})`);
        }
        if (watchState.teacherSnapshot.lastHandledMaterializationPackId !== null) {
            console.log(`  Last handled pack: ${watchState.teacherSnapshot.lastHandledMaterializationPackId}`);
        }
        if (watchState.teacherSnapshot.lastMaterializationPackId !== null &&
            watchState.teacherSnapshot.lastMaterializationPackId !== watchState.teacherSnapshot.lastHandledMaterializationPackId) {
            console.log(`  Last materialized pack: ${watchState.teacherSnapshot.lastMaterializationPackId}`);
        }
        if (watchState.teacherSnapshot.localSessionTailNoopReason !== null) {
            console.log(`  Tail state: ${watchState.teacherSnapshot.localSessionTailNoopReason}`);
        }
        if (watchState.teacherSnapshot.startupWarningCount !== null && watchState.teacherSnapshot.startupWarningCount > 0) {
            console.log(`  Startup warnings: ${watchState.teacherSnapshot.startupWarningCount}`);
        }
        if (watchState.teacherSnapshot.lastTeacherError !== null) {
            console.log(`  Teacher fail-open: ${watchState.teacherSnapshot.lastTeacherError}`);
        }
        if (watchState.teacherSnapshot.learningCadence !== null || watchState.teacherSnapshot.scanPolicy !== null) {
            console.log(`  Passive labeling: cadence=${watchState.teacherSnapshot.learningCadence ?? "unknown"} scan=${watchState.teacherSnapshot.scanPolicy ?? "unknown"} slices=${watchState.teacherSnapshot.liveSlicesPerCycle ?? "?"}/${watchState.teacherSnapshot.backfillSlicesPerCycle ?? "?"}`);
        }
        if (watchState.teacherSnapshot.failureMode !== null) {
            console.log(`  Failure: ${watchState.teacherSnapshot.failureMode}${watchState.teacherSnapshot.failureDetail === null ? "" : ` (${watchState.teacherSnapshot.failureDetail})`}`);
        }
        if (lastLogLines.length > 0) {
            console.log(`\nRecent log:`);
            for (const line of lastLogLines) {
                console.log(`  ${line}`);
            }
        }
    }
    return 0;
}
export function daemonLogs(activationRoot, json) {
    const serviceIdentity = buildDaemonServiceIdentity(activationRoot);
    const logPath = serviceIdentity.logPath;
    if (!existsSync(logPath)) {
        const msg = `No log file found at ${logPath}`;
        if (json) {
            console.log(JSON.stringify({
                command: "daemon logs",
                ok: false,
                activationRoot: serviceIdentity.requestedActivationRoot,
                serviceLabel: serviceIdentity.label,
                logPath,
                message: msg,
                lines: []
            }, null, 2));
        }
        else {
            console.log(msg);
        }
        return 1;
    }
    const lines = readLastLines(logPath, 50);
    if (json) {
        console.log(JSON.stringify({
            command: "daemon logs",
            ok: true,
            activationRoot: serviceIdentity.requestedActivationRoot,
            serviceLabel: serviceIdentity.label,
            logPath,
            lines
        }, null, 2));
    }
    else {
        if (lines.length === 0) {
            console.log("(log file is empty)");
        }
        else {
            for (const line of lines) {
                console.log(line);
            }
        }
    }
    return 0;
}
export function runDaemonCommand(args) {
    if (args.help) {
        console.log(daemonHelp());
        return 0;
    }
    switch (args.subcommand) {
        case "start": {
            return daemonStart(args.activationRoot, args.json);
        }
        case "stop":
            return daemonStop(args.activationRoot, args.json);
        case "status":
            return daemonStatus(args.activationRoot, args.json);
        case "logs":
            return daemonLogs(args.activationRoot, args.json);
        default:
            console.error(`Unknown daemon subcommand: ${args.subcommand}`);
            console.error(daemonHelp());
            return 1;
    }
}
export function daemonHelp() {
    return [
        "Usage:",
        "  openclawbrain daemon start  --activation-root <path> [--json]",
        "  openclawbrain daemon stop   --activation-root <path> [--json]",
        "  openclawbrain daemon status --activation-root <path> [--json]",
        "  openclawbrain daemon logs   --activation-root <path> [--json]",
        "",
        "Subcommands:",
        "  start   Generate a macOS launchd plist and start the daemon (runs openclawbrain watch).",
        "  stop    Stop the daemon and remove the launchd plist.",
        "  status  Show whether the daemon is running, its PID, configured launch command, and recent log lines.",
        "  logs    Show the last 50 lines of the per-activation-root daemon log under ~/.openclawbrain/daemon/.",
        "",
        "Options:",
        "  --activation-root <path>  Explicit activation root for the wrapped watch daemon.",
        "  --json                    Emit machine-readable JSON output.",
        "  --help                    Show this help.",
    ].join("\n");
}
export function parseDaemonArgs(argv) {
    // argv should be everything after "daemon", e.g. ["start", "--activation-root", "/path"]
    const args = [...argv];
    let subcommand = "status";
    let activationRoot = null;
    let json = false;
    let help = false;
    if (args.length > 0 && (args[0] === "start" || args[0] === "stop" || args[0] === "status" || args[0] === "logs")) {
        subcommand = args.shift();
    }
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if (arg === "--help" || arg === "-h") {
            help = true;
            continue;
        }
        if (arg === "--json") {
            json = true;
            continue;
        }
        if (arg === "--activation-root") {
            const next = args[i + 1];
            if (next === undefined) {
                throw new Error("--activation-root requires a value");
            }
            activationRoot = next;
            i += 1;
            continue;
        }
        throw new Error(`Unknown daemon argument: ${arg}`);
    }
    if (help) {
        return { command: "daemon", subcommand, activationRoot: "", json, help };
    }
    if (activationRoot === null || activationRoot.trim().length === 0) {
        throw new Error(`daemon ${subcommand} requires --activation-root <path>`);
    }
    return { command: "daemon", subcommand, activationRoot: path.resolve(activationRoot), json, help };
}
//# sourceMappingURL=daemon.js.map
