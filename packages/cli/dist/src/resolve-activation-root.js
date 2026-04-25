/**
 * Auto-detect the OpenClawBrain activation root.
 *
 * Resolution order:
 *   1. Explicit `--activation-root <path>` (passed as `explicit` arg)
 *   2. A selected OpenClaw home (`openclawHome` option or OPENCLAW_HOME env)
 *   3. Unpinned host auto-detect from ~/.openclawbrain/activation and installed hooks
 *   4. Refuse clearly if host-local signals disagree or are unresolved
 *   5. Fail with a clear error message
 *
 * Exported for use by CLI commands and other agents' code.
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describeOpenClawHomeInspection, discoverOpenClawHomes, formatOpenClawHomeLayout, inspectOpenClawHome } from "./openclaw-home-layout.js";
import { findInstalledOpenClawBrainPlugin } from "./openclaw-plugin-install.js";
function getHomeDir() {
    return process.env.HOME ?? process.env.USERPROFILE ?? "~";
}
function normalizeOptionalString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}
function buildInstallGuidance() {
    return "Run: openclawbrain install --openclaw-home <path>";
}
function buildProfilePinningGuidance() {
    return "Pass --activation-root <path> to pin the brain directly, or --openclaw-home <path> (or OPENCLAW_HOME) to select one installed OpenClaw home.";
}
function extractActivationRootFromExtension(filePath) {
    try {
        const content = readFileSync(filePath, "utf8");
        const match = content.match(/const\s+ACTIVATION_ROOT\s*=\s*["'`]([^"'`]+)["'`]/) ??
            content.match(/activationRoot:\s*["'`]([^"'`]+)["'`]/);
        if (!match || !match[1]) {
            return {
                activationRoot: null,
                diagnostic: `installed extension ${filePath} does not declare a hardcoded activation root`
            };
        }
        const candidate = match[1].trim();
        if (candidate === "__ACTIVATION_ROOT__" ||
            candidate === "__ACTIVATION_" + "ROOT__") {
            return {
                activationRoot: null,
                diagnostic: `installed extension ${filePath} still contains the ACTIVATION_ROOT placeholder`
            };
        }
        const resolvedCandidate = path.resolve(candidate);
        if (!existsSync(resolvedCandidate)) {
            return {
                activationRoot: null,
                diagnostic: `installed extension ${filePath} points at missing activation root ${resolvedCandidate}`
            };
        }
        return {
            activationRoot: resolvedCandidate,
            diagnostic: null
        };
    }
    catch {
        return {
            activationRoot: null,
            diagnostic: `installed extension ${filePath} exists but could not be read`
        };
    }
}
function readActivationRootFromOpenClawHome(openclawHome) {
    const installedPlugin = findInstalledOpenClawBrainPlugin(openclawHome);
    if (installedPlugin.selectedInstall === null ||
        installedPlugin.selectedInstall.loaderEntryPath === null ||
        installedPlugin.selectedInstall.runtimeGuardPath === null) {
        return null;
    }
    return extractActivationRootFromExtension(installedPlugin.selectedInstall.loaderEntryPath);
}
/**
 * Scan discoverable ~/.openclaw* directories for installed hooks and record
 * either the pinned activation root or the reason the hook is unresolved.
 */
function scanInstalledProfileActivationRoots() {
    const probes = [];
    for (const inspection of discoverOpenClawHomes(path.resolve(getHomeDir()))) {
        const openclawHome = inspection.openclawHome;
        const installedPlugin = findInstalledOpenClawBrainPlugin(openclawHome);
        if (installedPlugin.selectedInstall === null ||
            installedPlugin.selectedInstall.loaderEntryPath === null ||
            installedPlugin.selectedInstall.runtimeGuardPath === null) {
            continue;
        }
        const found = extractActivationRootFromExtension(installedPlugin.selectedInstall.loaderEntryPath);
        probes.push({
            openclawHome,
            extensionEntryPath: installedPlugin.selectedInstall.loaderEntryPath,
            activationRoot: found.activationRoot,
            diagnostic: found.diagnostic,
            inspection
        });
    }
    return probes.sort((left, right) => left.openclawHome.localeCompare(right.openclawHome));
}
function buildUnpinnedResolutionRefusal(input) {
    const details = [];
    const { defaultActivationRoot, installedProfileProbes } = input;
    if (defaultActivationRoot !== null) {
        details.push(`  - default HOME root -> ${defaultActivationRoot}`);
    }
    for (const probe of installedProfileProbes) {
        const inspectionSummary = describeOpenClawHomeInspection(probe.inspection);
        if (probe.activationRoot !== null) {
            details.push(`  - ${probe.openclawHome} [${inspectionSummary}] -> ${probe.activationRoot}`);
            continue;
        }
        details.push(`  - ${probe.openclawHome} [${inspectionSummary}] -> unresolved: ${probe.diagnostic ?? `installed extension ${probe.extensionEntryPath} is unresolved`}`);
    }
    return [
        "Refusing to auto-select an activation root from unpinned host state.",
        "Detected targets:",
        ...details,
        buildProfilePinningGuidance(),
        buildInstallGuidance()
    ].join("\n");
}
/**
 * Resolve the activation root path through the detection chain.
 *
 * @returns Absolute path to the activation root.
 * @throws If no activation root can be found (unless `quiet` is true).
 */
export function resolveActivationRoot(options = {}) {
    const { explicit, openclawHome, quiet } = options;
    // 1. Explicit flag
    if (typeof explicit === "string" && explicit.trim().length > 0) {
        return path.resolve(explicit);
    }
    const selectedOpenClawHome = normalizeOptionalString(openclawHome) ?? normalizeOptionalString(process.env.OPENCLAW_HOME);
    // 2. A selected OpenClaw home is authoritative on many-profile hosts.
    if (selectedOpenClawHome !== null) {
        const selectedInspection = inspectOpenClawHome(selectedOpenClawHome);
        const selectedActivationRoot = readActivationRootFromOpenClawHome(selectedOpenClawHome);
        const selectedActivationRootPath = selectedActivationRoot?.activationRoot ?? null;
        if (selectedActivationRootPath !== null) {
            return path.resolve(selectedActivationRootPath);
        }
        if (quiet) {
            return "";
        }
        const selectedActivationRootDiagnostic = selectedActivationRoot?.diagnostic ?? null;
        if (selectedActivationRootDiagnostic !== null) {
            throw new Error(`OpenClawBrain extension found for OpenClaw home ${path.resolve(selectedOpenClawHome)} (${formatOpenClawHomeLayout(selectedInspection.layout)}), but activation root is unresolved: ${selectedActivationRootDiagnostic}. ${buildInstallGuidance()}`);
        }
        throw new Error(`No brain found for OpenClaw home ${path.resolve(selectedOpenClawHome)} (${formatOpenClawHomeLayout(selectedInspection.layout)}). ${buildInstallGuidance()}`);
    }
    // 3. Default location: ~/.openclawbrain/activation
    const defaultPath = path.join(getHomeDir(), ".openclawbrain", "activation");
    const resolvedDefaultPath = existsSync(defaultPath) ? path.resolve(defaultPath) : null;
    // 4. Scan installed hooks under HOME
    const installedProfileProbes = scanInstalledProfileActivationRoots();
    const resolvedCandidateRoots = new Set();
    const hasUnresolvedInstalledProfiles = installedProfileProbes.some((probe) => probe.activationRoot === null);
    if (resolvedDefaultPath !== null) {
        resolvedCandidateRoots.add(resolvedDefaultPath);
    }
    for (const probe of installedProfileProbes) {
        if (probe.activationRoot !== null) {
            resolvedCandidateRoots.add(path.resolve(probe.activationRoot));
        }
    }
    if (resolvedCandidateRoots.size === 1 && !hasUnresolvedInstalledProfiles) {
        return [...resolvedCandidateRoots][0];
    }
    if ((resolvedCandidateRoots.size > 1 || hasUnresolvedInstalledProfiles) && installedProfileProbes.length > 0) {
        if (quiet) {
            return "";
        }
        throw new Error(buildUnpinnedResolutionRefusal({
            defaultActivationRoot: resolvedDefaultPath,
            installedProfileProbes
        }));
    }
    // 5. Nothing found
    if (quiet) {
        return "";
    }
    throw new Error(`No brain found. ${buildInstallGuidance()}`);
}
