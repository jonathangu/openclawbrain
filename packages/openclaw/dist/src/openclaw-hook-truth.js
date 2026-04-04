import { readFileSync } from "node:fs";
import path from "node:path";
import { describeOpenClawBrainInstallIdentity, describeOpenClawBrainInstallLayout, findInstalledOpenClawBrainPlugin, getOpenClawBrainAllowedPluginIds } from "./openclaw-plugin-install.js";
function toErrorMessage(error) {
    return error instanceof Error ? error.message : String(error);
}
function readJsonObjectRecord(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    return value;
}
function readOpenClawJsonConfig(openclawHome) {
    const openclawJsonPath = path.join(openclawHome, "openclaw.json");
    let parsed;
    try {
        parsed = JSON.parse(readFileSync(openclawJsonPath, "utf8"));
    }
    catch (error) {
        throw new Error(`Failed to read ${openclawJsonPath}: ${toErrorMessage(error)}`);
    }
    const config = readJsonObjectRecord(parsed);
    if (config === null) {
        throw new Error(`Failed to read ${openclawJsonPath}: openclaw.json must contain a top-level object`);
    }
    return {
        path: openclawJsonPath,
        config
    };
}
function shortenPath(fullPath) {
    const homeDir = process.env.HOME ?? "";
    if (homeDir.length > 0 && fullPath.startsWith(homeDir)) {
        return "~" + fullPath.slice(homeDir.length);
    }
    return fullPath;
}
function readInstalledHookPackageVersion(packageJsonPath) {
    if (typeof packageJsonPath !== "string" || packageJsonPath.trim().length === 0) {
        return null;
    }
    try {
        const parsed = JSON.parse(readFileSync(packageJsonPath, "utf8"));
        return typeof parsed?.version === "string" && parsed.version.trim().length > 0
            ? parsed.version.trim()
            : null;
    }
    catch {
        return null;
    }
}
function formatPackageIdentity(packageName, packageVersion, packageSpec = null) {
    if (typeof packageSpec === "string" && packageSpec.trim().length > 0) {
        return packageSpec.trim();
    }
    if (typeof packageName !== "string" || packageName.trim().length === 0) {
        return null;
    }
    const normalizedName = packageName.trim();
    return typeof packageVersion === "string" && packageVersion.trim().length > 0
        ? `${normalizedName}@${packageVersion.trim()}`
        : normalizedName;
}
function normalizeSurfacePath(filePath) {
    return typeof filePath === "string" && filePath.trim().length > 0
        ? path.resolve(filePath)
        : null;
}
function classifyRuntimeSurfaceConvergeState(input) {
    const reasons = [];
    if (input.scope === "activation_root_only") {
        reasons.push("selected_openclaw_home_unverified");
        return {
            state: "unverified",
            reasons
        };
    }
    if (input.daemonPath === null) {
        reasons.push("daemon_surface_unverified");
    }
    if (input.hookPath === null && input.runtimeGuardPath === null) {
        reasons.push("installed_surface_unverified");
        return {
            state: input.daemonPath === null ? "unverified" : "half_converged",
            reasons
        };
    }
    if (input.hookLoadability !== "loadable") {
        reasons.push("installed_surface_not_loadable");
        return {
            state: input.daemonPath === null ? "unverified" : "half_converged",
            reasons
        };
    }
    if (input.daemonPath === null) {
        return {
            state: "unverified",
            reasons
        };
    }
    if (input.samePath) {
        reasons.push("same_path");
        return {
            state: "converged",
            reasons
        };
    }
    if (input.daemonVersion !== null && input.hookVersion !== null) {
        reasons.push(input.daemonVersion === input.hookVersion ? "split_path_same_version" : "version_skew");
        return {
            state: input.daemonVersion === input.hookVersion ? "converged" : "half_converged",
            reasons
        };
    }
    reasons.push("version_unverified");
    return {
        state: "unverified",
        reasons
    };
}
function inspectInstalledHookActivationRoot(loaderEntryPath) {
    let content;
    try {
        content = readFileSync(loaderEntryPath, "utf8");
    }
    catch (error) {
        return {
            ready: false,
            detail: `installed loader entry ${shortenPath(loaderEntryPath)} could not be read: ${toErrorMessage(error)}`
        };
    }
    const match = content.match(/const\s+ACTIVATION_ROOT\s*=\s*["'`]([^"'`]+)["'`]/) ??
        content.match(/activationRoot:\s*["'`]([^"'`]+)["'`]/);
    if (!match || !match[1]) {
        return {
            ready: false,
            detail: `installed loader entry ${shortenPath(loaderEntryPath)} does not declare a pinned activation root`
        };
    }
    const activationRoot = match[1].trim();
    if (activationRoot === "__ACTIVATION_ROOT__" || activationRoot === "__ACTIVATION_" + "ROOT__") {
        return {
            ready: false,
            detail: `installed loader entry ${shortenPath(loaderEntryPath)} still contains the ACTIVATION_ROOT placeholder; rerun openclawbrain install/attach to pin the runtime hook`
        };
    }
    return {
        ready: true,
        detail: `installed loader entry ${shortenPath(loaderEntryPath)} pins activation root ${shortenPath(path.resolve(activationRoot))}`
    };
}
function describeAdditionalInstallDetail(additionalInstalls) {
    if (additionalInstalls.length === 0) {
        return "";
    }
    return `; additional OpenClawBrain installs also exist at ${additionalInstalls
        .map((install) => `${shortenPath(install.extensionDir)} (${describeOpenClawBrainInstallLayout(install.installLayout)}, ${describeOpenClawBrainInstallIdentity(install)})`)
        .join(", ")}`;
}
function summarizeOpenClawBrainHookGuard(inspection) {
    if (inspection.scope === "activation_root_only") {
        return {
            guardSeverity: "degraded",
            guardActionability: "pin_openclaw_home",
            guardSummary: "current-profile hook state is not self-proven from this status scope",
            guardAction: "Rerun status with --openclaw-home <path> to prove the current profile hook."
        };
    }
    if (inspection.installState === "installed" && inspection.loadability === "loadable") {
        return {
            guardSeverity: "none",
            guardActionability: "none",
            guardSummary: "profile hook is installed and loadable",
            guardAction: "none"
        };
    }
    return {
        guardSeverity: "blocking",
        guardActionability: "repair_install",
        guardSummary: inspection.installState === "not_installed"
            ? "profile hook is missing or incomplete"
            : "profile hook is present but OpenClaw will not load it",
        guardAction: "Run openclawbrain install --openclaw-home <path> to repair the installed hook."
    };
}
export function inspectOpenClawBrainPluginAllowlist(openclawHome) {
    const { path: openclawJsonPath, config } = readOpenClawJsonConfig(openclawHome);
    const installedPlugin = findInstalledOpenClawBrainPlugin(openclawHome);
    const allowedPluginIds = getOpenClawBrainAllowedPluginIds(installedPlugin.selectedInstall);
    const plugins = readJsonObjectRecord(config.plugins);
    if (plugins === null) {
        return {
            state: "unrestricted",
            detail: `${shortenPath(openclawJsonPath)} has no plugins object; the on-disk hook is not blocked by an allowlist`
        };
    }
    if (!Object.prototype.hasOwnProperty.call(plugins, "allow")) {
        return {
            state: "unrestricted",
            detail: `${shortenPath(openclawJsonPath)} does not pin plugins.allow; the on-disk hook is not blocked by an allowlist`
        };
    }
    if (!Array.isArray(plugins.allow)) {
        return {
            state: "invalid",
            detail: `${shortenPath(openclawJsonPath)} has a non-array plugins.allow value, so OpenClawBrain load cannot be proven from config`
        };
    }
    const matchedPluginId = allowedPluginIds.find((pluginId) => plugins.allow.includes(pluginId)) ?? null;
    return matchedPluginId !== null
        ? {
            state: "allowed",
            detail: `${shortenPath(openclawJsonPath)} plugins.allow explicitly includes ${matchedPluginId}; recognized OpenClawBrain ids are ${allowedPluginIds.join(", ")}`
        }
        : {
            state: "blocked",
            detail: `${shortenPath(openclawJsonPath)} plugins.allow excludes recognized OpenClawBrain ids ${allowedPluginIds.join(", ")}`
        };
}
export function inspectOpenClawBrainHookStatus(openclawHome) {
    if (openclawHome === null || openclawHome === undefined || openclawHome.trim().length === 0) {
        return {
            scope: "activation_root_only",
            openclawHome: null,
            extensionDir: null,
            hookPath: null,
            runtimeGuardPath: null,
            manifestPath: null,
            packageJsonPath: null,
            manifestId: null,
            installId: null,
            packageName: null,
            packageVersion: null,
            installLayout: null,
            additionalInstallCount: 0,
            installState: "unverified",
            loadability: "unverified",
            pluginAllowlistState: "unverified",
            desynced: false,
            detail: "profile hook state is unknown from activation-root-only status; pin --openclaw-home to prove install state"
        };
    }
    const resolvedHome = path.resolve(openclawHome);
    const installedPlugin = findInstalledOpenClawBrainPlugin(resolvedHome);
    if (installedPlugin.selectedInstall === null ||
        installedPlugin.selectedInstall.loaderEntryPath === null ||
        installedPlugin.selectedInstall.runtimeGuardPath === null) {
        const incompleteInstall = installedPlugin.selectedInstall;
        return {
            scope: "exact_openclaw_home",
            openclawHome: resolvedHome,
            extensionDir: incompleteInstall?.extensionDir ?? null,
            hookPath: incompleteInstall?.loaderEntryPath ?? null,
            runtimeGuardPath: incompleteInstall?.runtimeGuardPath ?? null,
            manifestPath: incompleteInstall?.manifestPath ?? null,
            packageJsonPath: incompleteInstall?.packageJsonPath ?? null,
            manifestId: incompleteInstall?.manifestId ?? null,
            installId: incompleteInstall?.installId ?? null,
            packageName: incompleteInstall?.packageName ?? null,
            packageVersion: incompleteInstall?.packageVersion ?? readInstalledHookPackageVersion(incompleteInstall?.packageJsonPath ?? null) ?? incompleteInstall?.manifestVersion ?? null,
            installLayout: incompleteInstall?.installLayout ?? null,
            additionalInstallCount: installedPlugin.additionalInstalls.length,
            installState: "not_installed",
            loadability: "not_installed",
            pluginAllowlistState: "unverified",
            desynced: false,
            detail: incompleteInstall === null
                ? `profile hook is not present under ${shortenPath(installedPlugin.extensionsDir)}`
                : `profile hook is incomplete under ${shortenPath(incompleteInstall.extensionDir)} (${describeOpenClawBrainInstallIdentity(incompleteInstall)})`
        };
    }
    const selectedInstall = installedPlugin.selectedInstall;
    const additionalInstallDetail = describeAdditionalInstallDetail(installedPlugin.additionalInstalls);
    const allowlist = inspectOpenClawBrainPluginAllowlist(resolvedHome);
    const layoutLabel = describeOpenClawBrainInstallLayout(selectedInstall.installLayout);
    const identityDetail = describeOpenClawBrainInstallIdentity(selectedInstall);
    const packageVersion = selectedInstall.packageVersion ?? readInstalledHookPackageVersion(selectedInstall.packageJsonPath) ?? selectedInstall.manifestVersion ?? null;
    const activationRootState = inspectInstalledHookActivationRoot(selectedInstall.loaderEntryPath);
    if (allowlist.state === "blocked") {
        return {
            scope: "exact_openclaw_home",
            openclawHome: resolvedHome,
            extensionDir: selectedInstall.extensionDir,
            hookPath: selectedInstall.loaderEntryPath,
            runtimeGuardPath: selectedInstall.runtimeGuardPath,
            manifestPath: selectedInstall.manifestPath,
            packageJsonPath: selectedInstall.packageJsonPath,
            manifestId: selectedInstall.manifestId,
            installId: selectedInstall.installId,
            packageName: selectedInstall.packageName,
            packageVersion,
            installLayout: selectedInstall.installLayout,
            additionalInstallCount: installedPlugin.additionalInstalls.length,
            installState: "blocked_by_allowlist",
            loadability: "blocked",
            pluginAllowlistState: allowlist.state,
            desynced: true,
            detail: `profile hook is installed via ${layoutLabel} at ${shortenPath(selectedInstall.extensionDir)} (${identityDetail}), but ${allowlist.detail}; ` +
                `the on-disk hook is not loadable and OpenClaw will not load it${additionalInstallDetail}`
        };
    }
    if (allowlist.state === "invalid") {
        return {
            scope: "exact_openclaw_home",
            openclawHome: resolvedHome,
            extensionDir: selectedInstall.extensionDir,
            hookPath: selectedInstall.loaderEntryPath,
            runtimeGuardPath: selectedInstall.runtimeGuardPath,
            manifestPath: selectedInstall.manifestPath,
            packageJsonPath: selectedInstall.packageJsonPath,
            manifestId: selectedInstall.manifestId,
            installId: selectedInstall.installId,
            packageName: selectedInstall.packageName,
            packageVersion,
            installLayout: selectedInstall.installLayout,
            additionalInstallCount: installedPlugin.additionalInstalls.length,
            installState: "blocked_by_allowlist",
            loadability: "blocked",
            pluginAllowlistState: allowlist.state,
            desynced: true,
            detail: `profile hook is installed via ${layoutLabel} at ${shortenPath(selectedInstall.extensionDir)} (${identityDetail}), but ${allowlist.detail}; ` +
                `treat hook-load state as broken until install/attach repairs the config${additionalInstallDetail}`
        };
    }
    if (!activationRootState.ready) {
        return {
            scope: "exact_openclaw_home",
            openclawHome: resolvedHome,
            extensionDir: selectedInstall.extensionDir,
            hookPath: selectedInstall.loaderEntryPath,
            runtimeGuardPath: selectedInstall.runtimeGuardPath,
            manifestPath: selectedInstall.manifestPath,
            packageJsonPath: selectedInstall.packageJsonPath,
            manifestId: selectedInstall.manifestId,
            installId: selectedInstall.installId,
            packageName: selectedInstall.packageName,
            packageVersion,
            installLayout: selectedInstall.installLayout,
            additionalInstallCount: installedPlugin.additionalInstalls.length,
            installState: "installed",
            loadability: "blocked",
            pluginAllowlistState: allowlist.state,
            desynced: true,
            detail: `profile hook is installed via ${layoutLabel} at ${shortenPath(selectedInstall.extensionDir)} (${identityDetail}), but ${activationRootState.detail}${additionalInstallDetail}`
        };
    }
    return {
        scope: "exact_openclaw_home",
        openclawHome: resolvedHome,
        extensionDir: selectedInstall.extensionDir,
        hookPath: selectedInstall.loaderEntryPath,
        runtimeGuardPath: selectedInstall.runtimeGuardPath,
        manifestPath: selectedInstall.manifestPath,
        packageJsonPath: selectedInstall.packageJsonPath,
        manifestId: selectedInstall.manifestId,
        installId: selectedInstall.installId,
        packageName: selectedInstall.packageName,
        packageVersion,
        installLayout: selectedInstall.installLayout,
        additionalInstallCount: installedPlugin.additionalInstalls.length,
        installState: "installed",
        loadability: "loadable",
        pluginAllowlistState: allowlist.state,
        desynced: false,
        detail: `profile hook is installed via ${layoutLabel} at ${shortenPath(selectedInstall.extensionDir)} (${identityDetail})${additionalInstallDetail}`
    };
}
export function summarizeOpenClawBrainHookLoad(inspection, statusProbeReady) {
    return {
        ...inspection,
        ...summarizeOpenClawBrainHookGuard(inspection),
        loadProof: inspection.loadability === "loadable" && statusProbeReady
            ? "status_probe_ready"
            : "not_ready"
    };
}
export function describeOpenClawBrainHotfixBoundary(input) {
    const hookInspection = input.hookInspection;
    const daemonInspection = input.daemonInspection ?? null;
    const daemonPath = normalizeSurfacePath(daemonInspection?.configuredRuntimePath ?? null);
    const hookPath = normalizeSurfacePath(hookInspection.hookPath);
    const runtimeGuardPath = normalizeSurfacePath(hookInspection.runtimeGuardPath);
    const selectedOpenClawHome = typeof hookInspection.openclawHome === "string" && hookInspection.openclawHome.trim().length > 0
        ? path.resolve(hookInspection.openclawHome)
        : null;
    const daemonSource = daemonInspection?.surfaceIdentitySource ?? (daemonPath === null ? "unverified" : "managed_service");
    const daemonPackageName = daemonInspection?.configuredRuntimePackageName ?? null;
    const daemonPackageVersion = daemonInspection?.configuredRuntimePackageVersion ?? null;
    const hookPackageName = hookInspection.packageName ?? null;
    const hookPackageVersion = hookInspection.packageVersion ?? null;
    const daemonPackage = formatPackageIdentity(daemonInspection?.configuredRuntimePackageName ?? null, daemonInspection?.configuredRuntimePackageVersion ?? null, daemonInspection?.configuredRuntimePackageSpec ?? null);
    const hookPackage = formatPackageIdentity(hookInspection.packageName, hookInspection.packageVersion);
    const samePath = daemonPath === hookPath || daemonPath === runtimeGuardPath;
    const daemonVersion = daemonPackageVersion;
    const hookVersion = hookPackageVersion;
    let boundary;
    let skew;
    let displayedHookPath = hookPath;
    let displayedRuntimeGuardPath = runtimeGuardPath;
    let displayedDaemonPackage = daemonPackage;
    let displayedHookPackage = hookPackage;
    let guidance;
    let detail;
    if (hookInspection.scope === "activation_root_only") {
        boundary = "hook_surface_unverified";
        skew = "unverified";
        displayedHookPath = null;
        displayedRuntimeGuardPath = null;
        displayedHookPackage = null;
        guidance = "Pin --openclaw-home before patching the installed hook/runtime-guard surface; activation-root-only status does not prove which OpenClaw install you would be changing.";
        detail = daemonPath === null
            ? "activation-root-only status does not expose the installed hook/runtime-guard surface."
            : `daemon runtime path ${shortenPath(daemonPath)} is visible, but activation-root-only status still does not expose the installed hook/runtime-guard surface.`;
    }
    else if (hookPath === null && runtimeGuardPath === null) {
        boundary = "hook_surface_unverified";
        skew = "unverified";
        displayedHookPath = null;
        displayedRuntimeGuardPath = null;
        guidance = daemonPath === null
            ? "Repair or reinstall the installed hook surface before patching OpenClaw load behavior."
            : "Patch the daemon runtime path for background watch fixes, but repair or reinstall the installed hook/runtime-guard surface before patching OpenClaw load behavior.";
        detail = daemonPath === null
            ? "no verified daemon runtime path or installed hook/runtime-guard path is visible from this status snapshot."
            : `daemon runtime path ${shortenPath(daemonPath)} is visible, but the installed hook/runtime-guard surface is not yet loadable.`;
    }
    else if (daemonPath === null) {
        boundary = "daemon_surface_only";
        skew = "unverified";
        displayedDaemonPackage = null;
        guidance = "Patch the installed hook/runtime-guard surface for OpenClaw load fixes. No configured daemon runtime path is visible from status.";
        detail = `installed hook loads from ${hookPath === null ? "unverified" : shortenPath(hookPath)}${runtimeGuardPath === null ? "" : ` with runtime-guard ${shortenPath(runtimeGuardPath)}`}, but no configured daemon runtime path is visible.`;
    }
    else {
        skew = samePath
            ? "same_path"
            : daemonVersion !== null && hookVersion !== null
                ? daemonVersion === hookVersion
                    ? "split_path_same_version"
                    : "split_path_version_skew"
                : "split_path_version_unverified";
        boundary = samePath ? "same_surface" : "split_surfaces";
        guidance = samePath
            ? "Daemon and installed hook paths currently collapse to the same file; verify both surfaces before patching anyway."
            : "Patch the daemon runtime path for background watch/learner fixes. Patch the installed hook/runtime-guard paths for OpenClaw load fixes.";
        detail = samePath
            ? `daemon runtime path ${shortenPath(daemonPath)} currently resolves to the same file as the installed OpenClaw hook surface.`
            : `daemon background watch runs from ${shortenPath(daemonPath)}${daemonPackage === null ? "" : ` (${daemonPackage})`}; OpenClaw loads the installed hook from ${hookPath === null ? "unverified" : shortenPath(hookPath)}${hookPackage === null ? "" : ` (${hookPackage})`}${runtimeGuardPath === null ? "" : ` and runtime-guard ${shortenPath(runtimeGuardPath)}`}.`;
    }
    const converge = classifyRuntimeSurfaceConvergeState({
        scope: hookInspection.scope,
        daemonPath,
        hookPath: displayedHookPath,
        runtimeGuardPath: displayedRuntimeGuardPath,
        hookLoadability: hookInspection.loadability,
        daemonVersion,
        hookVersion,
        samePath
    });
    return {
        boundary,
        skew,
        daemonPath,
        hookPath: displayedHookPath,
        runtimeGuardPath: displayedRuntimeGuardPath,
        daemonPackage: displayedDaemonPackage,
        hookPackage: displayedHookPackage,
        daemonPackageName,
        daemonPackageVersion,
        hookPackageName,
        hookPackageVersion,
        daemonSource,
        selectedOpenClawHome,
        convergeState: converge.state,
        convergeReasons: converge.reasons,
        guidance,
        detail
    };
}
//# sourceMappingURL=openclaw-hook-truth.js.map
