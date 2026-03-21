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
        loadProof: inspection.loadability === "loadable" && statusProbeReady
            ? "status_probe_ready"
            : "not_ready"
    };
}
//# sourceMappingURL=openclaw-hook-truth.js.map
