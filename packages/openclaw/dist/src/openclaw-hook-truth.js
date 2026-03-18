import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
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
export function inspectOpenClawBrainPluginAllowlist(openclawHome) {
    const { path: openclawJsonPath, config } = readOpenClawJsonConfig(openclawHome);
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
    return plugins.allow.includes("openclawbrain")
        ? {
            state: "allowed",
            detail: `${shortenPath(openclawJsonPath)} plugins.allow explicitly includes openclawbrain`
        }
        : {
            state: "blocked",
            detail: `${shortenPath(openclawJsonPath)} plugins.allow excludes openclawbrain`
        };
}
export function inspectOpenClawBrainHookStatus(openclawHome) {
    if (openclawHome === null || openclawHome === undefined || openclawHome.trim().length === 0) {
        return {
            scope: "activation_root_only",
            openclawHome: null,
            hookPath: null,
            runtimeGuardPath: null,
            manifestPath: null,
            installState: "unverified",
            loadability: "unverified",
            pluginAllowlistState: "unverified",
            desynced: false,
            detail: "profile hook state is unknown from activation-root-only status; pin --openclaw-home to prove install state"
        };
    }
    const resolvedHome = path.resolve(openclawHome);
    const extensionDir = path.join(resolvedHome, "extensions", "openclawbrain");
    const hookPath = path.join(extensionDir, "index.ts");
    const runtimeGuardPath = path.join(extensionDir, "runtime-guard.js");
    const manifestPath = path.join(extensionDir, "openclaw.plugin.json");
    if (!(existsSync(hookPath) && existsSync(runtimeGuardPath) && existsSync(manifestPath))) {
        return {
            scope: "exact_openclaw_home",
            openclawHome: resolvedHome,
            hookPath,
            runtimeGuardPath,
            manifestPath,
            installState: "not_installed",
            loadability: "not_installed",
            pluginAllowlistState: "unverified",
            desynced: false,
            detail: `profile hook is not present at ${shortenPath(extensionDir)}`
        };
    }
    const allowlist = inspectOpenClawBrainPluginAllowlist(resolvedHome);
    if (allowlist.state === "blocked") {
        return {
            scope: "exact_openclaw_home",
            openclawHome: resolvedHome,
            hookPath,
            runtimeGuardPath,
            manifestPath,
            installState: "blocked_by_allowlist",
            loadability: "blocked",
            pluginAllowlistState: allowlist.state,
            desynced: true,
            detail: `profile hook files exist at ${shortenPath(extensionDir)}, but ${allowlist.detail}; ` +
                "the on-disk hook is desynced and OpenClaw will not load it"
        };
    }
    if (allowlist.state === "invalid") {
        return {
            scope: "exact_openclaw_home",
            openclawHome: resolvedHome,
            hookPath,
            runtimeGuardPath,
            manifestPath,
            installState: "blocked_by_allowlist",
            loadability: "blocked",
            pluginAllowlistState: allowlist.state,
            desynced: true,
            detail: `profile hook files exist at ${shortenPath(extensionDir)}, but ${allowlist.detail}; ` +
                "treat hook-load state as broken until install/attach repairs the config"
        };
    }
    return {
        scope: "exact_openclaw_home",
        openclawHome: resolvedHome,
        hookPath,
        runtimeGuardPath,
        manifestPath,
        installState: "installed",
        loadability: "loadable",
        pluginAllowlistState: allowlist.state,
        desynced: false,
        detail: `profile hook is installed at ${shortenPath(extensionDir)}`
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