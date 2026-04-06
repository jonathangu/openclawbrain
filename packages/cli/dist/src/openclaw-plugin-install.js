import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
export const OPENCLAWBRAIN_PLUGIN_ID = "openclawbrain";
export const OPENCLAWBRAIN_SHADOW_PACKAGE_NAME = "openclawbrain";
export const OPENCLAWBRAIN_NATIVE_PACKAGE_NAME = "@openclawbrain/openclaw";
export const OPENCLAWBRAIN_NATIVE_INSTALL_ID = "openclaw";
function readJsonObject(filePath) {
    try {
        const parsed = JSON.parse(readFileSync(filePath, "utf8"));
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
            return null;
        }
        return parsed;
    }
    catch {
        return null;
    }
}
function resolveConfiguredEntries(packageJson) {
    const entries = packageJson?.openclaw?.extensions;
    if (!Array.isArray(entries)) {
        return [];
    }
    return entries
        .filter((entry) => typeof entry === "string")
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);
}
function resolveLoaderEntryPath(extensionDir, configuredEntries) {
    for (const configuredEntry of configuredEntries) {
        const candidate = path.join(extensionDir, configuredEntry);
        if (existsSync(candidate)) {
            return candidate;
        }
    }
    return null;
}
function resolveRuntimeGuardPath(loaderEntryPath) {
    const runtimeGuardPath = path.join(path.dirname(loaderEntryPath), "runtime-guard.js");
    return existsSync(runtimeGuardPath) ? runtimeGuardPath : null;
}
function normalizeOptionalString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}
function readJsonObjectRecord(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    return value;
}
function pushUniqueNormalizedId(ids, value) {
    const normalized = normalizeOptionalString(value);
    if (normalized === null || ids.includes(normalized)) {
        return;
    }
    ids.push(normalized);
}
function normalizeInstallId(value) {
    const normalized = normalizeOptionalString(value);
    if (normalized === null) {
        return null;
    }
    if (!normalized.startsWith("@")) {
        return normalized;
    }
    const parts = normalized.split("/");
    return normalizeOptionalString(parts[1] ?? null);
}
function listCandidateExtensionDirs(extensionsDir) {
    const candidates = [];
    let entries;
    try {
        entries = readdirSync(extensionsDir, { withFileTypes: true });
    }
    catch {
        return [];
    }
    for (const entry of entries) {
        if (!entry.isDirectory()) {
            continue;
        }
        const directPath = path.join(extensionsDir, entry.name);
        candidates.push(directPath);
        if (!entry.name.startsWith("@")) {
            continue;
        }
        let scopedEntries;
        try {
            scopedEntries = readdirSync(directPath, { withFileTypes: true });
        }
        catch {
            continue;
        }
        for (const scopedEntry of scopedEntries) {
            if (!scopedEntry.isDirectory()) {
                continue;
            }
            candidates.push(path.join(directPath, scopedEntry.name));
        }
    }
    return candidates.sort((left, right) => left.localeCompare(right));
}
function inferInstallLayout(input) {
    if (input.packageName === OPENCLAWBRAIN_NATIVE_PACKAGE_NAME) {
        return "native_package_plugin";
    }
    if (input.packageName === OPENCLAWBRAIN_SHADOW_PACKAGE_NAME) {
        return "generated_shadow_extension";
    }
    if (input.installId === OPENCLAWBRAIN_NATIVE_INSTALL_ID &&
        input.loaderEntryPath !== null &&
        input.loaderEntryPath.endsWith(path.join("dist", "extension", "index.js"))) {
        return "native_package_plugin";
    }
    if (input.loaderEntryPath !== null &&
        input.loaderEntryPath.endsWith(path.join("dist", "extension", "index.js"))) {
        return "native_package_plugin";
    }
    if (input.loaderEntryPath !== null &&
        path.basename(input.loaderEntryPath) === "index.ts" &&
        (input.installId === OPENCLAWBRAIN_PLUGIN_ID ||
            path.basename(input.extensionDir) === OPENCLAWBRAIN_PLUGIN_ID)) {
        return "generated_shadow_extension";
    }
    return null;
}
function compareInstalledPluginPriority(left, right) {
    const leftCompleteness = Number(left.loaderEntryPath !== null) + Number(left.runtimeGuardPath !== null);
    const rightCompleteness = Number(right.loaderEntryPath !== null) + Number(right.runtimeGuardPath !== null);
    if (leftCompleteness !== rightCompleteness) {
        return rightCompleteness - leftCompleteness;
    }
    const leftLayoutRank = left.installLayout === "native_package_plugin" ? 0 : 1;
    const rightLayoutRank = right.installLayout === "native_package_plugin" ? 0 : 1;
    if (leftLayoutRank !== rightLayoutRank) {
        return leftLayoutRank - rightLayoutRank;
    }
    return left.extensionDir.localeCompare(right.extensionDir);
}
function inspectOpenClawBrainPluginInstall(extensionDir, pluginId) {
    const manifestPath = path.join(extensionDir, "openclaw.plugin.json");
    if (!existsSync(manifestPath)) {
        return null;
    }
    const manifest = readJsonObject(manifestPath);
    const manifestId = normalizeOptionalString(manifest?.id);
    if (manifestId !== pluginId) {
        return null;
    }
    const packageJsonPath = path.join(extensionDir, "package.json");
    if (!existsSync(packageJsonPath)) {
        return null;
    }
    const packageJson = readJsonObject(packageJsonPath);
    if (packageJson === null) {
        return null;
    }
    const configuredEntries = resolveConfiguredEntries(packageJson);
    const loaderEntryPath = resolveLoaderEntryPath(extensionDir, configuredEntries);
    const packageName = typeof packageJson.name === "string" && packageJson.name.trim().length > 0
        ? packageJson.name.trim()
        : null;
    const packageVersion = typeof packageJson.version === "string" && packageJson.version.trim().length > 0
        ? packageJson.version.trim()
        : null;
    const manifestVersion = normalizeOptionalString(manifest?.version);
    const installId = normalizeInstallId(packageName) ?? normalizeInstallId(path.basename(extensionDir));
    const installLayout = inferInstallLayout({
        extensionDir,
        packageName,
        installId,
        loaderEntryPath
    });
    if (installLayout === null) {
        return null;
    }
    return {
        extensionDir,
        manifestPath,
        packageJsonPath,
        loaderEntryPath,
        runtimeGuardPath: loaderEntryPath === null ? null : resolveRuntimeGuardPath(loaderEntryPath),
        configuredEntries,
        manifestId,
        manifestVersion,
        installId,
        packageName,
        packageVersion,
        installLayout
    };
}
export function describeOpenClawBrainInstallLayout(installLayout) {
    return installLayout === "native_package_plugin"
        ? "native package plugin"
        : "generated shadow extension";
}
export function getOpenClawBrainAllowedPluginIds(install) {
    const ids = [];
    pushUniqueNormalizedId(ids, OPENCLAWBRAIN_PLUGIN_ID);
    pushUniqueNormalizedId(ids, install?.manifestId ?? null);
    return ids;
}
function getOpenClawBrainLegacyInstallHintIds(install) {
    const ids = [];
    pushUniqueNormalizedId(ids, install?.installId ?? null);
    if (install?.packageName === OPENCLAWBRAIN_NATIVE_PACKAGE_NAME) {
        pushUniqueNormalizedId(ids, OPENCLAWBRAIN_NATIVE_INSTALL_ID);
    }
    return ids.filter((id) => !getOpenClawBrainAllowedPluginIds(install).includes(id));
}
export function getOpenClawBrainKnownPluginIds(install) {
    return [
        ...getOpenClawBrainAllowedPluginIds(install),
        ...getOpenClawBrainLegacyInstallHintIds(install)
    ];
}
function mergePluginEntryValues(preferredValue, fallbackValue) {
    if (preferredValue === undefined) {
        return fallbackValue;
    }
    if (fallbackValue === undefined) {
        return preferredValue;
    }
    const preferredRecord = readJsonObjectRecord(preferredValue);
    const fallbackRecord = readJsonObjectRecord(fallbackValue);
    if (preferredRecord === null || fallbackRecord === null) {
        return preferredValue;
    }
    const mergedValue = {
        ...fallbackRecord,
        ...preferredRecord
    };
    const preferredConfig = readJsonObjectRecord(preferredRecord.config);
    const fallbackConfig = readJsonObjectRecord(fallbackRecord.config);
    if (preferredConfig !== null || fallbackConfig !== null) {
        mergedValue.config = {
            ...(fallbackConfig ?? {}),
            ...(preferredConfig ?? {})
        };
    }
    return mergedValue;
}
export function normalizeOpenClawBrainPluginsConfig(pluginsConfig, install) {
    const nextPluginsConfig = {
        ...pluginsConfig
    };
    const changes = [];
    let changed = false;
    const allowedPluginIds = getOpenClawBrainAllowedPluginIds(install);
    const legacyInstallHintIds = getOpenClawBrainLegacyInstallHintIds(install);
    if (Array.isArray(nextPluginsConfig.allow)) {
        const removedAllowEntries = nextPluginsConfig.allow.filter((entry) => legacyInstallHintIds.includes(entry));
        const filteredAllow = nextPluginsConfig.allow.filter((entry) => !legacyInstallHintIds.includes(entry));
        const addedAllowEntries = allowedPluginIds.filter((pluginId) => !filteredAllow.includes(pluginId));
        if (removedAllowEntries.length > 0 || addedAllowEntries.length > 0) {
            nextPluginsConfig.allow = [...filteredAllow, ...addedAllowEntries];
            changed = true;
            if (removedAllowEntries.length > 0) {
                changes.push(`removed plugins.allow entries ${removedAllowEntries.join(", ")}`);
            }
            if (addedAllowEntries.length > 0) {
                changes.push(`added plugins.allow entries ${addedAllowEntries.join(", ")}`);
            }
        }
    }
    const canonicalEntryId = allowedPluginIds[0] ?? OPENCLAWBRAIN_PLUGIN_ID;
    const entries = readJsonObjectRecord(nextPluginsConfig.entries);
    if (entries !== null) {
        const legacyEntryIds = legacyInstallHintIds.filter((entryId) => Object.prototype.hasOwnProperty.call(entries, entryId));
        if (legacyEntryIds.length > 0) {
            const nextEntries = {
                ...entries
            };
            let mergedEntryValue = Object.prototype.hasOwnProperty.call(nextEntries, canonicalEntryId)
                ? nextEntries[canonicalEntryId]
                : undefined;
            for (const entryId of legacyEntryIds) {
                mergedEntryValue = mergePluginEntryValues(mergedEntryValue, nextEntries[entryId]);
            }
            nextEntries[canonicalEntryId] = mergedEntryValue;
            for (const entryId of legacyEntryIds) {
                if (entryId === canonicalEntryId) {
                    continue;
                }
                delete nextEntries[entryId];
            }
            nextPluginsConfig.entries = nextEntries;
            changed = true;
            changes.push(`${Object.prototype.hasOwnProperty.call(entries, canonicalEntryId) ? "merged" : "moved"} plugins.entries.${legacyEntryIds.join(", plugins.entries.")} to plugins.entries.${canonicalEntryId}`);
        }
    }
    return {
        changed,
        pluginsConfig: nextPluginsConfig,
        changes,
        allowedPluginIds,
        canonicalEntryId
    };
}
export function describeOpenClawBrainInstallIdentity(install) {
    const displayInstallId = install.packageName === OPENCLAWBRAIN_NATIVE_PACKAGE_NAME
        ? OPENCLAWBRAIN_PLUGIN_ID
        : install.installId ?? path.basename(install.extensionDir);
    return [
        `manifest=${install.manifestId ?? OPENCLAWBRAIN_PLUGIN_ID}`,
        `install=${displayInstallId}`,
        `package=${install.packageName ?? "unknown"}`
    ].join(" ");
}
export function findInstalledOpenClawBrainPlugin(openclawHome, pluginId = OPENCLAWBRAIN_PLUGIN_ID) {
    const resolvedOpenclawHome = path.resolve(openclawHome);
    const extensionsDir = path.join(resolvedOpenclawHome, "extensions");
    if (!existsSync(extensionsDir)) {
        return {
            openclawHome: resolvedOpenclawHome,
            extensionsDir,
            selectedInstall: null,
            additionalInstalls: []
        };
    }
    const installs = listCandidateExtensionDirs(extensionsDir)
        .map((extensionDir) => inspectOpenClawBrainPluginInstall(extensionDir, pluginId))
        .filter((install) => install !== null)
        .map((install) => ({
        ...install,
        openclawHome: resolvedOpenclawHome,
        extensionsDir
    }))
        .sort(compareInstalledPluginPriority);
    return {
        openclawHome: resolvedOpenclawHome,
        extensionsDir,
        selectedInstall: installs[0] ?? null,
        additionalInstalls: installs.slice(1)
    };
}
export function resolveOpenClawBrainInstallTarget(openclawHome) {
    const installedPlugin = findInstalledOpenClawBrainPlugin(openclawHome);
    const selectedInstall = installedPlugin.selectedInstall;
    if (selectedInstall !== null &&
        selectedInstall.installLayout === "native_package_plugin" &&
        selectedInstall.loaderEntryPath !== null &&
        selectedInstall.runtimeGuardPath !== null) {
        return {
            installLayout: selectedInstall.installLayout,
            extensionDir: selectedInstall.extensionDir,
            hookPath: selectedInstall.loaderEntryPath,
            writeMode: "pin_native_package",
            selectedInstall,
            additionalInstalls: installedPlugin.additionalInstalls
        };
    }
    const extensionDir = path.join(openclawHome, "extensions", OPENCLAWBRAIN_PLUGIN_ID);
    return {
        installLayout: "generated_shadow_extension",
        extensionDir,
        hookPath: path.join(extensionDir, "index.ts"),
        writeMode: "write_shadow_extension",
        selectedInstall,
        additionalInstalls: installedPlugin.additionalInstalls
    };
}
export function pinInstalledOpenClawBrainPluginActivationRoot(loaderEntryPath, activationRoot) {
    const loaderSource = readFileSync(loaderEntryPath, "utf8");
    const activationRootPattern = /^const\s+ACTIVATION_ROOT\s*=\s*["'`][^"'`]*["'`];$/m;
    const pinnedActivationRoot = `const ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};`;
    const matchedActivationRoot = loaderSource.match(activationRootPattern)?.[0] ?? null;
    if (matchedActivationRoot === null) {
        throw new Error(`Installed loader entry ${loaderEntryPath} does not expose a patchable ACTIVATION_ROOT constant`);
    }
    if (matchedActivationRoot === pinnedActivationRoot) {
        return;
    }
    const nextLoaderSource = loaderSource.replace(activationRootPattern, pinnedActivationRoot);
    writeFileSync(loaderEntryPath, nextLoaderSource, "utf8");
}
export function resolveOpenClawHomeFromExtensionEntryPath(extensionEntryPath) {
    let currentPath = path.resolve(extensionEntryPath);
    while (true) {
        const parentPath = path.dirname(currentPath);
        if (parentPath === currentPath) {
            break;
        }
        if (path.basename(parentPath) === "extensions") {
            return path.dirname(parentPath);
        }
        currentPath = parentPath;
    }
    return null;
}
//# sourceMappingURL=openclaw-plugin-install.js.map
