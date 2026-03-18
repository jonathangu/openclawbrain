import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
export const OPENCLAWBRAIN_PLUGIN_ID = "openclawbrain";
export const OPENCLAWBRAIN_SHADOW_PACKAGE_NAME = "openclawbrain";
export const OPENCLAWBRAIN_NATIVE_PACKAGE_NAME = "@openclawbrain/openclaw";
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
    if (input.loaderEntryPath !== null &&
        input.loaderEntryPath.endsWith(path.join("dist", "extension", "index.js"))) {
        return "native_package_plugin";
    }
    if (input.loaderEntryPath !== null &&
        path.basename(input.loaderEntryPath) === "index.ts" &&
        path.basename(input.extensionDir) === OPENCLAWBRAIN_PLUGIN_ID) {
        return "generated_shadow_extension";
    }
    return null;
}
function inspectOpenClawBrainPluginInstall(extensionDir, pluginId) {
    const manifestPath = path.join(extensionDir, "openclaw.plugin.json");
    if (!existsSync(manifestPath)) {
        return null;
    }
    const manifest = readJsonObject(manifestPath);
    if (manifest?.id !== pluginId) {
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
    const installLayout = inferInstallLayout({
        extensionDir,
        packageName,
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
        packageName,
        installLayout
    };
}
export function describeOpenClawBrainInstallLayout(installLayout) {
    return installLayout === "native_package_plugin"
        ? "native package plugin"
        : "generated shadow extension";
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
    }));
    return {
        openclawHome: resolvedOpenclawHome,
        extensionsDir,
        selectedInstall: installs[0] ?? null,
        additionalInstalls: installs.slice(1)
    };
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
