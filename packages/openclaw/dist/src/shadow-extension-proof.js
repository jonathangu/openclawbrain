import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
const REQUIRED_RUNTIME_GUARD_EXPORTS = [
    "createBeforePromptBuildHandler",
    "isActivationRootPlaceholder",
    "normalizePromptBuildEvent",
    "validateExtensionRegistrationApi"
];
export function inspectInstalledOpenClawBrainExtension(openclawHome, extensionId = "openclawbrain") {
    const resolvedOpenclawHome = path.resolve(openclawHome);
    const extensionsDir = path.join(resolvedOpenclawHome, "extensions");
    const extensionDir = path.join(extensionsDir, extensionId);
    if (!existsSync(extensionsDir)) {
        throw new Error(`[shadow-extension-load-proof] OpenClaw extensions dir is missing: ${extensionsDir}`);
    }
    if (!existsSync(extensionDir)) {
        throw new Error(`[shadow-extension-load-proof] Installed extension dir is missing: ${extensionDir}`);
    }
    const manifestPath = path.join(extensionDir, "openclaw.plugin.json");
    const packageJsonPath = path.join(extensionDir, "package.json");
    const runtimeGuardPath = path.join(extensionDir, "runtime-guard.js");
    if (!existsSync(manifestPath)) {
        throw new Error(`[shadow-extension-load-proof] Plugin manifest missing: ${manifestPath}`);
    }
    if (!existsSync(packageJsonPath)) {
        throw new Error(`[shadow-extension-load-proof] Package manifest missing: ${packageJsonPath}`);
    }
    if (!existsSync(runtimeGuardPath)) {
        throw new Error(`[shadow-extension-load-proof] Installed runtime-guard.js is missing: ${runtimeGuardPath}`);
    }
    const manifest = readRequiredJson(manifestPath, "plugin manifest");
    if (manifest.id !== extensionId) {
        throw new Error(`[shadow-extension-load-proof] Plugin manifest id must be ${JSON.stringify(extensionId)} at ${manifestPath} ` +
            `(received=${describeValue(manifest.id)})`);
    }
    if (typeof manifest.version !== "string" || manifest.version.trim().length === 0) {
        throw new Error(`[shadow-extension-load-proof] Plugin manifest version must be a non-empty string at ${manifestPath}`);
    }
    const packageJson = readRequiredJson(packageJsonPath, "package manifest");
    const configuredEntries = readConfiguredEntries(packageJson, packageJsonPath);
    const loaderEntryPath = resolveLoaderEntryPath(extensionDir, configuredEntries);
    if (packageJson.type !== "module") {
        throw new Error(`[shadow-extension-load-proof] Installed package manifest must declare \"type\": \"module\" at ${packageJsonPath} ` +
            `(received=${describeValue(packageJson.type)})`);
    }
    if (packageJson.name !== extensionId) {
        throw new Error(`[shadow-extension-load-proof] Installed package manifest name must match plugin id ${JSON.stringify(extensionId)} at ${packageJsonPath} ` +
            `(received=${describeValue(packageJson.name)})`);
    }
    if (typeof packageJson.version === "string" &&
        packageJson.version.trim().length > 0 &&
        packageJson.version !== manifest.version) {
        throw new Error(`[shadow-extension-load-proof] Installed package and plugin manifest versions disagree ` +
            `(package=${packageJson.version}, manifest=${manifest.version})`);
    }
    return {
        openclawHome: resolvedOpenclawHome,
        extensionsDir,
        extensionDir,
        manifestPath,
        packageJsonPath,
        loaderEntryPath,
        runtimeGuardPath,
        manifest,
        packageJson,
        configuredEntries
    };
}
export async function proveInstalledOpenClawBrainExtensionLoad(openclawHome, extensionId = "openclawbrain") {
    const inspected = inspectInstalledOpenClawBrainExtension(openclawHome, extensionId);
    const runtimeGuardModule = await importWithHelpfulError(inspected.runtimeGuardPath, "runtime-guard.js");
    const runtimeGuardExportNames = Object.keys(runtimeGuardModule).sort((left, right) => left.localeCompare(right));
    const missingRuntimeGuardExports = REQUIRED_RUNTIME_GUARD_EXPORTS.filter((exportName) => !runtimeGuardExportNames.includes(exportName));
    if (missingRuntimeGuardExports.length > 0) {
        throw new Error(`[shadow-extension-load-proof] Installed runtime-guard.js is missing required exports ` +
            `${missingRuntimeGuardExports.join(", ")} at ${inspected.runtimeGuardPath}`);
    }
    const loaderModule = await importWithHelpfulError(inspected.loaderEntryPath, "loader entry");
    if (typeof loaderModule.default !== "function") {
        throw new Error(`[shadow-extension-load-proof] Installed loader entry default export must be a function at ${inspected.loaderEntryPath} ` +
            `(received=${describeValue(loaderModule.default)})`);
    }
    const registrations = [];
    loaderModule.default({
        on(eventName, handler, options) {
            registrations.push({
                eventName,
                handler,
                priority: typeof options?.priority === "number" ? options.priority : null
            });
        }
    });
    if (registrations.length !== 1) {
        throw new Error(`[shadow-extension-load-proof] Installed loader must register exactly one hook ` +
            `(received=${registrations.length})`);
    }
    const registeredHook = registrations[0];
    if (registeredHook.eventName !== "before_prompt_build") {
        throw new Error(`[shadow-extension-load-proof] Installed loader must register before_prompt_build ` +
            `(received=${JSON.stringify(registeredHook.eventName)})`);
    }
    if (registeredHook.priority !== 5) {
        throw new Error(`[shadow-extension-load-proof] Installed loader must register before_prompt_build with priority 5 ` +
            `(received=${registeredHook.priority === null ? "null" : String(registeredHook.priority)})`);
    }
    const warnings = [];
    const originalWarn = console.warn;
    console.warn = (...args) => {
        warnings.push(args.map((value) => String(value)).join(" "));
    };
    let probeResult;
    try {
        probeResult = await registeredHook.handler({
            messages: {
                content: "not-an-array"
            }
        }, {});
    }
    finally {
        console.warn = originalWarn;
    }
    if (!isRecord(probeResult) || Object.keys(probeResult).length !== 0) {
        throw new Error(`[shadow-extension-load-proof] Malformed before_prompt_build probe must fail open with an empty object ` +
            `(received=${describeValue(probeResult)})`);
    }
    const probeWarning = warnings.find((message) => message.includes("before_prompt_build event.messages is not an array"));
    if (probeWarning === undefined) {
        throw new Error(`[shadow-extension-load-proof] Installed runtime-guard did not emit the expected malformed-event diagnostic ` +
            `(warnings=${warnings.length === 0 ? "none" : warnings.join(" | ")})`);
    }
    const diagnosticLogPath = path.join(homedir(), ".openclawbrain", "extension-errors.log");
    if (!existsSync(diagnosticLogPath)) {
        throw new Error(`[shadow-extension-load-proof] Installed runtime-guard did not write a local diagnostic log at ${diagnosticLogPath}`);
    }
    const diagnosticLogContents = readFileSync(diagnosticLogPath, "utf8");
    if (!diagnosticLogContents.includes("before_prompt_build event.messages is not an array")) {
        throw new Error(`[shadow-extension-load-proof] Local diagnostic log at ${diagnosticLogPath} does not contain the malformed-event failure`);
    }
    return {
        ...inspected,
        runtimeGuardExportNames,
        registeredEventName: registeredHook.eventName,
        registeredPriority: registeredHook.priority,
        probeWarning,
        probeResult,
        diagnosticLogPath,
        diagnosticLogContents
    };
}
function readRequiredJson(filePath, label) {
    let raw;
    try {
        raw = readFileSync(filePath, "utf8");
    }
    catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(`[shadow-extension-load-proof] Failed to read ${label} at ${filePath}: ${detail}`);
    }
    try {
        return JSON.parse(raw);
    }
    catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(`[shadow-extension-load-proof] Failed to parse ${label} at ${filePath}: ${detail}`);
    }
}
function readConfiguredEntries(packageJson, packageJsonPath) {
    const extensions = packageJson.openclaw?.extensions;
    if (!Array.isArray(extensions)) {
        throw new Error(`[shadow-extension-load-proof] Installed package manifest must expose openclaw.extensions as a non-empty string array at ${packageJsonPath} ` +
            `(received=${describeValue(extensions)})`);
    }
    const configuredEntries = extensions.filter((entry) => typeof entry === "string" && entry.trim().length > 0);
    if (configuredEntries.length === 0) {
        throw new Error(`[shadow-extension-load-proof] Installed package manifest openclaw.extensions is empty at ${packageJsonPath}`);
    }
    return configuredEntries;
}
function resolveLoaderEntryPath(extensionDir, configuredEntries) {
    for (const configuredEntry of configuredEntries) {
        const candidate = path.join(extensionDir, configuredEntry);
        if (existsSync(candidate)) {
            return candidate;
        }
    }
    throw new Error(`[shadow-extension-load-proof] Installed package manifest points at missing loader entries in ${extensionDir} ` +
        `(configured=${configuredEntries.join(", ")})`);
}
async function importWithHelpfulError(filePath, label) {
    try {
        return await import(pathToFileURL(filePath).href);
    }
    catch (error) {
        const detail = error instanceof Error ? error.stack ?? error.message : String(error);
        throw new Error(`[shadow-extension-load-proof] Installed ${label} failed to import from ${filePath}: ${detail}`);
    }
}
function describeValue(value) {
    if (value === null) {
        return "null";
    }
    if (Array.isArray(value)) {
        return `array(len=${value.length})`;
    }
    if (typeof value === "string") {
        return `string(${JSON.stringify(value)})`;
    }
    if (typeof value === "object") {
        return `object(keys=${Object.keys(value).join(",") || "none"})`;
    }
    return `${typeof value}(${String(value)})`;
}
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
//# sourceMappingURL=shadow-extension-proof.js.map