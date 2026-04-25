import { existsSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
function normalizeOptionalString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}
function readRecord(value) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    return value;
}
function listConfiguredProfileIds(config) {
    const profiles = readRecord(config?.profiles);
    if (profiles === null) {
        return [];
    }
    return Object.keys(profiles)
        .map((profileId) => profileId.trim())
        .filter((profileId) => profileId.length > 0)
        .sort((left, right) => left.localeCompare(right));
}
function deriveDirectoryProfileId(openclawHome) {
    const basename = path.basename(openclawHome);
    if (!basename.startsWith(".openclaw-")) {
        return null;
    }
    const derived = basename.slice(".openclaw-".length).trim();
    return derived.length > 0 ? derived : null;
}
function detectLayout(input) {
    const basename = path.basename(input.openclawHome);
    if (basename.startsWith(".openclaw-")) {
        return "per_profile_home";
    }
    if (basename === ".openclaw" && input.configuredProfileIds.length > 0) {
        return "shared_home_profiles_in_config";
    }
    if (basename === ".openclaw") {
        return "single_openclaw_home";
    }
    if (input.configuredProfileIds.length > 0) {
        return "shared_home_profiles_in_config";
    }
    return "custom_openclaw_home";
}
function detectProfileId(input) {
    const directProfile = normalizeOptionalString(input.config?.profile);
    if (directProfile !== null) {
        return {
            profileId: directProfile,
            profileSource: "openclaw_json_profile"
        };
    }
    if (input.configuredProfileIds.length === 1) {
        return {
            profileId: input.configuredProfileIds[0],
            profileSource: "openclaw_json_single_profile_key"
        };
    }
    const directoryProfile = deriveDirectoryProfileId(input.openclawHome);
    if (directoryProfile !== null) {
        return {
            profileId: directoryProfile,
            profileSource: "directory_name"
        };
    }
    return {
        profileId: null,
        profileSource: "none"
    };
}
export function inspectOpenClawHome(openclawHome) {
    const resolvedHome = path.resolve(openclawHome);
    const openclawJsonPath = path.join(resolvedHome, "openclaw.json");
    let config = null;
    let configReadable = false;
    let configError = null;
    if (existsSync(openclawJsonPath)) {
        try {
            config = readRecord(JSON.parse(readFileSync(openclawJsonPath, "utf8")));
            configReadable = config !== null;
            if (config === null) {
                configError = "openclaw.json does not contain a top-level object";
            }
        }
        catch (error) {
            configError = error instanceof Error ? error.message : String(error);
        }
    }
    else {
        configError = "openclaw.json is missing";
    }
    const configuredProfileIds = listConfiguredProfileIds(config);
    const layout = detectLayout({
        openclawHome: resolvedHome,
        configuredProfileIds
    });
    const profileResolution = detectProfileId({
        openclawHome: resolvedHome,
        config,
        configuredProfileIds
    });
    return {
        openclawHome: resolvedHome,
        openclawJsonPath,
        layout,
        profileId: profileResolution.profileId,
        profileSource: profileResolution.profileSource,
        configuredProfileIds,
        configReadable,
        configError
    };
}
function isDiscoverableOpenClawHome(entry) {
    if (!entry.isDirectory()) {
        return false;
    }
    return entry.name === ".openclaw" || entry.name.startsWith(".openclaw-");
}
export function discoverOpenClawHomes(homeDir = process.env.HOME ?? process.env.USERPROFILE ?? homedir()) {
    const resolvedHomeDir = path.resolve(homeDir);
    let entries;
    try {
        entries = readdirSync(resolvedHomeDir, { withFileTypes: true });
    }
    catch {
        return [];
    }
    return entries
        .filter(isDiscoverableOpenClawHome)
        .map((entry) => path.join(resolvedHomeDir, entry.name))
        .filter((candidate) => existsSync(path.join(candidate, "openclaw.json")))
        .map((candidate) => inspectOpenClawHome(candidate))
        .sort((left, right) => left.openclawHome.localeCompare(right.openclawHome));
}
export function formatOpenClawHomeLayout(layout) {
    switch (layout) {
        case "per_profile_home":
            return "per-profile home";
        case "shared_home_profiles_in_config":
            return "single ~/.openclaw home with profiles in openclaw.json";
        case "single_openclaw_home":
            return "single ~/.openclaw home";
        case "custom_openclaw_home":
            return "custom OpenClaw home";
        default:
            return layout;
    }
}
export function formatOpenClawHomeProfileSource(source) {
    switch (source) {
        case "openclaw_json_profile":
            return "openclaw.json.profile";
        case "openclaw_json_single_profile_key":
            return "the only openclaw.json profiles entry";
        case "directory_name":
            return "the OpenClaw home directory name";
        case "none":
            return "none";
        default:
            return source;
    }
}
export function describeOpenClawHomeInspection(inspection) {
    const layout = formatOpenClawHomeLayout(inspection.layout);
    const configuredProfiles = inspection.configuredProfileIds.length === 0
        ? "none"
        : inspection.configuredProfileIds.join(", ");
    if (inspection.profileId !== null) {
        return `${layout}; target profile=${inspection.profileId} via ${formatOpenClawHomeProfileSource(inspection.profileSource)}; configured profiles=${configuredProfiles}`;
    }
    if (inspection.layout === "shared_home_profiles_in_config" || inspection.layout === "single_openclaw_home") {
        return `${layout}; target profile stays host-selected current_profile; configured profiles=${configuredProfiles}`;
    }
    return `${layout}; target profile unresolved; configured profiles=${configuredProfiles}`;
}
