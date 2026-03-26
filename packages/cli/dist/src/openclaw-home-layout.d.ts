export type OpenClawHomeLayout = "per_profile_home" | "shared_home_profiles_in_config" | "single_openclaw_home" | "custom_openclaw_home";
export type OpenClawHomeProfileSource = "openclaw_json_profile" | "openclaw_json_single_profile_key" | "directory_name" | "none";
export interface OpenClawHomeInspection {
    openclawHome: string;
    openclawJsonPath: string;
    layout: OpenClawHomeLayout;
    profileId: string | null;
    profileSource: OpenClawHomeProfileSource;
    configuredProfileIds: string[];
    configReadable: boolean;
    configError: string | null;
}
export declare function inspectOpenClawHome(openclawHome: string): OpenClawHomeInspection;
export declare function discoverOpenClawHomes(homeDir?: string): OpenClawHomeInspection[];
export declare function formatOpenClawHomeLayout(layout: OpenClawHomeLayout): string;
export declare function formatOpenClawHomeProfileSource(source: OpenClawHomeProfileSource): string;
export declare function describeOpenClawHomeInspection(inspection: OpenClawHomeInspection): string;
