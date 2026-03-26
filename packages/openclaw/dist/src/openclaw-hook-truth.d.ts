import { type OpenClawBrainInstallLayout } from "./openclaw-plugin-install.js";
export type OpenClawBrainHookInstallState = "installed" | "not_installed" | "blocked_by_allowlist" | "unverified";
export type OpenClawBrainHookLoadability = "loadable" | "blocked" | "not_installed" | "unverified";
export type OpenClawBrainHookLoadProof = "status_probe_ready" | "not_ready";
export type OpenClawBrainHookGuardSeverity = "none" | "degraded" | "blocking";
export type OpenClawBrainHookGuardActionability = "none" | "pin_openclaw_home" | "repair_install";
export type OpenClawBrainPluginAllowlistState = "unrestricted" | "allowed" | "blocked" | "invalid" | "unverified";
export interface OpenClawBrainHookInspection {
    scope: "exact_openclaw_home" | "activation_root_only";
    openclawHome: string | null;
    extensionDir: string | null;
    hookPath: string | null;
    runtimeGuardPath: string | null;
    manifestPath: string | null;
    packageJsonPath: string | null;
    manifestId: string | null;
    installId: string | null;
    packageName: string | null;
    installLayout: OpenClawBrainInstallLayout | null;
    additionalInstallCount: number;
    installState: OpenClawBrainHookInstallState;
    loadability: OpenClawBrainHookLoadability;
    pluginAllowlistState: OpenClawBrainPluginAllowlistState;
    desynced: boolean;
    detail: string;
}
export interface OpenClawBrainHookLoadSummary extends OpenClawBrainHookInspection {
    loadProof: OpenClawBrainHookLoadProof;
    guardSeverity: OpenClawBrainHookGuardSeverity;
    guardActionability: OpenClawBrainHookGuardActionability;
    guardSummary: string;
    guardAction: string;
}
export declare function inspectOpenClawBrainPluginAllowlist(openclawHome: string): {
    state: Exclude<OpenClawBrainPluginAllowlistState, "unverified">;
    detail: string;
};
export declare function inspectOpenClawBrainHookStatus(openclawHome: string | null | undefined): OpenClawBrainHookInspection;
export declare function summarizeOpenClawBrainHookLoad(inspection: OpenClawBrainHookInspection, statusProbeReady: boolean): OpenClawBrainHookLoadSummary;
