import { type OpenClawBrainInstallLayout } from "./openclaw-plugin-install.js";
export interface InstalledExtensionPackageJson {
    name?: unknown;
    version?: unknown;
    private?: unknown;
    type?: unknown;
    openclaw?: {
        extensions?: unknown;
    };
    dependencies?: Record<string, string>;
}
export interface InstalledPluginManifest {
    id?: unknown;
    name?: unknown;
    description?: unknown;
    version?: unknown;
    configSchema?: unknown;
}
export interface ShadowProfileExtensionInstall {
    openclawHome: string;
    extensionsDir: string;
    extensionDir: string;
    installLayout: OpenClawBrainInstallLayout;
    manifestPath: string;
    packageJsonPath: string;
    loaderEntryPath: string;
    runtimeGuardPath: string;
    manifest: InstalledPluginManifest;
    packageJson: InstalledExtensionPackageJson;
    configuredEntries: string[];
}
export interface ShadowProfileExtensionLoadProof extends ShadowProfileExtensionInstall {
    runtimeGuardExportNames: string[];
    registeredEventName: string;
    registeredPriority: number | null;
    probeWarning: string;
    probeResult: Record<string, unknown>;
    diagnosticLogPath: string;
    diagnosticLogContents: string;
}
export declare function inspectInstalledOpenClawBrainExtension(openclawHome: string, extensionId?: string): ShadowProfileExtensionInstall;
export declare function proveInstalledOpenClawBrainExtensionLoad(openclawHome: string, extensionId?: string): Promise<ShadowProfileExtensionLoadProof>;
