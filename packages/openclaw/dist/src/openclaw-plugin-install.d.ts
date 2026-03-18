export type OpenClawBrainInstallLayout = "generated_shadow_extension" | "native_package_plugin";
export interface OpenClawBrainInstalledPlugin {
    openclawHome: string;
    extensionsDir: string;
    extensionDir: string;
    manifestPath: string;
    packageJsonPath: string;
    loaderEntryPath: string | null;
    runtimeGuardPath: string | null;
    configuredEntries: string[];
    packageName: string | null;
    installLayout: OpenClawBrainInstallLayout;
}
export interface OpenClawBrainInstalledPluginLookup {
    openclawHome: string;
    extensionsDir: string;
    selectedInstall: OpenClawBrainInstalledPlugin | null;
    additionalInstalls: OpenClawBrainInstalledPlugin[];
}
export declare const OPENCLAWBRAIN_PLUGIN_ID = "openclawbrain";
export declare const OPENCLAWBRAIN_SHADOW_PACKAGE_NAME = "openclawbrain";
export declare const OPENCLAWBRAIN_NATIVE_PACKAGE_NAME = "@openclawbrain/openclaw";
export declare function describeOpenClawBrainInstallLayout(installLayout: OpenClawBrainInstallLayout): string;
export declare function findInstalledOpenClawBrainPlugin(openclawHome: string, pluginId?: string): OpenClawBrainInstalledPluginLookup;
export declare function resolveOpenClawHomeFromExtensionEntryPath(extensionEntryPath: string): string | null;
