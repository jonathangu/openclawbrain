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
    manifestId: string | null;
    installId: string | null;
    packageName: string | null;
    installLayout: OpenClawBrainInstallLayout;
}
export interface OpenClawBrainInstallTarget {
    installLayout: OpenClawBrainInstallLayout;
    extensionDir: string;
    hookPath: string;
    writeMode: "pin_native_package" | "write_shadow_extension";
    selectedInstall: OpenClawBrainInstalledPlugin | null;
    additionalInstalls: OpenClawBrainInstalledPlugin[];
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
export declare const OPENCLAWBRAIN_NATIVE_INSTALL_ID = "openclaw";
export declare function describeOpenClawBrainInstallLayout(installLayout: OpenClawBrainInstallLayout): string;
export declare function getOpenClawBrainAllowedPluginIds(install: OpenClawBrainInstalledPlugin | null | undefined): string[];
export declare function getOpenClawBrainKnownPluginIds(install: OpenClawBrainInstalledPlugin | null | undefined): string[];
export declare function normalizeOpenClawBrainPluginsConfig(pluginsConfig: Record<string, unknown>, install: OpenClawBrainInstalledPlugin | null | undefined): {
    changed: boolean;
    pluginsConfig: Record<string, unknown>;
    changes: string[];
    allowedPluginIds: string[];
    canonicalEntryId: string;
};
export declare function describeOpenClawBrainInstallIdentity(install: OpenClawBrainInstalledPlugin): string;
export declare function findInstalledOpenClawBrainPlugin(openclawHome: string, pluginId?: string): OpenClawBrainInstalledPluginLookup;
export declare function resolveOpenClawBrainInstallTarget(openclawHome: string): OpenClawBrainInstallTarget;
export declare function pinInstalledOpenClawBrainPluginActivationRoot(loaderEntryPath: string, activationRoot: string): void;
export declare function resolveOpenClawHomeFromExtensionEntryPath(extensionEntryPath: string): string | null;
