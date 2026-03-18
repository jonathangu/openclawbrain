export interface ExtensionCompileInput {
    activationRoot: string;
    message: string;
    sessionId?: string;
    channel?: string;
    _serveRouteBreadcrumbs?: {
        invocationSurface: "installed_extension_before_prompt_build";
        hostEvent: "before_prompt_build";
        installedEntryPath: string;
    };
}
export interface ExtensionCompileSuccess {
    ok: true;
    brainContext: string;
}
export interface ExtensionCompileFailure {
    ok: false;
    hardRequirementViolated: boolean;
    error: string;
    brainContext: string;
}
export type ExtensionCompileResult = ExtensionCompileSuccess | ExtensionCompileFailure;
export type ExtensionCompileRuntimeContext = (input: ExtensionCompileInput) => ExtensionCompileResult;
export interface ExtensionDiagnostic {
    key: string;
    message: string;
    once?: boolean;
}
export interface ExtensionRegistrationApi {
    on(eventName: string, handler: (event: unknown, ctx: unknown) => Promise<Record<string, unknown>>, options?: {
        priority?: number;
    }): void;
}
export interface NormalizedPromptBuildEvent {
    message: string;
    sessionId?: string;
    channel?: string;
    warnings: ExtensionDiagnostic[];
}
export declare function isActivationRootPlaceholder(activationRoot: string): boolean;
export declare function validateExtensionRegistrationApi(api: unknown): {
    ok: true;
    api: ExtensionRegistrationApi;
} | {
    ok: false;
    diagnostic: ExtensionDiagnostic;
};
export declare function normalizePromptBuildEvent(event: unknown): {
    ok: true;
    event: NormalizedPromptBuildEvent;
} | {
    ok: false;
    diagnostic: ExtensionDiagnostic;
};
export declare function createBeforePromptBuildHandler(input: {
    activationRoot: string;
    compileRuntimeContext: ExtensionCompileRuntimeContext;
    reportDiagnostic: (diagnostic: ExtensionDiagnostic) => void | Promise<void>;
    debug?: (message: string) => void;
    extensionEntryPath?: string;
}): (event: unknown, ctx: unknown) => Promise<Record<string, unknown>>;
