export declare const PLUGIN_ID = "openclawbrain";
export declare const PLUGIN_VERSION = "0.1.0";
export declare const DEFAULT_CONFIG: any;
export declare function resolveOpenClawBrainConfig(api?: any): {
    enabled: boolean;
    mode: any;
    activationRoot: any;
    proofEvents: boolean;
    proofRetentionEvents: number;
    maxContextChars: number;
    includeActivationContext: boolean;
    rawTranscriptUpload: boolean;
    failClosedReason: string;
    scopes: {
        agents: any;
    };
    hooks: {
        allowPromptInjection: boolean;
        allowConversationAccess: boolean;
    };
};
export declare function livePluginEntry(api?: any): any;
export declare function normalizePluginConfig(input?: any): {
    enabled: boolean;
    mode: any;
    activationRoot: any;
    proofEvents: boolean;
    proofRetentionEvents: number;
    maxContextChars: number;
    includeActivationContext: boolean;
    rawTranscriptUpload: boolean;
    failClosedReason: string;
    scopes: {
        agents: any;
    };
    hooks: {
        allowPromptInjection: boolean;
        allowConversationAccess: boolean;
    };
};
export declare function normalizeScopes(scopes?: any): {
    agents: any;
};
export declare function activationRootForAgent(config: any, agentId?: any): string;
export declare function isAgentAllowed(config: any, agentId: any): any;
export declare function isRemoteUrl(value: any): boolean;
