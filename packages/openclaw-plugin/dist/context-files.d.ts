export declare const FIXED_CONTEXT_FILES: readonly string[];
export declare function ensureActivationRoot(config: any, agentId: any): Promise<string>;
export declare function readActivationContext(config: any, agentId: any, decision: any): Promise<{
    text: string;
    usedFileIdsRedacted: string[];
    rejectedFiles: {
        fileIdRedacted: string;
        reasonCode: string;
    }[];
}>;
export declare function buildInjectionText(decision: any, activationContext: any, config: any): string;
export declare function filesForDecision(decision: any): string[];
