declare const OPENCLAWBRAIN_PROVIDER_DEFAULTS_CONTRACT = "openclawbrain_provider_defaults.v1";
declare const ALLOWED_TEACHER_PROVIDERS: readonly ["heuristic", "ollama", "off"];
declare const ALLOWED_EMBEDDER_PROVIDERS: readonly ["keywords", "ollama", "off"];
export type OpenClawBrainProviderConfigEnv = Readonly<Record<string, string | undefined>>;
export type OpenClawBrainTeacherProvider = (typeof ALLOWED_TEACHER_PROVIDERS)[number];
export type OpenClawBrainEmbedderProvider = (typeof ALLOWED_EMBEDDER_PROVIDERS)[number];
export type OpenClawBrainProviderResolvedNote = string;
export interface OpenClawBrainProviderDefaultsRecord {
    contract: typeof OPENCLAWBRAIN_PROVIDER_DEFAULTS_CONTRACT;
    writtenAt: string;
    source: "install";
    teacherBaseUrl?: string;
    embedderBaseUrl?: string;
    teacher?: {
        provider?: OpenClawBrainTeacherProvider;
        model?: string | null;
        timeoutMs?: number;
        maxPromptChars?: number;
        maxResponseChars?: number;
        maxOutputTokens?: number;
        maxArtifactsPerExport?: number;
        maxInteractionsPerExport?: number;
        detectedLocally?: boolean;
        detectedFromModel?: string | null;
    };
    embedder?: {
        provider?: OpenClawBrainEmbedderProvider;
        model?: string | null;
    };
}
export interface OpenClawBrainProviderDefaultsReadResult {
    defaults: OpenClawBrainProviderDefaultsRecord | null;
    warnings: string[];
}
export interface ReadOpenClawBrainProviderConfigInput {
    env?: OpenClawBrainProviderConfigEnv;
    activationRoot?: string | null;
    defaults?: OpenClawBrainProviderDefaultsRecord | null;
}
export interface OpenClawBrainProviderSelection<TProvider extends string> {
    provider: TProvider;
    model: string;
}
export interface OpenClawBrainTeacherConfig extends OpenClawBrainProviderSelection<OpenClawBrainTeacherProvider> {
    timeoutMs?: number;
    maxPromptChars?: number;
    maxResponseChars?: number;
    maxOutputTokens?: number;
    maxArtifactsPerExport?: number;
    maxInteractionsPerExport?: number;
}
export interface OpenClawBrainProviderConfig {
    teacherBaseUrl: string;
    embedderBaseUrl: string;
    teacher: OpenClawBrainTeacherConfig;
    embedder: OpenClawBrainProviderSelection<OpenClawBrainEmbedderProvider>;
    warnings: string[];
}
export declare function resolveOpenClawBrainProviderDefaultsPath(activationRoot: string): string;
export declare function readOpenClawBrainProviderDefaults(activationRoot: string): OpenClawBrainProviderDefaultsReadResult;
export declare function resolveOpenClawBrainProviderConfigNotes(config: OpenClawBrainProviderConfig): OpenClawBrainProviderResolvedNote[];
export declare function readOpenClawBrainProviderConfig(env?: OpenClawBrainProviderConfigEnv): OpenClawBrainProviderConfig;
export declare function readOpenClawBrainProviderConfigFromSources(input?: ReadOpenClawBrainProviderConfigInput): OpenClawBrainProviderConfig;
export {};
