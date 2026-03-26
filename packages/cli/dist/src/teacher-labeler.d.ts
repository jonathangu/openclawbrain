import { type NormalizedEventExportV1, type TeacherSupervisionArtifactV1 } from "@openclawbrain/contracts";
import type { LearningSpineServeRouteDecisionLogEntryV1 } from "@openclawbrain/pack-format";
export interface TeacherLabelerRunInputV1 {
    normalizedEventExport: NormalizedEventExportV1;
    observedAt: string;
    staleAfterMs: number;
    serveTimeDecisions?: readonly LearningSpineServeRouteDecisionLogEntryV1[];
    existingArtifacts?: readonly TeacherSupervisionArtifactV1[];
}
export interface TeacherLabelerResultV1 {
    artifacts: TeacherSupervisionArtifactV1[];
    status: "ok" | "skipped" | "fail_open";
    detail: string;
}
export interface TeacherLabeler {
    label(input: TeacherLabelerRunInputV1): Promise<TeacherLabelerResultV1>;
}
export interface OllamaTeacherLabelerGenerateInputV1 {
    model: string;
    prompt: string;
    maxOutputTokens: number;
    timeoutMs: number;
}
export interface OllamaTeacherLabelerClient {
    generate(input: OllamaTeacherLabelerGenerateInputV1): Promise<{
        response: string;
    }>;
}
export interface AsyncTeacherOllamaLabelerConfigV1 {
    provider: "ollama";
    baseUrl?: string;
    model?: string;
    timeoutMs?: number;
    maxPromptChars?: number;
    maxResponseChars?: number;
    maxOutputTokens?: number;
    maxArtifactsPerExport?: number;
    maxInteractionsPerExport?: number;
    maxUserMessageChars?: number;
    maxContextIdsPerDecision?: number;
    teacherIdentity?: string;
    client?: OllamaTeacherLabelerClient;
}
export interface AsyncTeacherNoopLabelerConfigV1 {
    provider: "none";
}
export type AsyncTeacherLabelerConfigV1 = AsyncTeacherNoopLabelerConfigV1 | AsyncTeacherOllamaLabelerConfigV1;
export declare function createHttpOllamaTeacherLabelerClient(baseUrl?: string): OllamaTeacherLabelerClient;
export declare function createOllamaTeacherLabeler(config: AsyncTeacherOllamaLabelerConfigV1): TeacherLabeler;
export declare function createTeacherLabeler(config: AsyncTeacherLabelerConfigV1 | null | undefined): TeacherLabeler | null;
