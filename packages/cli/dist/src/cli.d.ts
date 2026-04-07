#!/usr/bin/env node
import { type TextEmbedder } from "@openclawbrain/compiler";
import { type DaemonCliArgs } from "./daemon.js";
import { createAsyncTeacherLiveLoop, createOpenClawLocalSessionTail, createRuntimeEventExportScanner, type AsyncTeacherLabelerConfigV1, type CurrentProfileBrainStatusInput, type WatchEmbedInstrumentationTraceV1 } from "./index.js";
interface ParsedStatusRollbackCliArgs {
    command: "status" | "rollback";
    input: CurrentProfileBrainStatusInput;
    openclawHome: string | null;
    json: boolean;
    help: boolean;
    dryRun: boolean;
    detailed: boolean;
}
interface ParsedAttachCliArgs {
    command: "attach";
    openclawHome: string;
    openclawHomeSource: "explicit";
    activationRoot: string;
    activationRootSource: ParsedInstallCliArgs["activationRootSource"];
    shared: boolean;
    workspaceId: string;
    workspaceIdSource: ParsedInstallCliArgs["workspaceIdSource"];
    skipEmbedderProvision: boolean;
    skipEmbedderProvisionSource: ParsedInstallCliArgs["skipEmbedderProvisionSource"];
    json: boolean;
    help: boolean;
}
interface ParsedInstallCliArgs {
    command: "install";
    openclawHome: string;
    openclawHomeSource: "explicit" | "env" | "discovered_single_home";
    activationRoot: string;
    activationRootSource: "explicit" | "default_from_openclaw_home";
    shared: boolean;
    workspaceId: string;
    workspaceIdSource: "explicit" | "openclaw_json_profile" | "openclaw_json_single_profile_key" | "current_profile_boundary" | "openclaw_home_dir" | "fallback";
    skipEmbedderProvision: boolean;
    skipEmbedderProvisionSource: "flag" | "env" | null;
    json: boolean;
    help: boolean;
}
type RestartMode = "never" | "safe" | "external";
interface ParsedDetachCliArgs {
    command: "detach";
    openclawHome: string;
    activationRoot: string | null;
    restart: RestartMode;
    json: boolean;
    help: boolean;
}
interface ParsedUninstallCliArgs {
    command: "uninstall";
    openclawHome: string;
    activationRoot: string | null;
    dataMode: "keep" | "purge";
    restart: RestartMode;
    json: boolean;
    help: boolean;
}
interface ParsedScanCliArgs {
    command: "scan";
    json: boolean;
    help: boolean;
    sessionPath: string | null;
    livePath: string | null;
    rootDir: string | null;
    workspacePath: string | null;
    packLabel: string | null;
    observedAt: string | null;
    snapshotOutPath: string | null;
}
interface ParsedContextCliArgs {
    command: "context";
    message: string;
    activationRoot: string;
    json: boolean;
    help: boolean;
}
interface ParsedHistoryCliArgs {
    command: "history";
    activationRoot: string;
    limit: number;
    json: boolean;
    help: boolean;
}
interface ParsedLearnCliArgs {
    command: "learn";
    activationRoot: string;
    json: boolean;
    help: boolean;
}
interface ParsedWatchCliArgs {
    command: "watch";
    activationRoot: string;
    scanRoot: string | null;
    interval: number;
    json: boolean;
    help: boolean;
}
interface ParsedExportCliArgs {
    command: "export";
    activationRoot: string;
    outputPath: string;
    json: boolean;
    help: boolean;
}
interface ParsedImportCliArgs {
    command: "import";
    archivePath: string;
    activationRoot: string;
    force: boolean;
    json: boolean;
    help: boolean;
}
interface ParsedResetCliArgs {
    command: "reset";
    activationRoot: string;
    yes: boolean;
    json: boolean;
    help: boolean;
}
interface ParsedProofCliArgs {
    command: "proof";
    openclawHome: string;
    activationRoot: string | null;
    outputDir: string | null;
    skipInstall: boolean;
    skipRestart: boolean;
    pluginId: string;
    timeoutMs: number;
    json: boolean;
    help: boolean;
}
interface ParsedGraphifyExportCliArgs {
    command: "graphify-export";
    activationRoot: string;
    outputRoot: string | null;
    runId: string | null;
    repoRoot: string | null;
    workspaceRoot: string | null;
    sessionKey: string | null;
    sessionSourcePath: string | null;
    proofSummarySourcePath: string | null;
    docsRoot: string | null;
    codeRoot: string | null;
    generatedAt: string | null;
    json: boolean;
    help: boolean;
}
interface ParsedGraphifyRunCliArgs {
    command: "graphify-run";
    sourceBundlePath: string;
    outputRoot: string | null;
    runId: string | null;
    graphifyVersion: string | null;
    graphifyMode: string | null;
    graphifyCommand: string | null;
    graphifyArgs: string[];
    graphifyFlags: string[];
    graphifyConfig: Record<string, unknown>;
    labels: string[];
    json: boolean;
    help: boolean;
}
interface ParsedGraphifyCompiledArtifactsCliArgs {
    command: "graphify-compiled-artifacts";
    bundleId: string | null;
    outputDir: string | null;
    bundleStartedAt: string | null;
    graphifyRunId: string | null;
    graphifyVersion: string | null;
    graphifyCommand: string | null;
    sourceBundleId: string | null;
    sourceBundleHash: string | null;
    graphHash: string | null;
    configHash: string | null;
    labelsHash: string | null;
    json: boolean;
    help: boolean;
}
interface ParsedGraphifyDeterministicLintCliArgs {
    command: "graphify-lints";
    bundleRoot: string | null;
    repoRoot: string;
    workspaceRoot: string;
    outputRoot: string | null;
    runId: string | null;
    json: boolean;
    help: boolean;
}
type ParsedOperatorCliArgs = ParsedStatusRollbackCliArgs | ParsedAttachCliArgs | ParsedScanCliArgs | ParsedInstallCliArgs | ParsedDetachCliArgs | ParsedUninstallCliArgs | ParsedContextCliArgs | ParsedHistoryCliArgs | ParsedLearnCliArgs | ParsedWatchCliArgs | DaemonCliArgs | ParsedExportCliArgs | ParsedImportCliArgs | ParsedResetCliArgs | ParsedProofCliArgs | ParsedGraphifyExportCliArgs | ParsedGraphifyRunCliArgs | ParsedGraphifyCompiledArtifactsCliArgs | ParsedGraphifyDeterministicLintCliArgs;
export declare function parseOperatorCliArgs(argv: readonly string[]): ParsedOperatorCliArgs;
export interface WatchCommandRuntimeV1 {
    activationRoot: string;
    scanRoot: string;
    pollIntervalSeconds: number;
    sessionTailCursorPath: string;
    teacherSnapshotPath: string;
    startupWarnings: string[];
    lastTeacherError: string | null;
    replayState: {
        replayedBundleCount: number;
        replayedEventCount: number;
    };
    lastHandledMaterializationPackId: string | null;
    lastEmbedInstrumentation: WatchEmbedInstrumentationTraceV1 | null;
    scanner: ReturnType<typeof createRuntimeEventExportScanner>;
    teacherLoop: ReturnType<typeof createAsyncTeacherLiveLoop>;
    localSessionTail: ReturnType<typeof createOpenClawLocalSessionTail>;
    embedder: TextEmbedder | null;
    consumeBaselinePersisted: () => boolean;
}
export interface CreateWatchCommandRuntimeInputV1 {
    activationRoot: string;
    scanRoot?: string | null;
    pollIntervalSeconds?: number;
    profileRoots?: readonly string[];
    log?: (message: string) => void;
    teacherLabeler?: AsyncTeacherLabelerConfigV1 | null;
    embedder?: TextEmbedder | null;
}
export interface WatchCommandPassResultV1 {
    localPoll: ReturnType<WatchCommandRuntimeV1["localSessionTail"]["pollOnce"]>;
    exported: {
        exportedBundleCount: number;
        exportedEventCount: number;
        warnings: string[];
    };
    scanResult: ReturnType<WatchCommandRuntimeV1["scanner"]["scanOnce"]>;
    snapshot: ReturnType<WatchCommandRuntimeV1["teacherLoop"]["snapshot"]>;
    materializedPackId: string | null;
}
export declare function createWatchCommandRuntime(input: CreateWatchCommandRuntimeInputV1): Promise<WatchCommandRuntimeV1>;
export declare function runWatchCommandPass(runtime: WatchCommandRuntimeV1, options?: {
    observedAt?: string;
    json?: boolean;
    log?: (message: string) => void;
}): Promise<WatchCommandPassResultV1>;
export declare function runOperatorCli(argv?: readonly string[]): number;
export {};
