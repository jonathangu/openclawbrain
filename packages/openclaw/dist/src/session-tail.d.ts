import { type OpenClawSessionStoreV1 } from "./session-store.js";
import type { ScannedEventExportInputV1 } from "./index.js";
export type OpenClawLocalSessionTailNoopReasonV1 = "seeded_existing_sessions" | "no_local_session_stores" | "no_session_changes";
export type OpenClawLocalSessionTailChangeKindV1 = "new_session" | "appended_records" | "metadata_only" | "session_reset" | "missing_session_path" | "missing_session_file";
export interface OpenClawLocalSessionTailCursorV1 {
    sourceIndexPath: string;
    sessionKey: string;
    sessionId: string;
    sessionFile: string | null;
    updatedAt: number;
    rawRecordCount: number;
    bridgedEventCount: number;
}
export interface OpenClawLocalSessionTailChangeV1 {
    source: OpenClawSessionStoreV1;
    sessionKey: string;
    sessionId: string;
    sessionFile: string | null;
    changeKind: OpenClawLocalSessionTailChangeKindV1;
    rawRecordCount: number;
    bridgedEventCount: number;
    emittedEventCount: number;
    lastUserMessageAt: string | null;
    lastUserMessageText: string | null;
    warnings: string[];
    scannedEventExport: ScannedEventExportInputV1 | null;
}
export interface OpenClawLocalSessionTailPollResultV1 {
    runtimeOwner: "openclaw";
    lane: "local_session_tail";
    polledAt: string;
    sources: OpenClawSessionStoreV1[];
    changes: OpenClawLocalSessionTailChangeV1[];
    noopReason: OpenClawLocalSessionTailNoopReasonV1 | null;
    warnings: string[];
    cursor: OpenClawLocalSessionTailCursorV1[];
}
export interface OpenClawLocalSessionTailInput {
    homeDir?: string;
    profileRoots?: readonly string[];
    cursor?: readonly OpenClawLocalSessionTailCursorV1[];
    emitExistingOnFirstPoll?: boolean;
}
export interface OpenClawLocalSessionTailLoopOptionsV1 {
    pollIntervalMs?: number;
    maxPasses?: number;
    stopWhenIdle?: boolean;
    signal?: AbortSignal;
    onPass?: (result: OpenClawLocalSessionTailPollResultV1) => void | Promise<void>;
}
export interface OpenClawLocalSessionTailLoopResultV1 {
    runtimeOwner: "openclaw";
    passCount: number;
    changedSessionCount: number;
    emittedEventCount: number;
    stoppedReason: "idle" | "max_passes" | "aborted";
    lastPoll: OpenClawLocalSessionTailPollResultV1 | null;
    cursor: OpenClawLocalSessionTailCursorV1[];
}
export declare class OpenClawLocalSessionTail {
    readonly homeDir: string | undefined;
    readonly profileRoots: readonly string[] | undefined;
    private initialized;
    private readonly emitExistingOnFirstPoll;
    private readonly cursorBySession;
    constructor(input?: OpenClawLocalSessionTailInput);
    snapshot(): OpenClawLocalSessionTailCursorV1[];
    pollOnce(options?: {
        observedAt?: string;
    }): OpenClawLocalSessionTailPollResultV1;
    runLoop(options?: OpenClawLocalSessionTailLoopOptionsV1): Promise<OpenClawLocalSessionTailLoopResultV1>;
}
export declare function createOpenClawLocalSessionTail(input?: OpenClawLocalSessionTailInput): OpenClawLocalSessionTail;
