export interface NativeSqliteSmokeResult {
    ok: boolean;
    nodeVersion: string;
    sqliteEngine: string;
    fts5: boolean;
    error?: string;
}
export declare function nativeSqliteSmokeTest(): NativeSqliteSmokeResult;
