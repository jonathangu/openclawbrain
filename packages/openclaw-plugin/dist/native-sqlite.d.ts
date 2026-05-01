export interface NativeSqliteSmokeResult {
    ok: boolean;
    nodeVersion: string;
    betterSqlite3: string;
    fts5: boolean;
    error?: string;
}
export declare function nativeSqliteSmokeTest(): NativeSqliteSmokeResult;
