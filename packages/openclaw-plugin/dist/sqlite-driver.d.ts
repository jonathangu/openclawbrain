export interface StatementLike {
    run(...params: any[]): any;
    get(...params: any[]): any;
    all(...params: any[]): any[];
}
export interface DatabaseLike {
    exec(sql: string): void;
    prepare(sql: string): StatementLike;
    pragma(sql: string, options?: {
        simple?: boolean;
    }): any;
    transaction<T>(fn: () => T): () => T;
    close(): void;
}
export interface OpenDatabaseResult {
    db: DatabaseLike;
    engine: 'better-sqlite3' | 'node:sqlite';
}
export declare function openDatabase(filename: string): OpenDatabaseResult;
export declare function isNativeBindingFailure(error: any): boolean;
