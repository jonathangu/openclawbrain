/**
 * Brain import/export: backup and restore the activation root directory.
 *
 * export: tar + gzip the entire activation root → output.tar.gz
 * import: extract tar.gz → activation root, with safety checks
 */
export interface ExportOptions {
    activationRoot: string;
    outputPath: string;
}
export interface ExportResult {
    ok: boolean;
    outputPath: string;
    activationRoot: string;
    error?: string;
}
export interface ImportOptions {
    archivePath: string;
    activationRoot: string;
    force: boolean;
}
export interface ImportResult {
    ok: boolean;
    activationRoot: string;
    archivePath: string;
    warning?: string;
    error?: string;
}
/**
 * Export (backup) the activation root to a tar.gz archive.
 */
export declare function exportBrain(options: ExportOptions): ExportResult;
/**
 * Import (restore) a tar.gz archive into the activation root.
 */
export declare function importBrain(options: ImportOptions): ImportResult;
