/**
 * Brain import/export: backup and restore the activation root directory.
 *
 * export: tar + gzip the entire activation root → output.tar.gz
 * import: extract tar.gz → activation root, with safety checks
 */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
/**
 * Verify the activation root looks valid (has activation-pointers.json).
 */
function validateActivationRoot(activationRoot) {
    if (!existsSync(activationRoot)) {
        throw new Error(`Activation root does not exist: ${activationRoot}`);
    }
    const pointersPath = path.join(activationRoot, "activation-pointers.json");
    if (!existsSync(pointersPath)) {
        throw new Error(`activation-pointers.json not found in ${activationRoot}. ` +
            `This doesn't look like a valid activation root.`);
    }
}
/**
 * Check if a tar.gz archive contains activation-pointers.json at the top level.
 */
function archiveContainsPointers(archivePath) {
    try {
        const listing = execSync(`tar tzf ${JSON.stringify(archivePath)}`, {
            encoding: "utf8",
            maxBuffer: 10 * 1024 * 1024,
        });
        const entries = listing.split("\n").map((e) => e.replace(/^\.\//, ""));
        return entries.some((e) => e === "activation-pointers.json" || e.endsWith("/activation-pointers.json"));
    }
    catch {
        return false;
    }
}
/**
 * Check if activation root already has meaningful data.
 */
function activationRootHasData(activationRoot) {
    if (!existsSync(activationRoot))
        return false;
    try {
        const entries = readdirSync(activationRoot);
        return entries.length > 0;
    }
    catch {
        return false;
    }
}
/**
 * Export (backup) the activation root to a tar.gz archive.
 */
export function exportBrain(options) {
    const { activationRoot, outputPath } = options;
    const resolvedRoot = path.resolve(activationRoot);
    const resolvedOutput = path.resolve(outputPath);
    try {
        validateActivationRoot(resolvedRoot);
        // Ensure output directory exists
        const outputDir = path.dirname(resolvedOutput);
        if (!existsSync(outputDir)) {
            mkdirSync(outputDir, { recursive: true });
        }
        // Create tar.gz from the activation root contents
        execSync(`tar czf ${JSON.stringify(resolvedOutput)} -C ${JSON.stringify(resolvedRoot)} .`, { stdio: "pipe" });
        // Verify the archive was created
        if (!existsSync(resolvedOutput)) {
            return {
                ok: false,
                outputPath: resolvedOutput,
                activationRoot: resolvedRoot,
                error: "Archive was not created (tar returned success but file missing)",
            };
        }
        const stats = statSync(resolvedOutput);
        return {
            ok: true,
            outputPath: resolvedOutput,
            activationRoot: resolvedRoot,
        };
    }
    catch (err) {
        return {
            ok: false,
            outputPath: resolvedOutput,
            activationRoot: resolvedRoot,
            error: err instanceof Error ? err.message : String(err),
        };
    }
}
/**
 * Import (restore) a tar.gz archive into the activation root.
 */
export function importBrain(options) {
    const { archivePath, activationRoot, force } = options;
    const resolvedArchive = path.resolve(archivePath);
    const resolvedRoot = path.resolve(activationRoot);
    try {
        // Verify archive exists
        if (!existsSync(resolvedArchive)) {
            return {
                ok: false,
                activationRoot: resolvedRoot,
                archivePath: resolvedArchive,
                error: `Archive not found: ${resolvedArchive}`,
            };
        }
        // Verify archive contains activation-pointers.json
        if (!archiveContainsPointers(resolvedArchive)) {
            return {
                ok: false,
                activationRoot: resolvedRoot,
                archivePath: resolvedArchive,
                error: "Archive does not contain activation-pointers.json. " +
                    "This doesn't look like a valid brain backup.",
            };
        }
        // Check if activation root already has data
        let warning;
        if (activationRootHasData(resolvedRoot)) {
            if (!force) {
                return {
                    ok: false,
                    activationRoot: resolvedRoot,
                    archivePath: resolvedArchive,
                    error: `Activation root ${resolvedRoot} already contains data. ` +
                        `Use --force to overwrite.`,
                };
            }
            warning = `Overwrote existing data in ${resolvedRoot}`;
        }
        // Create activation root if needed
        if (!existsSync(resolvedRoot)) {
            mkdirSync(resolvedRoot, { recursive: true });
        }
        // Extract archive
        execSync(`tar xzf ${JSON.stringify(resolvedArchive)} -C ${JSON.stringify(resolvedRoot)}`, { stdio: "pipe" });
        // Verify extraction produced activation-pointers.json
        const pointersPath = path.join(resolvedRoot, "activation-pointers.json");
        if (!existsSync(pointersPath)) {
            return {
                ok: false,
                activationRoot: resolvedRoot,
                archivePath: resolvedArchive,
                error: "Extraction completed but activation-pointers.json not found. " +
                    "The archive may have a nested directory structure.",
            };
        }
        const result = {
            ok: true,
            activationRoot: resolvedRoot,
            archivePath: resolvedArchive,
        };
        if (warning !== undefined) {
            result.warning = warning;
        }
        return result;
    }
    catch (err) {
        return {
            ok: false,
            activationRoot: resolvedRoot,
            archivePath: resolvedArchive,
            error: err instanceof Error ? err.message : String(err),
        };
    }
}
