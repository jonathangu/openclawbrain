/**
 * Auto-detect the OpenClawBrain activation root.
 *
 * Resolution order:
 *   1. Explicit `--activation-root <path>` (passed as `explicit` arg)
 *   2. A selected OpenClaw home (`openclawHome` option or OPENCLAW_HOME env)
 *   3. Unpinned host auto-detect from ~/.openclawbrain/activation and installed hooks
 *   4. Refuse clearly if host-local signals disagree or are unresolved
 *   5. Fail with a clear error message
 *
 * Exported for use by CLI commands and other agents' code.
 */
export interface ResolveActivationRootOptions {
    /** Value from --activation-root flag, if provided. null/undefined/"" means not provided. */
    explicit?: string | null;
    /** Specific OpenClaw home to inspect for the installed extension. */
    openclawHome?: string | null;
    /** If true, return "" instead of throwing when nothing is found or auto-detect is ambiguous. */
    quiet?: boolean;
}
/**
 * Resolve the activation root path through the detection chain.
 *
 * @returns Absolute path to the activation root.
 * @throws If no activation root can be found (unless `quiet` is true).
 */
export declare function resolveActivationRoot(options?: ResolveActivationRootOptions): string;
