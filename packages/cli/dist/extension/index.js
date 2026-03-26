/**
 * OpenClawBrain extension template — canonical, pre-built, versioned with the package.
 *
 * The placeholder __ACTIVATION_ROOT__ is replaced by
 * `@openclawbrain/cli`'s `openclawbrain install`
 * with the real activation root path at install time.
 *
 * Design constraints:
 *   - Empty brain → returns empty object (no context injected)
 *   - Compilation errors → fail-open, never breaks the session
 *   - Missing activation root → fail-open with console.warn
 */
import { fileURLToPath } from "node:url";
import { compileRuntimeContext, recordOpenClawProfileRuntimeLoadProof } from "@openclawbrain/openclaw";
import { createBeforePromptBuildHandler, isActivationRootPlaceholder, validateExtensionRegistrationApi } from "./runtime-guard.js";
const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";
const EXTENSION_ENTRY_PATH = fileURLToPath(import.meta.url);
const warnedDiagnostics = new Set();
function warnOnce(key, message) {
    if (warnedDiagnostics.has(key)) {
        return;
    }
    warnedDiagnostics.add(key);
    console.warn(message);
}
async function appendLocalDiagnosticLog(message) {
    try {
        const fs = await import("fs");
        const os = await import("os");
        const rootDir = `${os.homedir()}/.openclawbrain`;
        const logPath = `${rootDir}/extension-errors.log`;
        fs.mkdirSync(rootDir, { recursive: true });
        fs.appendFileSync(logPath, `${new Date().toISOString()} ${message.replace(/\s+/g, " ").trim()}\n`);
    }
    catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        warnOnce(`local-diagnostic-log:${detail}`, `[openclawbrain] failed to append local extension diagnostic log: ${detail}`);
    }
}
async function reportDiagnostic(input) {
    if (input.once) {
        if (warnedDiagnostics.has(input.key)) {
            return;
        }
        warnedDiagnostics.add(input.key);
    }
    console.warn(input.message);
    await appendLocalDiagnosticLog(input.message);
}
function announceStartupBreadcrumb() {
    if (isActivationRootPlaceholder(ACTIVATION_ROOT)) {
        warnOnce("startup-brain-not-yet-loaded", "[openclawbrain] BRAIN NOT YET LOADED: install has not pinned ACTIVATION_ROOT yet. Install @openclawbrain/cli, then run: openclawbrain install --openclaw-home <path>");
        return;
    }
    warnOnce("startup-brain-loaded", `[openclawbrain] BRAIN LOADED: runtime hook registered for before_prompt_build (activationRoot=${ACTIVATION_ROOT})`);
}
export default function register(api) {
    const registration = validateExtensionRegistrationApi(api);
    if (!registration.ok) {
        void reportDiagnostic(registration.diagnostic);
        return;
    }
    try {
        registration.api.on("before_prompt_build", createBeforePromptBuildHandler({
            activationRoot: ACTIVATION_ROOT,
            extensionEntryPath: EXTENSION_ENTRY_PATH,
            compileRuntimeContext,
            reportDiagnostic,
            debug: (message) => console.debug(message)
        }), { priority: 5 });
        if (!isActivationRootPlaceholder(ACTIVATION_ROOT)) {
            try {
                recordOpenClawProfileRuntimeLoadProof({
                    activationRoot: ACTIVATION_ROOT,
                    extensionEntryPath: EXTENSION_ENTRY_PATH
                });
            }
            catch (error) {
                const detail = error instanceof Error ? error.message : String(error);
                void reportDiagnostic({
                    key: `runtime-load-proof:${detail}`,
                    once: true,
                    message: `[openclawbrain] runtime load proof failed: ${detail}`
                });
            }
        }
        announceStartupBreadcrumb();
    }
    catch (error) {
        const detail = error instanceof Error ? error.stack ?? error.message : String(error);
        void reportDiagnostic({
            key: "registration-failed",
            once: true,
            message: `[openclawbrain] extension registration failed: ${detail}`
        });
    }
}
//# sourceMappingURL=index.js.map