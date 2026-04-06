/**
 * OpenClawBrain extension template — canonical, pre-built, versioned with the package.
 *
 * The placeholder __ACTIVATION_ROOT__ is replaced by
 * OpenClawBrain's `openclawbrain install`
 * with the real activation root path at install time.
 *
 * Design constraints:
 *   - Empty brain → returns empty object (no context injected)
 *   - Compilation errors → fail-open, never breaks the session
 *   - Missing activation root → fail-open with console.warn
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  compileRuntimeContext,
  listOpenClawProfileRuntimeLoadProofs,
  recordOpenClawProfileRuntimeLoadProof,
  resolveAttachmentRuntimeLoadProofsPath
} from "@openclawbrain/openclaw";
import {
  createBeforePromptBuildHandler,
  type ExtensionDiagnostic,
  isActivationRootPlaceholder,
  resolveInstalledActivationRoot,
  validateExtensionRegistrationApi
} from "./runtime-guard.js";

const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";
const EXTENSION_ENTRY_PATH = fileURLToPath(import.meta.url);
const RESOLVED_ACTIVATION_ROOT = resolveInstalledActivationRoot({
  activationRoot: ACTIVATION_ROOT,
  extensionEntryPath: EXTENSION_ENTRY_PATH
});
const warnedDiagnostics = new Set<string>();
const RUNTIME_LOAD_PROOFS_CONTRACT = "openclaw_profile_runtime_load_proofs.v1";
const ACTIVATION_ROOT_PIN_PATTERN = /const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";/;

function warnOnce(key: string, message: string): void {
  if (warnedDiagnostics.has(key)) {
    return;
  }

  warnedDiagnostics.add(key);
  console.warn(message);
}

async function appendLocalDiagnosticLog(message: string): Promise<void> {
  try {
    const fs = await import("fs");
    const os = await import("os");
    const rootDir = `${os.homedir()}/.openclawbrain`;
    const logPath = `${rootDir}/extension-errors.log`;
    fs.mkdirSync(rootDir, { recursive: true });
    fs.appendFileSync(logPath, `${new Date().toISOString()} ${message.replace(/\s+/g, " ").trim()}\n`);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    warnOnce(
      `local-diagnostic-log:${detail}`,
      `[openclawbrain] failed to append local extension diagnostic log: ${detail}`
    );
  }
}

async function reportDiagnostic(input: ExtensionDiagnostic): Promise<void> {
  if (input.once) {
    if (warnedDiagnostics.has(input.key)) {
      return;
    }
    warnedDiagnostics.add(input.key);
  }

  const formatted = formatDiagnosticMessage(input);
  console.warn(formatted);
  await appendLocalDiagnosticLog(formatted);
}

function formatDiagnosticMessage(input: ExtensionDiagnostic): string {
  if (
    input.severity === undefined ||
    input.actionability === undefined ||
    input.summary === undefined ||
    input.action === undefined
  ) {
    return input.message;
  }

  const detail = input.message.replace(/^\[openclawbrain\]\s*/, "");
  return [
    "[openclawbrain]",
    `severity=${input.severity}`,
    `actionability=${input.actionability}`,
    `summary=${JSON.stringify(input.summary)}`,
    `action=${JSON.stringify(input.action)}`,
    `detail=${JSON.stringify(detail)}`
  ].join(" ");
}

function repairRuntimeLoadProofsIfUnreadable(activationRoot: string): { repaired: boolean; path: string; error: string | null } {
  const loadedProofs = listOpenClawProfileRuntimeLoadProofs(activationRoot);
  if (loadedProofs.error === null) {
    return {
      repaired: false,
      path: loadedProofs.path,
      error: null
    };
  }

  const proofPath = resolveAttachmentRuntimeLoadProofsPath(activationRoot);
  mkdirSync(path.dirname(proofPath), { recursive: true });
  writeFileSync(
    proofPath,
    `${JSON.stringify({
      contract: RUNTIME_LOAD_PROOFS_CONTRACT,
      runtimeOwner: "openclaw",
      activationRoot,
      updatedAt: new Date().toISOString(),
      profiles: []
    }, null, 2)}\n`,
    "utf8"
  );

  return {
    repaired: true,
    path: proofPath,
    error: loadedProofs.error
  };
}

function maybeRegisterRuntimeLoadProofIntegrityService(
  api: { registerService?: (service: { id: string; start: (ctx: unknown) => void | Promise<void> }) => void },
  activationRoot: string,
): void {
  if (typeof api.registerService !== "function" || isActivationRootPlaceholder(activationRoot)) {
    return;
  }

  api.registerService({
    id: "openclawbrain-runtime-load-proof-integrity",
    start: async () => {
      try {
        const repair = repairRuntimeLoadProofsIfUnreadable(activationRoot);
        if (repair.repaired) {
          await reportDiagnostic({
            key: "runtime-load-proof-reset",
            once: true,
            severity: "degraded",
            actionability: "inspect_local_proof_write",
            summary: "runtime-load proof file was unreadable and was reset to an empty proof set",
            action: "Inspect the activation-root proof path if historical runtime-load proof entries were expected; future runtime loads will repopulate the file.",
            message: `[openclawbrain] runtime load proof file was unreadable and was reset: ${repair.path} (${repair.error ?? "unknown error"})`
          });
        }
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        await reportDiagnostic({
          key: "runtime-load-proof-reset-failed",
          once: true,
          severity: "degraded",
          actionability: "inspect_local_proof_write",
          summary: "runtime-load proof integrity service could not repair the local proof file",
          action: "Inspect the activation-root proof path permissions and contents; runtime load proof capture may stay degraded until repaired.",
          message: `[openclawbrain] runtime load proof repair failed: ${detail}`
        });
      }
    }
  });
}

function announceStartupBreadcrumb(): void {
  if (isActivationRootPlaceholder(RESOLVED_ACTIVATION_ROOT.activationRoot)) {
    warnOnce(
      "startup-brain-not-yet-loaded",
      "[openclawbrain] BRAIN NOT YET LOADED: install has not pinned ACTIVATION_ROOT yet. Install OpenClawBrain, then run: openclawbrain install --openclaw-home <path>"
    );
    return;
  }

  if (RESOLVED_ACTIVATION_ROOT.recoveredFromPlaceholder) {
    warnOnce(
      "startup-brain-loaded-recovered",
      `[openclawbrain] BRAIN LOADED: runtime hook recovered activationRoot=${RESOLVED_ACTIVATION_ROOT.activationRoot} from the installed extension location because the loader file still contains the ACTIVATION_ROOT placeholder`
    );
    return;
  }

  warnOnce(
    "startup-brain-loaded",
    `[openclawbrain] BRAIN LOADED: runtime hook registered for before_prompt_build (activationRoot=${RESOLVED_ACTIVATION_ROOT.activationRoot})`
  );
}

function selfHealPinnedActivationRoot(): void {
  if (!RESOLVED_ACTIVATION_ROOT.recoveredFromPlaceholder) {
    return;
  }

  try {
    const loaderSource = readFileSync(EXTENSION_ENTRY_PATH, "utf8");
    if (!ACTIVATION_ROOT_PIN_PATTERN.test(loaderSource)) {
      return;
    }

    const nextLoaderSource = loaderSource.replace(
      ACTIVATION_ROOT_PIN_PATTERN,
      `const ACTIVATION_ROOT = ${JSON.stringify(RESOLVED_ACTIVATION_ROOT.activationRoot)};`
    );
    if (nextLoaderSource !== loaderSource) {
      writeFileSync(EXTENSION_ENTRY_PATH, nextLoaderSource, "utf8");
      warnOnce(
        "startup-brain-self-healed",
        `[openclawbrain] BRAIN SELF-HEALED: repinned ACTIVATION_ROOT in ${EXTENSION_ENTRY_PATH} to ${RESOLVED_ACTIVATION_ROOT.activationRoot}`
      );
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    warnOnce(
      `startup-brain-self-heal-failed:${detail}`,
      `[openclawbrain] failed to self-heal ACTIVATION_ROOT pin in ${EXTENSION_ENTRY_PATH}: ${detail}`
    );
  }
}

function register(api: unknown) {
  const registration = validateExtensionRegistrationApi(api);
  if (!registration.ok) {
    void reportDiagnostic(registration.diagnostic);
    return;
  }

  try {
    selfHealPinnedActivationRoot();
    maybeRegisterRuntimeLoadProofIntegrityService(registration.api, RESOLVED_ACTIVATION_ROOT.activationRoot);
    registration.api.on(
      "before_prompt_build",
      createBeforePromptBuildHandler({
        activationRoot: RESOLVED_ACTIVATION_ROOT.activationRoot,
        extensionEntryPath: EXTENSION_ENTRY_PATH,
        compileRuntimeContext,
        reportDiagnostic,
        debug: (message) => console.debug(message)
      }),
      { priority: 5 }
    );
    if (!isActivationRootPlaceholder(RESOLVED_ACTIVATION_ROOT.activationRoot)) {
      try {
        const repair = repairRuntimeLoadProofsIfUnreadable(RESOLVED_ACTIVATION_ROOT.activationRoot);
        if (repair.repaired) {
          void reportDiagnostic({
            key: "runtime-load-proof-reset",
            once: true,
            severity: "degraded",
            actionability: "inspect_local_proof_write",
            summary: "runtime-load proof file was unreadable and was reset to an empty proof set",
            action: "Inspect the activation-root proof path if historical runtime-load proof entries were expected; future runtime loads will repopulate the file.",
            message: `[openclawbrain] runtime load proof file was unreadable and was reset: ${repair.path} (${repair.error ?? "unknown error"})`
          });
        }
        recordOpenClawProfileRuntimeLoadProof({
          activationRoot: RESOLVED_ACTIVATION_ROOT.activationRoot,
          extensionEntryPath: EXTENSION_ENTRY_PATH
        });
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        void reportDiagnostic({
          key: "runtime-load-proof-failed",
          once: true,
          severity: "degraded",
          actionability: "inspect_local_proof_write",
          summary: "runtime-load proof write failed after hook registration",
          action: "Inspect local filesystem permissions and the activation-root proof path if proof capture is expected.",
          message: `[openclawbrain] runtime load proof failed: ${detail}`
        });
      }
    }
    announceStartupBreadcrumb();
  } catch (error) {
    const detail = error instanceof Error ? error.stack ?? error.message : String(error);
    void reportDiagnostic({
      key: "registration-failed",
      once: true,
      severity: "blocking",
      actionability: "rerun_install",
      summary: "extension registration threw before the runtime hook was fully attached",
      action: "Rerun openclawbrain install --openclaw-home <path>; if it still fails, inspect the extension loader/runtime.",
      message: `[openclawbrain] extension registration failed: ${detail}`
    });
  }
}

const openclawbrainPlugin = {
  id: "openclawbrain",
  name: "OpenClawBrain",
  description: "Learned memory and context from OpenClawBrain",
  register
};

export default openclawbrainPlugin;
