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

import {
  compileRuntimeContext,
  recordOpenClawProfileRuntimeLoadProof
} from "@openclawbrain/openclaw";
import {
  createBeforePromptBuildHandler,
  type ExtensionDiagnostic,
  isActivationRootPlaceholder,
  validateExtensionRegistrationApi
} from "./runtime-guard.js";

const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";
const EXTENSION_ENTRY_PATH = fileURLToPath(import.meta.url);
const warnedDiagnostics = new Set<string>();

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

function announceStartupBreadcrumb(): void {
  if (isActivationRootPlaceholder(ACTIVATION_ROOT)) {
    warnOnce(
      "startup-brain-not-yet-loaded",
      "[openclawbrain] BRAIN NOT YET LOADED: install has not pinned ACTIVATION_ROOT yet. Install @openclawbrain/cli, then run: openclawbrain install --openclaw-home <path>"
    );
    return;
  }

  warnOnce(
    "startup-brain-loaded",
    `[openclawbrain] BRAIN LOADED: runtime hook registered for before_prompt_build (activationRoot=${ACTIVATION_ROOT})`
  );
}

export default function register(api: unknown) {
  const registration = validateExtensionRegistrationApi(api);
  if (!registration.ok) {
    void reportDiagnostic(registration.diagnostic);
    return;
  }

  try {
    registration.api.on(
      "before_prompt_build",
      createBeforePromptBuildHandler({
        activationRoot: ACTIVATION_ROOT,
        extensionEntryPath: EXTENSION_ENTRY_PATH,
        compileRuntimeContext,
        reportDiagnostic,
        debug: (message) => console.debug(message)
      }),
      { priority: 5 }
    );
    if (!isActivationRootPlaceholder(ACTIVATION_ROOT)) {
      try {
        recordOpenClawProfileRuntimeLoadProof({
          activationRoot: ACTIVATION_ROOT,
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
