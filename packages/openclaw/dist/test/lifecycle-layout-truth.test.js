import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { inspectOpenClawBrainHookStatus } from "../src/openclaw-hook-truth.js";
import { listOpenClawProfileRuntimeLoadProofs, recordOpenClawProfileRuntimeLoadProof } from "../src/attachment-truth.js";
import { resolveActivationRoot } from "../src/resolve-activation-root.js";
import { inspectInstalledOpenClawBrainExtension, proveInstalledOpenClawBrainExtensionLoad } from "../src/shadow-extension-proof.js";
import { findInstalledOpenClawBrainPlugin, normalizeOpenClawBrainPluginsConfig, pinInstalledOpenClawBrainPluginActivationRoot, resolveOpenClawBrainInstallTarget } from "../src/openclaw-plugin-install.js";
function canonicalizePath(filePath) {
    try {
        return realpathSync(filePath);
    }
    catch {
        return path.resolve(filePath);
    }
}
function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-lifecycle-proof-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}
function createOpenClawHome(rootDir, name = ".openclaw", config = {
    profile: "Tern",
    plugins: {
        allow: ["openclawbrain"]
    }
}) {
    const openclawHome = path.join(rootDir, name);
    mkdirSync(openclawHome, { recursive: true });
    writeFileSync(path.join(openclawHome, "openclaw.json"), JSON.stringify(config, null, 2));
    return openclawHome;
}
function createShadowInstall(openclawHome, activationRoot) {
    const extensionDir = path.join(openclawHome, "extensions", "openclawbrain");
    mkdirSync(extensionDir, { recursive: true });
    writeFileSync(path.join(extensionDir, "index.ts"), `const ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};\nexport default function register() {}\n`);
    writeFileSync(path.join(extensionDir, "runtime-guard.js"), "export function createBeforePromptBuildHandler() { return async () => ({}); }\n");
    writeFileSync(path.join(extensionDir, "package.json"), JSON.stringify({
        name: "openclawbrain",
        version: "0.3.5",
        type: "module",
        openclaw: {
            extensions: ["index.ts"]
        }
    }, null, 2));
    writeFileSync(path.join(extensionDir, "openclaw.plugin.json"), JSON.stringify({
        id: "openclawbrain",
        version: "0.3.5"
    }, null, 2));
    return extensionDir;
}
function createNativeInstall(openclawHome, activationRoot, installId = "openclaw") {
    const extensionDir = path.join(openclawHome, "extensions", installId);
    const loaderDir = path.join(extensionDir, "dist", "extension");
    const loaderPinnedActivationRoot = activationRoot === "__ACTIVATION_ROOT__"
        ? path.join(path.dirname(openclawHome), ".openclawbrain", "activation")
        : activationRoot;
    mkdirSync(loaderDir, { recursive: true });
    writeFileSync(path.join(extensionDir, "package.json"), JSON.stringify({
        name: "@openclawbrain/openclaw",
        version: "0.3.5",
        type: "module",
        openclaw: {
            extensions: ["./dist/extension/index.js"]
        }
    }, null, 2));
    writeFileSync(path.join(extensionDir, "openclaw.plugin.json"), JSON.stringify({
        id: "openclawbrain",
        version: "0.3.5"
    }, null, 2));
    writeFileSync(path.join(loaderDir, "runtime-guard.js"), `
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

export function createBeforePromptBuildHandler(input) {
  return async (event) => {
    if (!Array.isArray(event?.messages)) {
      await input.reportDiagnostic({
        key: "runtime-messages-not-array",
        message: "before_prompt_build event.messages is not an array"
      });
      return {};
    }
    return {};
  };
}

export function isActivationRootPlaceholder(value) {
  return value === "__ACTIVATION_ROOT__";
}

export function normalizePromptBuildEvent(event) {
  return event;
}

export function validateExtensionRegistrationApi(api) {
  return { ok: true, api };
}
`);
    writeFileSync(path.join(loaderDir, "index.js"), `
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createBeforePromptBuildHandler, validateExtensionRegistrationApi } from "./runtime-guard.js";

const ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};
const EXTENSION_ENTRY_PATH = fileURLToPath(import.meta.url);
const RESOLVED_ACTIVATION_ROOT = {
  activationRoot: ${JSON.stringify(activationRoot === "__ACTIVATION_ROOT__" ? path.join(path.dirname(openclawHome), ".openclawbrain", "activation") : activationRoot)},
  recoveredFromPlaceholder: ${JSON.stringify(activationRoot === "__ACTIVATION_ROOT__")}
};
const ACTIVATION_ROOT_PIN_PATTERN = /^const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";$/m;

async function reportDiagnostic(diagnostic) {
  console.warn(diagnostic.message);
  const logDir = path.join(homedir(), ".openclawbrain");
  mkdirSync(logDir, { recursive: true });
  appendFileSync(path.join(logDir, "extension-errors.log"), diagnostic.message + "\\n", "utf8");
}

function selfHealPinnedActivationRoot() {
  if (!RESOLVED_ACTIVATION_ROOT.recoveredFromPlaceholder) {
    return;
  }

  const loaderSource = readFileSync(EXTENSION_ENTRY_PATH, "utf8");
  if (!ACTIVATION_ROOT_PIN_PATTERN.test(loaderSource)) {
    return;
  }

  const nextLoaderSource = loaderSource.replace(
    ACTIVATION_ROOT_PIN_PATTERN,
    ${JSON.stringify(`const ACTIVATION_ROOT = ${JSON.stringify(loaderPinnedActivationRoot)};`)}
  );

  if (nextLoaderSource !== loaderSource) {
    writeFileSync(EXTENSION_ENTRY_PATH, nextLoaderSource, "utf8");
  }
}

export default function register(api) {
  const registration = validateExtensionRegistrationApi(api);
  if (!registration.ok) {
    return;
  }

  selfHealPinnedActivationRoot();
  registration.api.on(
    "before_prompt_build",
    createBeforePromptBuildHandler({
      activationRoot: RESOLVED_ACTIVATION_ROOT.activationRoot,
      compileRuntimeContext: () => ({ ok: true, brainContext: "" }),
      reportDiagnostic
    }),
    { priority: 5 }
  );
}
`);
    return {
        extensionDir,
        loaderEntryPath: path.join(loaderDir, "index.js")
    };
}
test("hook truth recognizes generated shadow installs", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    createShadowInstall(openclawHome, activationRoot);
    const inspection = inspectOpenClawBrainHookStatus(openclawHome);
    assert.equal(inspection.installState, "installed");
    assert.equal(inspection.loadability, "loadable");
    assert.equal(inspection.installLayout, "generated_shadow_extension");
    assert.equal(inspection.hookPath, path.join(openclawHome, "extensions", "openclawbrain", "index.ts"));
});
test("hook truth, runtime proofs, and activation-root discovery recognize native package installs", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root, ".openclaw-Tern", {
        profile: "Tern",
        plugins: {
            allow: ["openclawbrain"]
        }
    });
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, activationRoot);
    const inspection = inspectOpenClawBrainHookStatus(openclawHome);
    assert.equal(inspection.installState, "installed");
    assert.equal(inspection.loadability, "loadable");
    assert.equal(inspection.installLayout, "native_package_plugin");
    assert.equal(inspection.installId, "openclaw");
    assert.equal(inspection.packageName, "@openclawbrain/openclaw");
    assert.equal(inspection.hookPath, nativeInstall.loaderEntryPath);
    assert.match(inspection.detail, /manifest=openclawbrain/);
    assert.match(inspection.detail, /install=openclaw/);
    const record = recordOpenClawProfileRuntimeLoadProof({
        activationRoot,
        extensionEntryPath: nativeInstall.loaderEntryPath,
        loadedAt: "2026-03-18T12:00:00.000Z"
    });
    assert.equal(record.openclawHome, canonicalizePath(openclawHome));
    const proofs = listOpenClawProfileRuntimeLoadProofs(activationRoot);
    assert.equal(proofs.error, null);
    assert.equal(proofs.proofs?.profiles.length, 1);
    assert.equal(proofs.proofs?.profiles[0]?.openclawHome, canonicalizePath(openclawHome));
    assert.equal(resolveActivationRoot({ openclawHome }), activationRoot);
});
test("allowlist truth rejects the native package install hint as a loadable plugin id", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root, ".openclaw-Tern", {
        profile: "Tern",
        plugins: {
            allow: ["openclaw"]
        }
    });
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    createNativeInstall(openclawHome, activationRoot);
    const inspection = inspectOpenClawBrainHookStatus(openclawHome);
    assert.equal(inspection.installState, "blocked_by_allowlist");
    assert.equal(inspection.loadability, "blocked");
    assert.equal(inspection.pluginAllowlistState, "blocked");
    assert.match(inspection.detail, /plugins\.allow excludes recognized OpenClawBrain ids openclawbrain/);
});
test("plugin config normalization repairs split-package allowlist and entry-hint drift", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root, ".openclaw-Tern", {
        profile: "Tern",
        plugins: {
            allow: ["openclaw"],
            entries: {
                openclaw: {
                    enabled: true,
                    config: {
                        enabled: true
                    }
                }
            }
        }
    });
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    createNativeInstall(openclawHome, activationRoot);
    const selectedInstall = findInstalledOpenClawBrainPlugin(openclawHome).selectedInstall;
    assert.ok(selectedInstall);
    const normalized = normalizeOpenClawBrainPluginsConfig({
        allow: ["openclaw"],
        entries: {
            openclaw: {
                enabled: true,
                config: {
                    enabled: true
                }
            }
        }
    }, selectedInstall);
    assert.equal(normalized.changed, true);
    assert.deepEqual(normalized.pluginsConfig.allow, ["openclawbrain"]);
    assert.deepEqual(normalized.pluginsConfig.entries, {
        openclawbrain: {
            enabled: true,
            config: {
                enabled: true
            }
        }
    });
    assert.deepEqual(normalized.changes, [
        "removed plugins.allow entries openclaw",
        "added plugins.allow entries openclawbrain",
        "moved plugins.entries.openclaw to plugins.entries.openclawbrain"
    ]);
});
test("install target selection and activation-root pinning keep native package plugins authoritative", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root, ".openclaw", {
        profile: "Tern",
        plugins: {
            allow: ["openclawbrain"]
        }
    });
    const shadowActivationRoot = path.join(root, "shadow-activation");
    const pinnedActivationRoot = path.join(root, "pinned-activation");
    mkdirSync(shadowActivationRoot, { recursive: true });
    mkdirSync(pinnedActivationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, "__ACTIVATION_ROOT__");
    const shadowExtensionDir = createShadowInstall(openclawHome, shadowActivationRoot);
    const installTarget = resolveOpenClawBrainInstallTarget(openclawHome);
    assert.equal(installTarget.writeMode, "pin_native_package");
    assert.equal(installTarget.hookPath, nativeInstall.loaderEntryPath);
    assert.equal(installTarget.selectedInstall?.installLayout, "native_package_plugin");
    assert.equal(installTarget.additionalInstalls.length, 1);
    pinInstalledOpenClawBrainPluginActivationRoot(installTarget.hookPath, pinnedActivationRoot);
    assert.match(readFileSync(nativeInstall.loaderEntryPath, "utf8"), new RegExp(pinnedActivationRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.match(readFileSync(path.join(shadowExtensionDir, "index.ts"), "utf8"), new RegExp(shadowActivationRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.equal(resolveActivationRoot({ openclawHome }), pinnedActivationRoot);
    const inspection = inspectOpenClawBrainHookStatus(openclawHome);
    assert.equal(inspection.installLayout, "native_package_plugin");
    assert.equal(inspection.additionalInstallCount, 1);
    assert.equal(inspection.loadability, "loadable");
});
test("activation-root pinning treats an already-pinned native package loader as a successful no-op", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, activationRoot);
    const originalLoaderSource = readFileSync(nativeInstall.loaderEntryPath, "utf8");
    pinInstalledOpenClawBrainPluginActivationRoot(nativeInstall.loaderEntryPath, activationRoot);
    assert.equal(readFileSync(nativeInstall.loaderEntryPath, "utf8"), originalLoaderSource);
});
test("activation-root pinning still throws when the native package loader no longer exposes ACTIVATION_ROOT", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, activationRoot);
    writeFileSync(
        nativeInstall.loaderEntryPath,
        readFileSync(nativeInstall.loaderEntryPath, "utf8").replaceAll(
            `const ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};`,
            `const INSTALLED_ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};`
        ).replaceAll(
            'const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";',
            'const INSTALLED_ACTIVATION_ROOT = "__ACTIVATION_ROOT__";'
        ),
        "utf8"
    );
    assert.throws(() => pinInstalledOpenClawBrainPluginActivationRoot(nativeInstall.loaderEntryPath, activationRoot), /does not expose a patchable ACTIVATION_ROOT constant/);
});
test("installed extension inspection keeps native package layout truth even when a shadow copy also exists", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, activationRoot);
    createShadowInstall(openclawHome, path.join(root, "shadow-activation"));
    const inspection = inspectInstalledOpenClawBrainExtension(openclawHome);
    assert.equal(inspection.installLayout, "native_package_plugin");
    assert.equal(inspection.loaderEntryPath, nativeInstall.loaderEntryPath);
    assert.equal(inspection.runtimeGuardPath, path.join(nativeInstall.extensionDir, "dist", "extension", "runtime-guard.js"));
    assert.equal(inspection.additionalInstalls.length, 1);
});
test("native package plugin load proof succeeds and writes the expected diagnostic log", async (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    createNativeInstall(openclawHome, activationRoot);
    const previousHome = process.env.HOME;
    process.env.HOME = root;
    t.after(() => {
        if (previousHome === undefined) {
            delete process.env.HOME;
            return;
        }
        process.env.HOME = previousHome;
    });
    const proof = await proveInstalledOpenClawBrainExtensionLoad(openclawHome);
    assert.equal(proof.installLayout, "native_package_plugin");
    assert.equal(proof.registeredEventName, "before_prompt_build");
    assert.equal(proof.registeredPriority, 5);
    assert.equal(proof.probeResult && Object.keys(proof.probeResult).length, 0);
    assert.match(proof.probeWarning, /before_prompt_build event\.messages is not an array/);
    assert.match(proof.diagnosticLogContents, /before_prompt_build event\.messages is not an array/);
});

test("native package proof self-heals a placeholder-stranded refresh from the installed extension location", async (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, "__ACTIVATION_ROOT__");
    const packageJsonPath = path.join(nativeInstall.extensionDir, "package.json");
    const pluginManifestPath = path.join(nativeInstall.extensionDir, "openclaw.plugin.json");
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8"));
    packageJson.version = "0.3.6";
    writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
    writeFileSync(pluginManifestPath, JSON.stringify({ id: "openclawbrain", version: "0.3.6" }, null, 2));

    const inspection = inspectOpenClawBrainHookStatus(openclawHome);
    assert.equal(inspection.installLayout, "native_package_plugin");
    assert.equal(inspection.loadability, "loadable");
    assert.match(readFileSync(nativeInstall.loaderEntryPath, "utf8"), /const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";/);
    assert.equal(resolveActivationRoot({ openclawHome }), activationRoot);

    const installTarget = resolveOpenClawBrainInstallTarget(openclawHome);
    assert.equal(installTarget.writeMode, "pin_native_package");
    const proof = await proveInstalledOpenClawBrainExtensionLoad(openclawHome);
    assert.equal(proof.installLayout, "native_package_plugin");
    assert.equal(proof.registeredEventName, "before_prompt_build");
    const loaderLines = readFileSync(nativeInstall.loaderEntryPath, "utf8").split(/\r?\n/);
    const activationRootLine = loaderLines.find((line) => line.startsWith("const ACTIVATION_ROOT = ")) ?? null;
    assert.ok(activationRootLine);
    assert.match(activationRootLine, new RegExp(activationRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.notEqual(activationRootLine, 'const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";');

    pinInstalledOpenClawBrainPluginActivationRoot(installTarget.hookPath, activationRoot);
    const postPinInspection = inspectOpenClawBrainHookStatus(openclawHome);
    assert.equal(postPinInspection.loadability, "loadable");
    assert.equal(resolveActivationRoot({ openclawHome }), activationRoot);
    const postPinProof = await proveInstalledOpenClawBrainExtensionLoad(openclawHome);
    assert.equal(postPinProof.installLayout, "native_package_plugin");
    assert.equal(postPinProof.registeredEventName, "before_prompt_build");
});

test("native package self-heal stays idempotent across repeated register calls in one process", async (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, "__ACTIVATION_ROOT__");
    const moduleUrl = `${pathToFileURL(nativeInstall.loaderEntryPath).href}?idempotent=${Date.now()}`;
    const { default: register } = await import(moduleUrl);
    const registeredEvents = [];
    const api = {
        on(eventName, handler, options) {
            registeredEvents.push({ eventName, handler, options });
        }
    };
    register(api);
    register(api);
    assert.equal(registeredEvents.length, 2);
    const loaderSource = readFileSync(nativeInstall.loaderEntryPath, "utf8");
    const activationRootLine = loaderSource.split(/\r?\n/).find((line) => line.startsWith("const ACTIVATION_ROOT = ")) ?? null;
    assert.equal(activationRootLine, `const ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};`);
    assert.ok(loaderSource.includes('const ACTIVATION_ROOT_PIN_PATTERN = /^const ACTIVATION_ROOT = "__ACTIVATION_ROOT__";$/m;'));
    await import(`${pathToFileURL(nativeInstall.loaderEntryPath).href}?after-idempotent=${Date.now()}`);
});
