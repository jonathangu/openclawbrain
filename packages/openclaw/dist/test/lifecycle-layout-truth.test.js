import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { inspectOpenClawBrainHookStatus } from "../src/openclaw-hook-truth.js";
import { listOpenClawProfileRuntimeLoadProofs, recordOpenClawProfileRuntimeLoadProof } from "../src/attachment-truth.js";
import { resolveActivationRoot } from "../src/resolve-activation-root.js";
import { inspectInstalledOpenClawBrainExtension, proveInstalledOpenClawBrainExtensionLoad } from "../src/shadow-extension-proof.js";
import { pinInstalledOpenClawBrainPluginActivationRoot, resolveOpenClawBrainInstallTarget } from "../src/openclaw-plugin-install.js";
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
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { createBeforePromptBuildHandler, validateExtensionRegistrationApi } from "./runtime-guard.js";

const ACTIVATION_ROOT = ${JSON.stringify(activationRoot)};

async function reportDiagnostic(diagnostic) {
  console.warn(diagnostic.message);
  const logDir = path.join(homedir(), ".openclawbrain");
  mkdirSync(logDir, { recursive: true });
  appendFileSync(path.join(logDir, "extension-errors.log"), diagnostic.message + "\\n", "utf8");
}

export default function register(api) {
  const registration = validateExtensionRegistrationApi(api);
  if (!registration.ok) {
    return;
  }
  registration.api.on(
    "before_prompt_build",
    createBeforePromptBuildHandler({
      activationRoot: ACTIVATION_ROOT,
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
            allow: ["openclaw"]
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
test("install target selection and activation-root pinning keep native package plugins authoritative", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root, ".openclaw", {
        profile: "Tern",
        plugins: {
            allow: ["openclaw"]
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
    writeFileSync(nativeInstall.loaderEntryPath, readFileSync(nativeInstall.loaderEntryPath, "utf8").replace("const ACTIVATION_ROOT", "const INSTALLED_ACTIVATION_ROOT"), "utf8");
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
