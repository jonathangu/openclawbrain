import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { inspectOpenClawBrainHookStatus } from "../src/openclaw-hook-truth.js";
import { listOpenClawProfileRuntimeLoadProofs, recordOpenClawProfileRuntimeLoadProof } from "../src/attachment-truth.js";
import { resolveActivationRoot } from "../src/resolve-activation-root.js";
import { inspectInstalledOpenClawBrainExtension, proveInstalledOpenClawBrainExtensionLoad } from "../src/shadow-extension-proof.js";
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
function createOpenClawHome(rootDir, name = ".openclaw") {
    const openclawHome = path.join(rootDir, name);
    mkdirSync(openclawHome, { recursive: true });
    writeFileSync(path.join(openclawHome, "openclaw.json"), JSON.stringify({
        profile: "Tern",
        plugins: {
            allow: ["openclawbrain"]
        }
    }, null, 2));
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
function createNativeInstall(openclawHome, activationRoot) {
    const extensionDir = path.join(openclawHome, "extensions", "@openclawbrain", "openclaw");
    const loaderDir = path.join(extensionDir, "dist", "extension");
    mkdirSync(loaderDir, { recursive: true });
    writeFileSync(path.join(extensionDir, "package.json"), JSON.stringify({
        name: "@openclawbrain/openclaw",
        version: "0.3.5",
        type: "module",
        openclaw: {
            extensions: ["dist/extension/index.js"]
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
    const openclawHome = createOpenClawHome(root, ".openclaw-Tern");
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, activationRoot);
    const inspection = inspectOpenClawBrainHookStatus(openclawHome);
    assert.equal(inspection.installState, "installed");
    assert.equal(inspection.loadability, "loadable");
    assert.equal(inspection.installLayout, "native_package_plugin");
    assert.equal(inspection.hookPath, nativeInstall.loaderEntryPath);
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
test("installed extension inspection recognizes native package plugin layout", (t) => {
    const root = createTempRoot(t);
    const openclawHome = createOpenClawHome(root);
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    mkdirSync(activationRoot, { recursive: true });
    const nativeInstall = createNativeInstall(openclawHome, activationRoot);
    const inspection = inspectInstalledOpenClawBrainExtension(openclawHome);
    assert.equal(inspection.installLayout, "native_package_plugin");
    assert.equal(inspection.loaderEntryPath, nativeInstall.loaderEntryPath);
    assert.equal(inspection.runtimeGuardPath, path.join(nativeInstall.extensionDir, "dist", "extension", "runtime-guard.js"));
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
