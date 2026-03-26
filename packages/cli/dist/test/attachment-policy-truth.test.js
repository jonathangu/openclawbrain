import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  loadAttachmentPolicyDeclaration,
  resolveEffectiveAttachmentPolicyTruth,
  resolveAttachmentPolicyDeclarationPath,
  writeAttachmentPolicyDeclaration,
} from "../src/attachment-policy-truth.js";

function createTempActivationRoot(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-attachment-policy-"));
  const activationRoot = path.join(root, "activation-root");
  mkdirSync(activationRoot, { recursive: true });
  t.after(() => {
    rmSync(root, { recursive: true, force: true });
  });
  return activationRoot;
}

test("attachment policy declaration round-trips shared install truth", (t) => {
  const activationRoot = createTempActivationRoot(t);
  const written = writeAttachmentPolicyDeclaration({
    activationRoot,
    policy: "shared",
    source: "install",
    openclawHome: "/tmp/.openclaw-shared",
    updatedAt: "2026-03-20T19:40:00.000Z",
  });

  assert.equal(written.path, resolveAttachmentPolicyDeclarationPath(activationRoot));
  assert.equal(written.declaration.policy, "shared");
  assert.equal(written.declaration.source, "install");
  assert.equal(written.declaration.openclawHome, "/tmp/.openclaw-shared");

  const loaded = loadAttachmentPolicyDeclaration(activationRoot);
  assert.equal(loaded.path, written.path);
  assert.equal(loaded.error, null);
  assert.equal(loaded.declaration?.policy, "shared");
  assert.equal(loaded.declaration?.source, "install");
  assert.equal(loaded.declaration?.openclawHome, "/tmp/.openclaw-shared");
});

test("attachment policy declaration fails open when missing", (t) => {
  const activationRoot = createTempActivationRoot(t);
  const loaded = loadAttachmentPolicyDeclaration(activationRoot);
  assert.equal(loaded.declaration, null);
  assert.equal(loaded.error, null);
});

test("attachment policy declaration fails open when malformed", (t) => {
  const activationRoot = createTempActivationRoot(t);
  const declarationPath = resolveAttachmentPolicyDeclarationPath(activationRoot);
  mkdirSync(path.dirname(declarationPath), { recursive: true });
  writeFileSync(declarationPath, "{not-json", "utf8");

  const loaded = loadAttachmentPolicyDeclaration(activationRoot);
  assert.equal(loaded.declaration, null);
  assert.match(loaded.error ?? "", /Expected property name|JSON|Unexpected token/);
});

test("effective attachment policy keeps declared shared truth when status underreports", () => {
  const resolved = resolveEffectiveAttachmentPolicyTruth({
    statusPolicy: null,
    reportPolicy: "undeclared",
    declaredPolicy: "shared",
    referenceCount: 1,
  });

  assert.equal(resolved.effectivePolicy, "shared");
  assert.equal(resolved.statusPolicy, "shared");
  assert.equal(resolved.reportPolicy, "shared");
});

test("effective attachment policy lets discoverable multi-home reality win", () => {
  const resolved = resolveEffectiveAttachmentPolicyTruth({
    statusPolicy: "dedicated",
    reportPolicy: "dedicated",
    declaredPolicy: "dedicated",
    referenceCount: 2,
  });

  assert.equal(resolved.effectivePolicy, "shared");
  assert.equal(resolved.statusPolicy, "shared");
  assert.equal(resolved.reportPolicy, "shared");
});
