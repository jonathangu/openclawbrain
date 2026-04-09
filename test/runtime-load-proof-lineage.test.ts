import { afterEach, expect, it } from "vitest";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  listOpenClawProfileRuntimeLoadProofs as listCliRuntimeLoadProofs,
} from "../packages/cli/dist/src/attachment-truth.js";
import {
  listOpenClawProfileRuntimeLoadProofs as listOpenClawRuntimeLoadProofs,
  recordOpenClawProfileRuntimeLoadProof,
} from "../packages/openclaw/dist/src/attachment-truth.js";

const tempRoots = new Set<string>();

afterEach(() => {
  for (const root of tempRoots) {
    rmSync(root, { recursive: true, force: true });
  }
  tempRoots.clear();
});

function createTempRoot(): string {
  const root = path.join(os.tmpdir(), `openclawbrain-lineage-proof-${Math.random().toString(16).slice(2)}`);
  mkdirSync(root, { recursive: true });
  tempRoots.add(root);
  return root;
}

it("runtime load proof records package lineage from the installed extension entry path", () => {
  const root = createTempRoot();
  const packageRoot = path.join(root, ".openclaw", "extensions", "openclawbrain");
  const extensionEntryPath = path.join(packageRoot, "dist", "extension", "index.js");
  const activationRoot = path.join(root, "activation-root");

  mkdirSync(path.dirname(extensionEntryPath), { recursive: true });
  mkdirSync(path.join(activationRoot, "attachment-truth"), { recursive: true });
  writeFileSync(
    path.join(packageRoot, "package.json"),
    JSON.stringify({ name: "@openclawbrain/openclaw", version: "0.4.40" }, null, 2),
    "utf8",
  );
  writeFileSync(extensionEntryPath, "export {};\n", "utf8");

  const written = recordOpenClawProfileRuntimeLoadProof({
    activationRoot,
    extensionEntryPath,
    loadedAt: "2026-04-09T15:00:00.000Z",
  });

  expect(written.packageName).toBe("@openclawbrain/openclaw");
  expect(written.packageVersion).toBe("0.4.40");
  expect(written.packageIdentity).toBe("@openclawbrain/openclaw@0.4.40");
  expect(written.packageJsonPath ?? "").toMatch(/\.openclaw\/extensions\/openclawbrain\/package\.json$/);

  const openclawSnapshot = listOpenClawRuntimeLoadProofs(activationRoot);
  expect(openclawSnapshot.error).toBeNull();
  expect(openclawSnapshot.proofs?.profiles[0].packageIdentity).toBe("@openclawbrain/openclaw@0.4.40");

  const cliSnapshot = listCliRuntimeLoadProofs(activationRoot);
  expect(cliSnapshot.error).toBeNull();
  expect(cliSnapshot.proofs?.profiles[0].packageName).toBe("@openclawbrain/openclaw");
  expect(cliSnapshot.proofs?.profiles[0].packageVersion).toBe("0.4.40");
  expect(cliSnapshot.proofs?.profiles[0].packageIdentity).toBe("@openclawbrain/openclaw@0.4.40");
});
