import { expect, test } from "vitest";
import { spawnSync } from "node:child_process";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "..");
const compatBinPath = path.join(repoRoot, "bin", "openclawbrain.js");

test("compatibility-package openclawbrain binary fails closed with the canonical recovery path", () => {
  const result = spawnSync(process.execPath, [compatBinPath, "install", "--openclaw-home", "~/.openclaw"], {
    cwd: repoRoot,
    encoding: "utf8",
    env: {
      ...process.env,
      OPENCLAWBRAIN_COMPAT_ALLOW_LEGACY_CLI: "",
    },
  });

  expect(result.status).toBe(1);
  expect(result.stderr).toContain("repo-root compatibility shim");
  expect(result.stderr).toContain("openclawbrain install --openclaw-home <path>");
  expect(result.stderr).toContain("openclawbrain proof --openclaw-home <path>");
});
