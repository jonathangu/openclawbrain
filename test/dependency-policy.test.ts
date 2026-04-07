import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { verifyDependencyPolicy } from "../scripts/verify-dependency-policy.mjs";

const createdDirs: string[] = [];

afterEach(() => {
  while (createdDirs.length > 0) {
    rmSync(createdDirs.pop() as string, { recursive: true, force: true });
  }
});

function makeTempRepo(): string {
  const root = mkdtempSync(path.join(tmpdir(), "openclaw-dep-policy-"));
  createdDirs.push(root);
  return root;
}

function writeText(root: string, relativePath: string, contents: string): void {
  const filePath = path.join(root, relativePath);
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, contents, "utf8");
}

function writeJson(root: string, relativePath: string, value: unknown): void {
  writeText(root, relativePath, `${JSON.stringify(value, null, 2)}\n`);
}

function writePublishableScaffold(root: string): void {
  writeJson(root, "package.json", {
    name: "@jonathangu/openclawbrain",
    version: "0.3.8",
    dependencies: {
      "@mariozechner/pi-agent-core": "0.53.0",
      "@mariozechner/pi-ai": "0.53.0",
      "@sinclair/typebox": "0.34.48",
      tsx: "4.21.0",
    },
    peerDependencies: {
      openclaw: "*",
    },
  });
  writeJson(root, "packages/cli/package.json", {
    name: "@openclawbrain/cli",
    version: "0.4.35",
    dependencies: {
      "@openclawbrain/compiler": "0.3.5",
      "@openclawbrain/contracts": "0.3.5",
      "@openclawbrain/events": "0.3.4",
    },
  });
  writeJson(root, "packages/openclaw/package.json", {
    name: "@openclawbrain/openclaw",
    version: "0.4.35",
    dependencies: {
      "@openclawbrain/compiler": "0.3.5",
      "@openclawbrain/contracts": "0.3.5",
      "@openclawbrain/learner": "0.3.4",
      "@openclawbrain/pack-format": "0.3.4",
    },
  });
}

describe("verifyDependencyPolicy", () => {
  it("passes when publishable manifests use exact dependency pins and the host peer stays loose", () => {
    const repoRoot = makeTempRepo();
    writePublishableScaffold(repoRoot);

    const result = verifyDependencyPolicy({ repoRoot });

    expect(result.ok).toBe(true);
    expect(result.blockers).toEqual([]);
  });

  it("fails when publishable dependency specs are loose or transient", () => {
    const repoRoot = makeTempRepo();
    writePublishableScaffold(repoRoot);
    writeJson(repoRoot, "packages/cli/package.json", {
      name: "@openclawbrain/cli",
      version: "0.4.35",
      dependencies: {
        "@openclawbrain/compiler": "^0.3.5",
        "@openclawbrain/contracts": "0.3.5",
      },
      optionalDependencies: {
        "transient-pkg": "git+https://example.com/acme/transient-pkg.git",
      },
    });

    const result = verifyDependencyPolicy({ repoRoot });

    expect(result.ok).toBe(false);
    expect(result.blockers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "loose_dependency_spec", package: "@openclawbrain/compiler" }),
        expect.objectContaining({ code: "transient_dependency_spec", package: "transient-pkg" }),
      ]),
    );
  });

  it("flags transient peer or override specs where practical", () => {
    const repoRoot = makeTempRepo();
    writePublishableScaffold(repoRoot);
    writeJson(repoRoot, "package.json", {
      name: "@jonathangu/openclawbrain",
      version: "0.3.8",
      dependencies: {
        "@mariozechner/pi-agent-core": "0.53.0",
        "@mariozechner/pi-ai": "0.53.0",
        "@sinclair/typebox": "0.34.48",
        tsx: "4.21.0",
      },
      peerDependencies: {
        openclaw: "*",
        "other-peer": "file:../other-peer",
      },
      overrides: {
        "left-pad": "github:example/left-pad#main",
      },
    });

    const result = verifyDependencyPolicy({ repoRoot });

    expect(result.ok).toBe(false);
    expect(result.blockers.map((blocker: any) => blocker.code)).toEqual(
      expect.arrayContaining(["transient_dependency_spec", "transient_dependency_spec"]),
    );
  });
});
