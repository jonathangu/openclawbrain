#!/usr/bin/env node

import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(repoRoot, relativePath), "utf8"));
}

function runGit(args) {
  return execFileSync("git", args, {
    cwd: repoRoot,
    encoding: "utf8"
  }).trim();
}

function maybeRunGit(args) {
  try {
    return runGit(args);
  } catch {
    return null;
  }
}

const workspacePackage = readJson("package.json");
const packagePaths = readdirSync(path.join(repoRoot, "packages"), { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => path.join("packages", entry.name, "package.json"));
const packageVersions = packagePaths.map((packagePath) => {
  const packageJson = readJson(packagePath);
  return {
    name: packageJson.name,
    version: packageJson.version
  };
});

const workspaceVersion = workspacePackage.version;
const expectedTag = `v${workspaceVersion}`;
const headSha = runGit(["rev-parse", "HEAD"]);
const headShortSha = runGit(["rev-parse", "--short=7", "HEAD"]);
const branch = maybeRunGit(["branch", "--show-current"]);
const tagsOnHead = runGit(["tag", "--points-at", "HEAD"])
  .split("\n")
  .map((tag) => tag.trim())
  .filter(Boolean);
const matchingTagOnHead = tagsOnHead.includes(expectedTag);
const dirtyEntries = runGit(["status", "--porcelain"])
  .split("\n")
  .map((line) => line.trimEnd())
  .filter(Boolean);
const releaseDir = path.join(repoRoot, ".release");
const tarballs = existsSync(releaseDir)
  ? readdirSync(releaseDir)
      .filter((file) => file.endsWith(".tgz"))
      .sort()
  : [];
const versionMismatches = packageVersions.filter((pkg) => pkg.version !== workspaceVersion);

const shipSurface = matchingTagOnHead ? "tagged-release-candidate" : "repo-tip";
const publishState = matchingTagOnHead ? "awaiting-publish-verification" : "not-tagged-for-publish";

const result = {
  shipSurface,
  publishState,
  workspaceVersion,
  expectedTag,
  git: {
    branch,
    headSha,
    headShortSha,
    matchingTagOnHead,
    tagsOnHead,
    dirty: dirtyEntries.length > 0,
    dirtyEntries
  },
  packages: {
    count: packageVersions.length,
    allMatchWorkspaceVersion: versionMismatches.length === 0,
    versionMismatches,
    versions: packageVersions
  },
  releaseArtifacts: {
    directory: ".release",
    tarballCount: tarballs.length,
    tarballs
  },
  operatorNotes: matchingTagOnHead
    ? [
        `HEAD carries ${expectedTag}; treat this as a tagged release candidate until post-publish verification completes.`,
        "Run pnpm release:check before pushing or trusting package publication.",
        "Confirm npm registry versions separately before claiming public package shipment."
      ]
    : [
        `HEAD ${headShortSha} does not carry ${expectedTag}; the truthful ship surface for this wave is the repo tip, not npm publication.`,
        "Use pnpm release:pack if you need installable local tarballs for outside-consumer smoke tests.",
        `Cut and push ${expectedTag} only in the same change that actually starts the publish lane.`
      ]
};

console.log(JSON.stringify(result, null, 2));
