import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { buildReleasePlan, verifyReleasePlan } from "../scripts/release-plan.mjs";

const createdDirs: string[] = [];

afterEach(() => {
  while (createdDirs.length > 0) {
    rmSync(createdDirs.pop() as string, { recursive: true, force: true });
  }
});

function makeTempRepo(): string {
  const root = mkdtempSync(path.join(tmpdir(), "openclaw-release-plan-"));
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

function writeReleaseScaffold(root: string): void {
  writeJson(root, "packages/openclaw/package.json", {
    name: "@openclawbrain/openclaw",
    version: "0.4.4",
  });
  writeJson(root, "packages/cli/package.json", {
    name: "@openclawbrain/cli",
    version: "0.4.15",
  });
  writeText(root, ".changeset/README.md", "# Changesets\n");
  writeText(
    root,
    "docs/release-notes-0.4.15.md",
    [
      "# OpenClawBrain 0.4.15",
      "",
      "Canonical install lane:",
      "",
      "openclawbrain install --openclaw-home ~/.openclaw",
    ].join("\n"),
  );
  writeText(
    root,
    "CHANGELOG.md",
    [
      "# Changelog",
      "",
      "## 0.4.15",
      "",
      "- [docs/release-notes-0.4.15.md](docs/release-notes-0.4.15.md)",
    ].join("\n"),
  );
}

const MAINLINE_GIT_FACTS = {
  headSha: "4ccd71a22418b9170128b8d948f5a95801a10380",
  branch: null,
  mainlineRef: "origin/main",
  onMainline: true,
  tagExists: false,
  gitProblem: null,
};

describe("buildReleasePlan", () => {
  it("derives canonical public release metadata from the package versions", () => {
    const repoRoot = makeTempRepo();
    writeReleaseScaffold(repoRoot);

    const plan = buildReleasePlan(repoRoot);

    expect(plan.openclawVersion).toBe("0.4.4");
    expect(plan.cliVersion).toBe("0.4.15");
    expect(plan.publicVersion).toBe("0.4.15");
    expect(plan.tag).toBe("openclawbrain-v0.4.15");
    expect(plan.releaseNotesFile).toBe("docs/release-notes-0.4.15.md");
    expect(plan.title).toBe("OpenClawBrain 0.4.15");
  });
});

describe("verifyReleasePlan", () => {
  it("passes when the ref, changelog, and release notes all match the split publish contract", () => {
    const repoRoot = makeTempRepo();
    writeReleaseScaffold(repoRoot);

    const result = verifyReleasePlan({
      repoRoot,
      gitFacts: MAINLINE_GIT_FACTS,
    });

    expect(result.ok).toBe(true);
    expect(result.blockers).toEqual([]);
  });

  it("fails when pending changesets still exist on the selected publish ref", () => {
    const repoRoot = makeTempRepo();
    writeReleaseScaffold(repoRoot);
    writeText(repoRoot, ".changeset/late-fix.md", "---\n---\nlate fix\n");

    const result = verifyReleasePlan({
      repoRoot,
      gitFacts: MAINLINE_GIT_FACTS,
    });

    expect(result.ok).toBe(false);
    expect(result.blockers.some((blocker: any) => blocker.code === "pending_changesets")).toBe(true);
  });

  it("fails when the versioned release note file is missing", () => {
    const repoRoot = makeTempRepo();
    writeJson(repoRoot, "packages/openclaw/package.json", {
      name: "@openclawbrain/openclaw",
      version: "0.4.4",
    });
    writeJson(repoRoot, "packages/cli/package.json", {
      name: "@openclawbrain/cli",
      version: "0.4.15",
    });
    writeText(repoRoot, ".changeset/README.md", "# Changesets\n");
    writeText(
      repoRoot,
      "CHANGELOG.md",
      [
        "# Changelog",
        "",
        "## 0.4.15",
        "",
        "- [docs/release-notes-0.4.15.md](docs/release-notes-0.4.15.md)",
      ].join("\n"),
    );

    const result = verifyReleasePlan({
      repoRoot,
      gitFacts: MAINLINE_GIT_FACTS,
    });

    expect(result.ok).toBe(false);
    expect(result.blockers.some((blocker: any) => blocker.code === "missing_release_notes")).toBe(true);
  });

  it("fails when the changelog does not carry the current split release heading", () => {
    const repoRoot = makeTempRepo();
    writeReleaseScaffold(repoRoot);
    writeText(repoRoot, "CHANGELOG.md", "# Changelog\n");

    const result = verifyReleasePlan({
      repoRoot,
      gitFacts: MAINLINE_GIT_FACTS,
    });

    expect(result.ok).toBe(false);
    expect(result.blockers.some((blocker: any) => blocker.code === "missing_changelog_heading")).toBe(true);
  });

  it("fails when the selected ref is not reachable from main", () => {
    const repoRoot = makeTempRepo();
    writeReleaseScaffold(repoRoot);

    const result = verifyReleasePlan({
      repoRoot,
      gitFacts: {
        ...MAINLINE_GIT_FACTS,
        onMainline: false,
      },
    });

    expect(result.ok).toBe(false);
    expect(result.blockers.some((blocker: any) => blocker.code === "release_ref_not_on_mainline")).toBe(true);
  });
});
