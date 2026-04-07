import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { verifyReleaseDocsDrift } from "../scripts/verify-release-docs-drift.mjs";

const createdDirs: string[] = [];
const CURRENT_VERSION = "0.4.36";

afterEach(() => {
  while (createdDirs.length > 0) {
    rmSync(createdDirs.pop() as string, { recursive: true, force: true });
  }
});

function makeTempRepo(): string {
  const root = mkdtempSync(path.join(tmpdir(), "openclaw-release-docs-drift-"));
  createdDirs.push(root);
  return root;
}

function writeText(root: string, relativePath: string, contents: string): void {
  const filePath = path.join(root, relativePath);
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, contents, "utf8");
}

function writeReleaseScaffold(
  root: string,
  options?: { readmeVersion?: string; docsIndexVersion?: string; endStateVersion?: string; teacherV3LintVersion?: string },
) {
  const readmeVersion = options?.readmeVersion ?? CURRENT_VERSION;
  const docsIndexVersion = options?.docsIndexVersion ?? CURRENT_VERSION;
  const endStateVersion = options?.endStateVersion ?? CURRENT_VERSION;
  const teacherV3LintVersion = options?.teacherV3LintVersion ?? CURRENT_VERSION;

  writeText(
    root,
    "CHANGELOG.md",
    [
      "# Changelog",
      "",
      "## Unreleased",
      "",
      `## ${CURRENT_VERSION}`,
      "",
      `- [docs/release-notes-${CURRENT_VERSION}.md](docs/release-notes-${CURRENT_VERSION}.md)`,
    ].join("\n"),
  );
  writeText(
    root,
    "README.md",
    [
      "# OpenClawBrain",
      "",
      `Current version: **${readmeVersion}** · [Changelog](CHANGELOG.md) · [Claims boundary](CLAIMS.md)`,
    ].join("\n"),
  );
  writeText(
    root,
    "docs/README.md",
    [
      "# OpenClawBrain documentation",
      "",
      "Release history:",
      `- [Current release notes (${docsIndexVersion})](release-notes-${docsIndexVersion}.md)`,
      "- [Full changelog](../CHANGELOG.md)",
    ].join("\n"),
  );
  writeText(
    root,
    `docs/release-notes-${CURRENT_VERSION}.md`,
    [
      `# OpenClawBrain ${CURRENT_VERSION}`,
      "",
      "Canonical install lane:",
      "",
      "openclawbrain install --openclaw-home ~/.openclaw",
    ].join("\n"),
  );
  writeText(
    root,
    "docs/END_STATE.md",
    [
      "# OpenClawBrain v2 — End-State Guide",
      "",
      "## Current repo reality",
      "",
      "### Already true",
      `- split packages \`@openclawbrain/openclaw@${endStateVersion}\` and \`@openclawbrain/cli@${endStateVersion}\` are published`,
      `- split packages \`@openclawbrain/openclaw@${endStateVersion}\` and \`@openclawbrain/cli@${endStateVersion}\` are published`,
    ].join("\n"),
  );
  writeText(
    root,
    "docs/architecture/teacher-v3-lints.md",
    [
      "# Teacher v3 lint families",
      "",
      "## 3) Release-drift motivating case",
      "",
      "The repo now lines up on the current release surface:",
      "",
      `- the root \`README.md\` says \`Current version: ${teacherV3LintVersion}\``,
      `- \`docs/README.md\` points the release-history index at \`Current release notes (${teacherV3LintVersion})\``,
      `- \`docs/END_STATE.md\` keeps its current split-package truth on \`${teacherV3LintVersion}\``,
      `- \`docs/release-notes-${teacherV3LintVersion}.md\` exists and describes \`${teacherV3LintVersion}\``,
      "",
      "## 4) Placeholder checklist for an eventual runner",
    ].join("\n"),
  );
}

describe("verifyReleaseDocsDrift", () => {
  it("passes when public docs surfaces match the current changelog release", () => {
    const repoRoot = makeTempRepo();
    writeReleaseScaffold(repoRoot);

    const result = verifyReleaseDocsDrift({ repoRoot });

    expect(result.ok).toBe(true);
    expect(result.currentVersion).toBe(CURRENT_VERSION);
    expect(result.readmeVersion).toBe(CURRENT_VERSION);
    expect(result.docsIndexVersion).toBe(CURRENT_VERSION);
    expect(result.docsIndexTargetVersion).toBe(CURRENT_VERSION);
    expect(result.endStateVersions).toEqual([
      [CURRENT_VERSION, CURRENT_VERSION],
      [CURRENT_VERSION, CURRENT_VERSION],
    ]);
    expect(result.teacherV3LintsReleaseSurfaceVersions).toEqual([
      CURRENT_VERSION,
      CURRENT_VERSION,
      CURRENT_VERSION,
      CURRENT_VERSION,
      CURRENT_VERSION,
    ]);
    expect(result.blockers).toEqual([]);
  });

  it("fails when README.md, docs/README.md, and docs/END_STATE.md still point at stale release surfaces", () => {
    const repoRoot = makeTempRepo();
    writeReleaseScaffold(repoRoot, {
      readmeVersion: "0.4.26",
      docsIndexVersion: "0.4.24",
      endStateVersion: "0.4.24",
      teacherV3LintVersion: "0.4.24",
    });

    const result = verifyReleaseDocsDrift({ repoRoot });

    expect(result.ok).toBe(false);
    expect(result.blockers.map((blocker: any) => blocker.code)).toEqual(
      expect.arrayContaining(["readme_version_mismatch", "docs_index_release_notes_mismatch"]),
    );
    expect(result.readmeVersion).toBe("0.4.26");
    expect(result.docsIndexVersion).toBe("0.4.24");
    expect(result.docsIndexTargetVersion).toBe("0.4.24");
    expect(result.endStateVersions).toEqual([["0.4.24", "0.4.24"], ["0.4.24", "0.4.24"]]);
    expect(result.teacherV3LintsReleaseSurfaceVersions).toEqual([
      "0.4.24",
      "0.4.24",
      "0.4.24",
      "0.4.24",
      "0.4.24",
    ]);
    expect(result.blockers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "end_state_split_package_version_mismatch" }),
        expect.objectContaining({ code: "teacher_v3_lints_release_surface_mismatch" }),
      ]),
    );
  });
});
