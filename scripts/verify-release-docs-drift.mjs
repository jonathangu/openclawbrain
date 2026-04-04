#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

function usage() {
  process.stderr.write(
    [
      "Usage: node scripts/verify-release-docs-drift.mjs [options]",
      "",
      "Options:",
      "  --repo-root <path>   Repository root to inspect (default: current working directory)",
      "  --json               Emit JSON only",
      "  --help               Show this help",
      "",
      "This deterministic lint compares the current release version in CHANGELOG.md",
      "against public release-surfaces in README.md and docs/README.md.",
    ].join("\n") + "\n",
  );
}

function parseArgs(argv) {
  const options = {
    repoRoot: process.cwd(),
    json: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--repo-root":
        options.repoRoot = path.resolve(argv[++index] ?? "");
        break;
      case "--json":
        options.json = true;
        break;
      case "--help":
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function readText(filePath) {
  return readFileSync(filePath, "utf8");
}

function escapeForRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function detectCurrentReleaseVersion(changelogText) {
  for (const line of changelogText.split(/\r?\n/)) {
    const match = /^##\s+((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\s*$/.exec(line);
    if (match) {
      return match[1];
    }
  }

  return null;
}

function readVersionMatch(text, pattern) {
  const match = pattern.exec(text);
  if (!match) {
    return null;
  }

  return match.slice(1);
}

export function verifyReleaseDocsDrift(options = {}) {
  const repoRoot = path.resolve(options.repoRoot ?? process.cwd());
  const blockers = [];

  const changelogPath = path.join(repoRoot, "CHANGELOG.md");
  if (!existsSync(changelogPath)) {
    blockers.push({
      code: "missing_changelog",
      detail: "CHANGELOG.md is required to determine the current release version",
    });

    return {
      ok: false,
      repoRoot,
      currentVersion: null,
      readmeVersion: null,
      docsIndexVersion: null,
      releaseNotesFile: null,
      blockers,
      message: "release/docs drift lint is blocked until the current release version can be determined",
    };
  }

  const changelogText = readText(changelogPath);
  const currentVersion = detectCurrentReleaseVersion(changelogText);
  if (currentVersion === null) {
    blockers.push({
      code: "missing_current_release_heading",
      detail: "CHANGELOG.md does not contain a current semver release heading",
    });
  }

  const readmePath = path.join(repoRoot, "README.md");
  if (!existsSync(readmePath)) {
    blockers.push({
      code: "missing_readme",
      detail: "README.md is required for the public version surface",
    });
  }

  const docsReadmePath = path.join(repoRoot, "docs", "README.md");
  if (!existsSync(docsReadmePath)) {
    blockers.push({
      code: "missing_docs_index",
      detail: "docs/README.md is required for the release-history index surface",
    });
  }

  const releaseNotesFile = currentVersion === null ? null : `docs/release-notes-${currentVersion}.md`;
  const releaseNotesPath = releaseNotesFile === null ? null : path.join(repoRoot, releaseNotesFile);
  if (releaseNotesPath !== null && !existsSync(releaseNotesPath)) {
    blockers.push({
      code: "missing_release_notes",
      detail: `${releaseNotesFile} is required for the current release`,
    });
  }

  let readmeVersion = null;
  if (currentVersion !== null && existsSync(readmePath)) {
    const readmeText = readText(readmePath);
    const match = readVersionMatch(
      readmeText,
      /Current version:\s+\*\*((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\*\*/m,
    );
    readmeVersion = match?.[0] ?? null;

    if (readmeVersion === null) {
      blockers.push({
        code: "readme_version_missing",
        detail: "README.md must advertise the current version in the public version banner",
      });
    } else if (readmeVersion !== currentVersion) {
      blockers.push({
        code: "readme_version_mismatch",
        detail: `README.md advertises ${readmeVersion} but CHANGELOG.md says the current release is ${currentVersion}`,
      });
    }
  }

  let docsIndexVersion = null;
  let docsIndexTargetVersion = null;
  if (currentVersion !== null && existsSync(docsReadmePath)) {
    const docsReadmeText = readText(docsReadmePath);
    const match = readVersionMatch(
      docsReadmeText,
      /-\s+\[Current release notes \(((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\)\]\(release-notes-((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))\.md\)/m,
    );

    docsIndexVersion = match?.[0] ?? null;
    docsIndexTargetVersion = match?.[1] ?? null;

    if (docsIndexVersion === null || docsIndexTargetVersion === null) {
      blockers.push({
        code: "docs_index_missing_current_release_notes_link",
        detail: "docs/README.md must point at the current release notes link in the release-history index",
      });
    } else if (docsIndexVersion !== currentVersion || docsIndexTargetVersion !== currentVersion) {
      blockers.push({
        code: "docs_index_release_notes_mismatch",
        detail: `docs/README.md still points at ${docsIndexVersion} / ${docsIndexTargetVersion} instead of ${currentVersion}`,
      });
    }
  }

  if (currentVersion !== null) {
    const expectedReleaseNotesLink = `[docs/release-notes-${currentVersion}.md](docs/release-notes-${currentVersion}.md)`;
    if (!changelogText.includes(expectedReleaseNotesLink)) {
      blockers.push({
        code: "changelog_missing_current_release_notes_link",
        detail: `CHANGELOG.md must link to ${expectedReleaseNotesLink}`,
      });
    }
  }

  if (releaseNotesPath !== null && existsSync(releaseNotesPath)) {
    const releaseNotesText = readText(releaseNotesPath);
    const expectedHeading = new RegExp(`^#\\s+OpenClawBrain\\s+${escapeForRegex(currentVersion)}\\s*$`, "m");
    if (!expectedHeading.test(releaseNotesText)) {
      blockers.push({
        code: "release_notes_heading_mismatch",
        detail: `${releaseNotesFile} must headline OpenClawBrain ${currentVersion}`,
      });
    }
  }

  return {
    ok: blockers.length === 0,
    repoRoot,
    currentVersion,
    readmeVersion,
    docsIndexVersion,
    docsIndexTargetVersion,
    releaseNotesFile,
    blockers,
    message:
      blockers.length === 0
        ? `release/docs drift lint is clean for ${currentVersion}`
        : "release/docs drift lint found public-surface version drift",
  };
}

function formatResult(result) {
  const lines = [];
  lines.push(`release/docs drift lint: ${result.ok ? "clean" : "blocked"}`);
  lines.push(`repo: ${result.repoRoot}`);
  lines.push(`current release: ${result.currentVersion ?? "unknown"}`);
  lines.push(`README.md version: ${result.readmeVersion ?? "missing"}`);
  lines.push(`docs/README.md release notes: ${result.docsIndexVersion ?? "missing"} -> ${result.docsIndexTargetVersion ?? "missing"}`);
  lines.push(`release notes file: ${result.releaseNotesFile ?? "missing"}`);

  if (result.blockers.length > 0) {
    lines.push("blockers:");
    for (const blocker of result.blockers) {
      lines.push(`- ${blocker.code}: ${blocker.detail}`);
    }
  }

  return lines.join("\n");
}

export function runCli(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const result = verifyReleaseDocsDrift({ repoRoot: options.repoRoot });

  process.stdout.write(options.json ? `${JSON.stringify(result, null, 2)}\n` : `${formatResult(result)}\n`);
  if (!result.ok) {
    process.exitCode = 1;
  }
}

const isMainModule = process.argv[1]
  ? pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url
  : false;

if (isMainModule) {
  try {
    runCli();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  }
}
