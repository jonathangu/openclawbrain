#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

export const SPLIT_PACKAGE_SPECS = [
  {
    key: "openclaw",
    name: "@openclawbrain/openclaw",
    packageJsonPath: "packages/openclaw/package.json",
  },
  {
    key: "cli",
    name: "@openclawbrain/cli",
    packageJsonPath: "packages/cli/package.json",
  },
];

function usage() {
  process.stderr.write(
    [
      "Usage: node scripts/release-plan.mjs [options]",
      "",
      "Options:",
      "  --repo-root <path>         Repository root to inspect (default: current working directory)",
      "  --json                     Emit JSON only",
      "  --github-output <path>     Write GitHub Actions step outputs to the given file",
      "  --write-summary <path>     Write a Markdown release summary to the given file",
      "  --help                     Show this help",
      "",
      "This preflight derives the canonical split-package release metadata and fails",
      "closed when the selected ref or docs no longer match the publish contract.",
    ].join("\n") + "\n",
  );
}

function parseArgs(argv) {
  const options = {
    repoRoot: process.cwd(),
    json: false,
    githubOutputPath: null,
    summaryPath: null,
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
      case "--github-output":
        options.githubOutputPath = path.resolve(argv[++index] ?? "");
        break;
      case "--write-summary":
        options.summaryPath = path.resolve(argv[++index] ?? "");
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

function readJson(filePath) {
  return JSON.parse(readText(filePath));
}

function escapeForRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function gitExec(repoRoot, args) {
  return execFileSync("git", args, {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
}

function gitTryExec(repoRoot, args) {
  try {
    return {
      ok: true,
      stdout: gitExec(repoRoot, args),
      code: 0,
    };
  } catch (error) {
    return {
      ok: false,
      stdout: "",
      code: error && typeof error === "object" && "status" in error ? error.status : null,
      error,
    };
  }
}

function gitRefExists(repoRoot, ref) {
  return gitTryExec(repoRoot, ["rev-parse", "--verify", "--quiet", `${ref}^{commit}`]).ok;
}

function gitIsAncestor(repoRoot, olderRef, newerRef) {
  const result = gitTryExec(repoRoot, ["merge-base", "--is-ancestor", olderRef, newerRef]);
  if (result.ok) {
    return true;
  }
  if (result.code === 1) {
    return false;
  }
  throw result.error;
}

export function buildReleasePlan(repoRoot = process.cwd()) {
  const resolvedRepoRoot = path.resolve(repoRoot);
  const versions = {};

  for (const spec of SPLIT_PACKAGE_SPECS) {
    const packageJsonPath = path.join(resolvedRepoRoot, spec.packageJsonPath);
    const packageJson = readJson(packageJsonPath);
    if (typeof packageJson.version !== "string" || packageJson.version.trim().length === 0) {
      throw new Error(`${spec.packageJsonPath} is missing a valid version`);
    }
    versions[spec.key] = packageJson.version;
  }

  const openclawVersion = versions.openclaw;
  const cliVersion = versions.cli;

  return {
    packages: SPLIT_PACKAGE_SPECS.map((spec) => ({
      key: spec.key,
      name: spec.name,
      version: versions[spec.key],
      packageJsonPath: spec.packageJsonPath,
    })),
    openclawVersion,
    cliVersion,
    tag: `split-openclaw-v${openclawVersion}-cli-v${cliVersion}`,
    title: `OpenClawBrain split release: openclaw ${openclawVersion} / cli ${cliVersion}`,
    changelogHeading: `${cliVersion} / ${openclawVersion}`,
    releaseNotesFile: `docs/release-notes-${cliVersion}.md`,
  };
}

export function listPendingChangesets(repoRoot = process.cwd()) {
  const changesetDir = path.join(path.resolve(repoRoot), ".changeset");
  if (!existsSync(changesetDir)) {
    return [];
  }

  return readdirSync(changesetDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".md") && entry.name !== "README.md")
    .map((entry) => `.changeset/${entry.name}`)
    .sort();
}

export function detectGitFacts(repoRoot = process.cwd(), releasePlan = buildReleasePlan(repoRoot)) {
  const resolvedRepoRoot = path.resolve(repoRoot);
  const headShaResult = gitTryExec(resolvedRepoRoot, ["rev-parse", "HEAD"]);
  const branchResult = gitTryExec(resolvedRepoRoot, ["branch", "--show-current"]);
  const mainlineRef = gitRefExists(resolvedRepoRoot, "origin/main")
    ? "origin/main"
    : gitRefExists(resolvedRepoRoot, "main")
      ? "main"
      : null;

  let onMainline = null;
  let gitProblem = null;
  if (mainlineRef !== null) {
    try {
      onMainline = gitIsAncestor(resolvedRepoRoot, "HEAD", mainlineRef);
    } catch (error) {
      gitProblem = error instanceof Error ? error.message : String(error);
    }
  }

  return {
    headSha: headShaResult.ok ? headShaResult.stdout : null,
    branch: branchResult.ok && branchResult.stdout.length > 0 ? branchResult.stdout : null,
    mainlineRef,
    onMainline,
    tagExists: gitRefExists(resolvedRepoRoot, `refs/tags/${releasePlan.tag}`),
    gitProblem,
  };
}

function verifyReleaseNotes(repoRoot, releasePlan, blockers) {
  const releaseNotesPath = path.join(repoRoot, releasePlan.releaseNotesFile);
  if (!existsSync(releaseNotesPath)) {
    blockers.push({
      code: "missing_release_notes",
      detail: `${releasePlan.releaseNotesFile} is required for the GitHub release body`,
    });
    return;
  }

  const releaseNotes = readText(releaseNotesPath);
  const headingPattern = new RegExp(
    `^#\\s+OpenClawBrain\\s+${escapeForRegex(releasePlan.cliVersion)}\\s*/\\s*${escapeForRegex(releasePlan.openclawVersion)}\\s+split-package release notes\\s*$`,
    "m",
  );

  if (!headingPattern.test(releaseNotes)) {
    blockers.push({
      code: "release_notes_heading_mismatch",
      detail: `${releasePlan.releaseNotesFile} must headline ${releasePlan.cliVersion} / ${releasePlan.openclawVersion}`,
    });
  }

  const expectedPackageVersions = [
    `@openclawbrain/openclaw@${releasePlan.openclawVersion}`,
    `@openclawbrain/cli@${releasePlan.cliVersion}`,
  ];
  const missingPackageVersions = expectedPackageVersions.filter((entry) => !releaseNotes.includes(entry));
  if (missingPackageVersions.length > 0) {
    blockers.push({
      code: "release_notes_version_mismatch",
      detail: `${releasePlan.releaseNotesFile} is missing package version lines: ${missingPackageVersions.join(", ")}`,
    });
  }
}

function verifyChangelog(repoRoot, releasePlan, blockers) {
  const changelogPath = path.join(repoRoot, "CHANGELOG.md");
  if (!existsSync(changelogPath)) {
    blockers.push({
      code: "missing_changelog",
      detail: "CHANGELOG.md is required for the split release flow",
    });
    return;
  }

  const changelog = readText(changelogPath);
  if (!new RegExp(`^##\\s+${escapeForRegex(releasePlan.changelogHeading)}\\s*$`, "m").test(changelog)) {
    blockers.push({
      code: "missing_changelog_heading",
      detail: `CHANGELOG.md is missing the release heading ${releasePlan.changelogHeading}`,
    });
  }

  const releaseNotesLink = `[docs/release-notes-${releasePlan.cliVersion}.md](docs/release-notes-${releasePlan.cliVersion}.md)`;
  if (!changelog.includes(releaseNotesLink)) {
    blockers.push({
      code: "missing_release_notes_link",
      detail: `CHANGELOG.md must link to ${releasePlan.releaseNotesFile}`,
    });
  }
}

export function verifyReleasePlan(options = {}) {
  const repoRoot = path.resolve(options.repoRoot ?? process.cwd());
  const release = buildReleasePlan(repoRoot);
  const git = options.gitFacts ?? detectGitFacts(repoRoot, release);
  const pendingChangesets = listPendingChangesets(repoRoot);
  const blockers = [];

  if (git.mainlineRef === null) {
    blockers.push({
      code: "missing_mainline_ref",
      detail: "could not resolve origin/main or main; publish only from mainline release commits",
    });
  } else if (git.onMainline !== true) {
    blockers.push({
      code: "release_ref_not_on_mainline",
      detail: `HEAD ${git.headSha ?? "(unknown)"} is not reachable from ${git.mainlineRef}; publish the merged release commit on main`,
    });
  }

  if (git.gitProblem) {
    blockers.push({
      code: "git_state_unavailable",
      detail: git.gitProblem,
    });
  }

  if (pendingChangesets.length > 0) {
    blockers.push({
      code: "pending_changesets",
      detail: `pending changesets still exist on the publish ref: ${pendingChangesets.join(", ")}`,
    });
  }

  if (git.tagExists) {
    blockers.push({
      code: "release_tag_exists",
      detail: `git tag ${release.tag} already exists`,
    });
  }

  verifyReleaseNotes(repoRoot, release, blockers);
  verifyChangelog(repoRoot, release, blockers);

  return {
    ok: blockers.length === 0,
    repoRoot,
    release,
    git,
    pendingChangesets,
    blockers,
    message:
      blockers.length === 0
        ? `publish plan is ready for ${release.tag}`
        : "publish plan is blocked until the release contract matches the selected ref",
  };
}

function appendGitHubOutput(outputPath, key, value) {
  writeFileSync(outputPath, `${key}<<__OPENCLAW_RELEASE_PLAN__\n${value}\n__OPENCLAW_RELEASE_PLAN__\n`, {
    encoding: "utf8",
    flag: "a",
  });
}

function writeGitHubOutputs(outputPath, result) {
  const outputs = {
    ok: result.ok ? "true" : "false",
    openclaw_version: result.release.openclawVersion,
    cli_version: result.release.cliVersion,
    tag: result.release.tag,
    title: result.release.title,
    release_notes_file: result.release.releaseNotesFile,
    changelog_heading: result.release.changelogHeading,
    head_sha: result.git.headSha ?? "",
    mainline_ref: result.git.mainlineRef ?? "",
    pending_changesets: result.pendingChangesets.join(","),
  };

  for (const [key, value] of Object.entries(outputs)) {
    appendGitHubOutput(outputPath, key, value);
  }
}

function renderSummary(result) {
  const lines = [];
  lines.push("## Publish Plan");
  lines.push("");
  lines.push(`- Result: ${result.ok ? "ready" : "blocked"}`);
  lines.push(`- Ref: \`${result.git.headSha ?? "unknown"}\``);
  lines.push(`- Mainline ref: \`${result.git.mainlineRef ?? "missing"}\``);
  lines.push(`- Tag: \`${result.release.tag}\``);
  lines.push(`- Title: \`${result.release.title}\``);
  lines.push(`- Release notes: \`${result.release.releaseNotesFile}\``);
  lines.push(`- Publish order: \`@openclawbrain/openclaw@${result.release.openclawVersion}\`, then \`@openclawbrain/cli@${result.release.cliVersion}\``);

  if (result.pendingChangesets.length > 0) {
    lines.push(`- Pending changesets: ${result.pendingChangesets.map((entry) => `\`${entry}\``).join(", ")}`);
  }

  if (result.blockers.length > 0) {
    lines.push("");
    lines.push("## Blockers");
    lines.push("");
    for (const blocker of result.blockers) {
      lines.push(`- \`${blocker.code}\`: ${blocker.detail}`);
    }
  }

  return `${lines.join("\n")}\n`;
}

function formatResult(result) {
  const lines = [];
  lines.push(`publish plan: ${result.ok ? "ready" : "blocked"}`);
  lines.push(`ref: ${result.git.headSha ?? "unknown"}`);
  lines.push(`mainline: ${result.git.mainlineRef ?? "missing"}${result.git.onMainline === true ? " (contains HEAD)" : ""}`);
  lines.push(`tag: ${result.release.tag}`);
  lines.push(`title: ${result.release.title}`);
  lines.push(`release notes: ${result.release.releaseNotesFile}`);
  lines.push(`publish order: @openclawbrain/openclaw@${result.release.openclawVersion} -> @openclawbrain/cli@${result.release.cliVersion}`);

  if (result.pendingChangesets.length > 0) {
    lines.push(`pending changesets: ${result.pendingChangesets.join(", ")}`);
  }

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
  const result = verifyReleasePlan({ repoRoot: options.repoRoot });

  if (options.githubOutputPath) {
    writeGitHubOutputs(options.githubOutputPath, result);
  }
  if (options.summaryPath) {
    writeFileSync(options.summaryPath, renderSummary(result), "utf8");
  }

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
