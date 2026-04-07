#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const MANIFEST_PATHS = [
  "package.json",
  "packages/cli/package.json",
  "packages/openclaw/package.json",
];

const EXACT_VERSION_RE = /^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z-.]+)?(?:\+[0-9A-Za-z-.]+)?$/;
const TRANSIENT_SPEC_RE = /^(?:git\+|git:|github:|gitlab:|bitbucket:|file:|link:|workspace:|https?:\/\/|ssh:\/\/)/i;
const PATH_SPEC_RE = /^(?:\.\.?\/|\/)/;

function usage() {
  process.stderr.write(
    [
      "Usage: node scripts/verify-dependency-policy.mjs [options]",
      "",
      "Options:",
      "  --repo-root <path>   Repository root to inspect (default: current working directory)",
      "  --json               Emit JSON only",
      "  --help               Show this help",
      "",
      "This guard fails closed on loose publishable dependency specs and on",
      "disallowed transient package patterns in publishable manifests.",
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

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function isExactVersionSpec(spec) {
  return EXACT_VERSION_RE.test(spec.trim());
}

function isTransientSpec(spec) {
  return TRANSIENT_SPEC_RE.test(spec.trim()) || PATH_SPEC_RE.test(spec.trim());
}

function inspectDependencyRecord({ manifestPath, manifest, section, exactRequired, blockers }) {
  const record = manifest?.[section];
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    return;
  }

  for (const [name, rawSpec] of Object.entries(record)) {
    if (typeof rawSpec !== "string" || rawSpec.trim().length === 0) {
      blockers.push({
        code: "invalid_dependency_spec",
        manifestPath,
        section,
        package: name,
        spec: rawSpec,
        detail: `missing or non-string spec for ${name} in ${section}`,
      });
      continue;
    }

    const spec = rawSpec.trim();
    if (isTransientSpec(spec)) {
      blockers.push({
        code: "transient_dependency_spec",
        manifestPath,
        section,
        package: name,
        spec,
        detail: `${section}.${name} uses a disallowed transient or path-based spec (${spec})`,
      });
      continue;
    }

    if (exactRequired && !isExactVersionSpec(spec)) {
      blockers.push({
        code: "loose_dependency_spec",
        manifestPath,
        section,
        package: name,
        spec,
        detail: `${section}.${name} must be an exact version, received ${spec}`,
      });
    }
  }
}

export function verifyDependencyPolicy(options = {}) {
  const repoRoot = path.resolve(options.repoRoot ?? process.cwd());
  const blockers = [];
  const scannedManifests = [];

  for (const relativePath of MANIFEST_PATHS) {
    const manifestPath = path.join(repoRoot, relativePath);
    if (!existsSync(manifestPath)) {
      blockers.push({
        code: "missing_manifest",
        manifestPath: relativePath,
        detail: `required publishable manifest is missing: ${relativePath}`,
      });
      continue;
    }

    const manifest = readJson(manifestPath);
    scannedManifests.push({
      manifestPath: relativePath,
      name: manifest.name ?? null,
      version: manifest.version ?? null,
    });

    inspectDependencyRecord({ manifestPath: relativePath, manifest, section: "dependencies", exactRequired: true, blockers });
    inspectDependencyRecord({ manifestPath: relativePath, manifest, section: "optionalDependencies", exactRequired: true, blockers });
    inspectDependencyRecord({ manifestPath: relativePath, manifest, section: "overrides", exactRequired: true, blockers });
    inspectDependencyRecord({ manifestPath: relativePath, manifest, section: "peerDependencies", exactRequired: false, blockers });
  }

  return {
    ok: blockers.length === 0,
    repoRoot,
    manifests: scannedManifests,
    blockers,
    message:
      blockers.length === 0
        ? "dependency policy is clean"
        : "dependency policy found loose or transient publishable specs",
  };
}

function formatResult(result) {
  const lines = [];
  lines.push(`dependency policy: ${result.ok ? "clean" : "blocked"}`);
  lines.push(`repo: ${result.repoRoot}`);
  lines.push(`manifests scanned: ${result.manifests.map((entry) => `${entry.manifestPath}@${entry.version ?? "unknown"}`).join(", ")}`);
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
  const result = verifyDependencyPolicy({ repoRoot: options.repoRoot });
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
