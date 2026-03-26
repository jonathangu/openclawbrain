#!/usr/bin/env node

import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

export const DEFAULT_MAX_AGE_DAYS = 21;

export const REQUIRED_PROOF_FILES = [
  "summary.md",
  "validation-report.json",
  "status.json",
  "doctor.json",
  "config-snapshot.json",
  "logs.txt",
  "trace.json",
  "status-all.txt",
  "gateway-probe.txt",
  "gateway-status.txt",
  "channels-status.txt",
];

export const REQUIRED_ASSERTION_KEYS = [
  "teachRetrieval",
  "workerDownFailOpen",
  "recurrentQuery",
  "shortLookup",
  "shadowMode",
  "noEmbedding",
  "uninitialized",
];

const DAY_MS = 24 * 60 * 60 * 1000;

const CLAIM_SIGNAL_MATCHERS = [
  {
    code: "release_contract_operationally_validated",
    file: "docs/RELEASE_CONTRACT.md",
    pattern: /operationally validated\W+yes/i,
    detail: "release contract still claims the repo is operationally validated",
  },
  {
    code: "evidence_level4_frozen",
    file: "docs/EVIDENCE.md",
    pattern: /Level 4:\s*.*frozen/i,
    detail: "evidence ladder still marks the public host/operator proof lane as frozen",
  },
  {
    code: "claims_durable_proof_bundles",
    file: "CLAIMS.md",
    pattern: /durable proof bundles/i,
    detail: "claims boundary still advertises durable proof bundles on the operator surfaces",
  },
];

function usage() {
  process.stderr.write(
    [
      "Usage: node scripts/verify-proof-smoke.mjs [options]",
      "",
      "Options:",
      "  --repo-root <path>       Repository root to inspect (default: current working directory)",
      `  --max-age-days <days>    Maximum proof age in days (default: ${DEFAULT_MAX_AGE_DAYS})`,
      "  --now <iso>              Override current time for deterministic checks",
      "  --json                   Emit JSON only",
      "  --help                   Show this help",
      "",
      "This smoke gate enforces a fresh checked-in proof bundle only when the repo's",
      "public docs still claim a frozen/operator-validated proof boundary.",
    ].join("\n") + "\n",
  );
}

function parseArgs(argv) {
  const options = {
    repoRoot: process.cwd(),
    maxAgeDays: DEFAULT_MAX_AGE_DAYS,
    now: new Date(),
    json: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--repo-root":
        options.repoRoot = path.resolve(argv[++index] ?? "");
        break;
      case "--max-age-days":
        options.maxAgeDays = Number.parseInt(argv[++index] ?? "", 10);
        break;
      case "--now":
        options.now = new Date(argv[++index] ?? "");
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

  if (!Number.isFinite(options.maxAgeDays) || options.maxAgeDays < 0) {
    throw new Error(`--max-age-days must be a non-negative integer, received ${options.maxAgeDays}`);
  }
  if (!(options.now instanceof Date) || Number.isNaN(options.now.getTime())) {
    throw new Error("--now must be a valid ISO timestamp");
  }

  return options;
}

function readText(filePath) {
  return readFileSync(filePath, "utf8");
}

function tryReadJson(filePath, label, problems) {
  try {
    return JSON.parse(readText(filePath));
  } catch (error) {
    problems.push({
      code: `${label}_invalid_json`,
      detail: `${label} is unreadable JSON: ${(error instanceof Error ? error.message : String(error))}`,
    });
    return null;
  }
}

function toPosixPath(filePath) {
  return filePath.split(path.sep).join("/");
}

function roundAgeDays(value) {
  return Math.round(value * 100) / 100;
}

function parseDateDirectory(name) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(name)) {
    return null;
  }
  const parsed = new Date(`${name}T00:00:00.000Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function detectProofClaimSignals(repoRoot) {
  const signals = [];
  for (const matcher of CLAIM_SIGNAL_MATCHERS) {
    const filePath = path.join(repoRoot, matcher.file);
    if (!existsSync(filePath)) {
      continue;
    }
    const text = readText(filePath);
    if (matcher.pattern.test(text)) {
      signals.push({
        code: matcher.code,
        file: matcher.file,
        detail: matcher.detail,
      });
    }
  }
  return signals;
}

export function collectEvidenceBundles(repoRoot) {
  const evidenceRoot = path.join(repoRoot, "docs", "evidence");
  if (!existsSync(evidenceRoot)) {
    return [];
  }

  const bundles = [];
  for (const dateEntry of readdirSync(evidenceRoot, { withFileTypes: true })) {
    if (!dateEntry.isDirectory()) {
      continue;
    }
    const observedAt = parseDateDirectory(dateEntry.name);
    if (observedAt === null) {
      continue;
    }
    const dateDir = path.join(evidenceRoot, dateEntry.name);
    for (const shaEntry of readdirSync(dateDir, { withFileTypes: true })) {
      if (!shaEntry.isDirectory() || !/^[a-f0-9]{40}$/i.test(shaEntry.name)) {
        continue;
      }
      const dir = path.join(dateDir, shaEntry.name);
      bundles.push({
        date: dateEntry.name,
        sha: shaEntry.name,
        dir,
        relativeDir: toPosixPath(path.relative(repoRoot, dir)),
        observedAt,
      });
    }
  }

  bundles.sort((left, right) => {
    const dateDiff = right.observedAt.getTime() - left.observedAt.getTime();
    if (dateDiff !== 0) {
      return dateDiff;
    }
    return left.sha.localeCompare(right.sha);
  });

  return bundles;
}

function validateAssertionShape(assertions, problems) {
  if (!assertions || typeof assertions !== "object") {
    problems.push({
      code: "validation_report_missing_assertions",
      detail: "validation-report.json does not expose an assertions object",
    });
    return;
  }

  const missingAssertionKeys = REQUIRED_ASSERTION_KEYS.filter((key) => !(key in assertions));
  if (missingAssertionKeys.length > 0) {
    problems.push({
      code: "validation_report_missing_assertion_keys",
      detail: `validation-report.json is missing required assertion keys: ${missingAssertionKeys.join(", ")}`,
    });
    return;
  }

  const teachRetrieval = assertions.teachRetrieval ?? {};
  if (teachRetrieval.retrievedCorrectionVisible !== true || teachRetrieval.traceIncludesTaughtNode !== true) {
    problems.push({
      code: "teach_retrieval_not_proven",
      detail: "teachRetrieval must show a retrieved correction and a matching trace node",
    });
  }

  const workerDownFailOpen = assertions.workerDownFailOpen ?? {};
  if (
    workerDownFailOpen.servedBeforeCrash !== true ||
    workerDownFailOpen.servedAfterCrash !== true ||
    workerDownFailOpen.servedPullRequestGuidanceBeforeCrash !== true ||
    workerDownFailOpen.servedPullRequestGuidance !== true
  ) {
    problems.push({
      code: "worker_down_fail_open_not_proven",
      detail: "workerDownFailOpen must show serving before and after the crash with the expected guidance visible",
    });
  }

  const recurrentQuery = assertions.recurrentQuery ?? {};
  if (recurrentQuery.aborted === true || typeof recurrentQuery.currentPackVersion !== "number") {
    problems.push({
      code: "recurrent_query_not_proven",
      detail: "recurrentQuery must complete without aborting and report a current pack version",
    });
  }

  const shortLookup = assertions.shortLookup ?? {};
  if (shortLookup.aborted === true || typeof shortLookup.bypassEvidence !== "string" || shortLookup.bypassEvidence.trim().length === 0) {
    problems.push({
      code: "short_lookup_not_proven",
      detail: "shortLookup must complete without aborting and record non-empty bypass evidence",
    });
  }

  const shadowMode = assertions.shadowMode ?? {};
  if (shadowMode.shadowMode !== true || shadowMode.injectedContextVisible !== false || shadowMode.aborted === true) {
    problems.push({
      code: "shadow_mode_not_proven",
      detail: "shadowMode must stay in shadow and keep injected context invisible",
    });
  }

  if ((assertions.noEmbedding ?? {}).aborted === true) {
    problems.push({
      code: "no_embedding_not_proven",
      detail: "noEmbedding must complete without aborting",
    });
  }

  if ((assertions.uninitialized ?? {}).aborted === true) {
    problems.push({
      code: "uninitialized_not_proven",
      detail: "uninitialized must complete without aborting",
    });
  }
}

export function validateEvidenceBundle(bundle, options) {
  const problems = [];
  const ageDays = roundAgeDays((options.now.getTime() - bundle.observedAt.getTime()) / DAY_MS);

  if (ageDays > options.maxAgeDays) {
    problems.push({
      code: "stale_bundle",
      detail: `${bundle.relativeDir} is ${ageDays} days old, above the ${options.maxAgeDays}-day smoke limit`,
    });
  }

  const missingFiles = REQUIRED_PROOF_FILES.filter((file) => !existsSync(path.join(bundle.dir, file)));
  if (missingFiles.length > 0) {
    problems.push({
      code: "missing_required_files",
      detail: `${bundle.relativeDir} is missing required proof files: ${missingFiles.join(", ")}`,
    });
  }

  const summaryPath = path.join(bundle.dir, "summary.md");
  if (existsSync(summaryPath)) {
    const summary = readText(summaryPath);
    if (!summary.includes(`- commit: \`${bundle.sha}\``)) {
      problems.push({
        code: "summary_commit_mismatch",
        detail: `summary.md does not pin commit ${bundle.sha}`,
      });
    }
  }

  const validationReportPath = path.join(bundle.dir, "validation-report.json");
  const validationReport = existsSync(validationReportPath)
    ? tryReadJson(validationReportPath, "validation_report", problems)
    : null;
  if (validationReport !== null) {
    if (validationReport.gitSha !== bundle.sha) {
      problems.push({
        code: "validation_report_git_sha_mismatch",
        detail: `validation-report.json gitSha ${validationReport.gitSha ?? "missing"} does not match ${bundle.sha}`,
      });
    }
    validateAssertionShape(validationReport.assertions, problems);
  }

  const statusPath = path.join(bundle.dir, "status.json");
  const status = existsSync(statusPath) ? tryReadJson(statusPath, "status", problems) : null;
  if (status !== null && status.ok !== true) {
    problems.push({
      code: "status_not_ok",
      detail: "status.json must report ok=true",
    });
  }

  const doctorPath = path.join(bundle.dir, "doctor.json");
  const doctor = existsSync(doctorPath) ? tryReadJson(doctorPath, "doctor", problems) : null;
  if (doctor !== null && doctor.ok !== true) {
    problems.push({
      code: "doctor_not_ok",
      detail: "doctor.json must report ok=true",
    });
  }

  const tracePath = path.join(bundle.dir, "trace.json");
  const trace = existsSync(tracePath) ? tryReadJson(tracePath, "trace", problems) : null;
  if (trace !== null) {
    if (trace.ok !== true) {
      problems.push({
        code: "trace_not_ok",
        detail: "trace.json must report ok=true",
      });
    }
    if (!trace.parsed?.trace?.id) {
      problems.push({
        code: "trace_missing_trace_id",
        detail: "trace.json must expose a parsed trace id",
      });
    }
  }

  return {
    ok: problems.length === 0,
    bundle: {
      date: bundle.date,
      sha: bundle.sha,
      relativeDir: bundle.relativeDir,
      ageDays,
    },
    problems,
  };
}

export function verifyProofSmoke(options = {}) {
  const repoRoot = path.resolve(options.repoRoot ?? process.cwd());
  const now = options.now instanceof Date ? options.now : new Date(options.now ?? Date.now());
  const maxAgeDays = options.maxAgeDays ?? DEFAULT_MAX_AGE_DAYS;

  if (Number.isNaN(now.getTime())) {
    throw new Error("verifyProofSmoke received an invalid current time");
  }

  const claimSignals = detectProofClaimSignals(repoRoot);
  if (claimSignals.length === 0) {
    return {
      ok: true,
      enforced: false,
      repoRoot,
      maxAgeDays,
      now: now.toISOString(),
      claimSignals: [],
      bundlesChecked: 0,
      message: "no active public proof-freshness claim detected; smoke gate not enforced",
    };
  }

  const bundles = collectEvidenceBundles(repoRoot);
  const validations = bundles.map((bundle) => validateEvidenceBundle(bundle, { now, maxAgeDays }));
  const passingBundle = validations.find((result) => result.ok);

  if (passingBundle) {
    return {
      ok: true,
      enforced: true,
      repoRoot,
      maxAgeDays,
      now: now.toISOString(),
      claimSignals,
      bundlesChecked: validations.length,
      bundle: passingBundle.bundle,
      message: `fresh proof bundle found at ${passingBundle.bundle.relativeDir}`,
    };
  }

  return {
    ok: false,
    enforced: true,
    repoRoot,
    maxAgeDays,
    now: now.toISOString(),
    claimSignals,
    bundlesChecked: validations.length,
    failures: validations.slice(0, 5),
    message: "no fresh, complete proof bundle satisfies the current public proof claim",
  };
}

function formatResult(result) {
  const lines = [];
  lines.push(`proof smoke: ${result.ok ? "ok" : "failed"}`);
  if (result.enforced === false) {
    lines.push(result.message);
    return lines.join("\n");
  }

  lines.push(`claims: ${result.claimSignals.map((signal) => signal.code).join(", ")}`);
  lines.push(`checked bundles: ${result.bundlesChecked}`);

  if (result.ok) {
    lines.push(`bundle: ${result.bundle.relativeDir}`);
    lines.push(`age days: ${result.bundle.ageDays} (limit ${result.maxAgeDays})`);
  } else {
    lines.push(result.message);
    for (const failure of result.failures ?? []) {
      lines.push(`- ${failure.bundle.relativeDir}`);
      for (const problem of failure.problems) {
        lines.push(`  - ${problem.code}: ${problem.detail}`);
      }
    }
  }

  return lines.join("\n");
}

export function runCli(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const result = verifyProofSmoke(options);
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
