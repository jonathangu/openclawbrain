import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
// @ts-ignore runtime script module has no checked-in typed surface yet
import { DEFAULT_MAX_AGE_DAYS, REQUIRED_ASSERTION_KEYS, REQUIRED_PROOF_FILES, verifyProofSmoke } from "../scripts/verify-proof-smoke.mjs";

const createdDirs: string[] = [];

afterEach(() => {
  while (createdDirs.length > 0) {
    rmSync(createdDirs.pop() as string, { recursive: true, force: true });
  }
});

function makeTempRepo(): string {
  const root = mkdtempSync(path.join(tmpdir(), "openclaw-proof-smoke-"));
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

function writeClaimDocs(root: string): void {
  writeText(
    root,
    "docs/RELEASE_CONTRACT.md",
    [
      "# Release Contract",
      "",
      "- operationally validated: yes",
    ].join("\n"),
  );
  writeText(
    root,
    "docs/EVIDENCE.md",
    [
      "# Evidence",
      "",
      "- Level 4: frozen",
    ].join("\n"),
  );
  writeText(
    root,
    "CLAIMS.md",
    [
      "# Claims",
      "",
      "- durable proof bundles",
    ].join("\n"),
  );
}

function writeProofBundle(root: string, options?: { date?: string; sha?: string; omitFile?: string; missingAssertion?: string }) {
  const date = options?.date ?? "2026-03-16";
  const sha = options?.sha ?? "4ccd71a22418b9170128b8d948f5a95801a10380";
  const bundleRoot = path.join("docs", "evidence", date, sha);
  const omitFile = options?.omitFile ?? null;
  const assertions = Object.fromEntries(REQUIRED_ASSERTION_KEYS.map((key: string) => [key, {}]));

  Object.assign(assertions.teachRetrieval, {
    retrievedCorrectionVisible: true,
    traceIncludesTaughtNode: true,
  });
  Object.assign(assertions.workerDownFailOpen, {
    servedBeforeCrash: true,
    servedAfterCrash: true,
    servedPullRequestGuidanceBeforeCrash: true,
    servedPullRequestGuidance: true,
  });
  Object.assign(assertions.recurrentQuery, {
    aborted: false,
    currentPackVersion: 12,
  });
  Object.assign(assertions.shortLookup, {
    aborted: false,
    bypassEvidence: "lookup bypass captured",
  });
  Object.assign(assertions.shadowMode, {
    shadowMode: true,
    injectedContextVisible: false,
    aborted: false,
  });
  Object.assign(assertions.noEmbedding, {
    aborted: false,
  });
  Object.assign(assertions.uninitialized, {
    aborted: false,
  });

  if (options?.missingAssertion) {
    delete assertions[options.missingAssertion as keyof typeof assertions];
  }

  for (const file of REQUIRED_PROOF_FILES) {
    if (file === omitFile) {
      continue;
    }
    if (file.endsWith(".json")) {
      switch (file) {
        case "validation-report.json":
          writeJson(root, path.join(bundleRoot, file), {
            gitSha: sha,
            assertions,
          });
          break;
        case "status.json":
        case "doctor.json":
          writeJson(root, path.join(bundleRoot, file), { ok: true });
          break;
        case "trace.json":
          writeJson(root, path.join(bundleRoot, file), {
            ok: true,
            parsed: { trace: { id: "bt_fixture" } },
          });
          break;
        default:
          writeJson(root, path.join(bundleRoot, file), { ok: true });
          break;
      }
      continue;
    }

    if (file === "summary.md") {
      writeText(root, path.join(bundleRoot, file), `# Summary\n\n- commit: \`${sha}\`\n`);
      continue;
    }

    writeText(root, path.join(bundleRoot, file), `${file}\n`);
  }

  return { date, sha, bundleRoot };
}

describe("verifyProofSmoke", () => {
  it("passes when a fresh checked-in proof bundle matches the active claim surface", () => {
    const repoRoot = makeTempRepo();
    writeClaimDocs(repoRoot);
    writeProofBundle(repoRoot);

    const result = verifyProofSmoke({
      repoRoot,
      now: new Date("2026-03-26T00:00:00.000Z"),
      maxAgeDays: DEFAULT_MAX_AGE_DAYS,
    });

    expect(result.ok).toBe(true);
    expect(result.enforced).toBe(true);
    expect(result.bundle?.relativeDir).toBe("docs/evidence/2026-03-16/4ccd71a22418b9170128b8d948f5a95801a10380");
  });

  it("fails when the bundle is stale for the configured smoke window", () => {
    const repoRoot = makeTempRepo();
    writeClaimDocs(repoRoot);
    writeProofBundle(repoRoot, { date: "2026-02-01" });

    const result = verifyProofSmoke({
      repoRoot,
      now: new Date("2026-03-26T00:00:00.000Z"),
      maxAgeDays: 14,
    });

    expect(result.ok).toBe(false);
    expect(result.enforced).toBe(true);
    expect(result.failures?.[0]?.problems.some((problem: any) => problem.code === "stale_bundle")).toBe(true);
  });

  it("fails when the fresh bundle is missing required proof files", () => {
    const repoRoot = makeTempRepo();
    writeClaimDocs(repoRoot);
    writeProofBundle(repoRoot, { omitFile: "trace.json" });

    const result = verifyProofSmoke({
      repoRoot,
      now: new Date("2026-03-26T00:00:00.000Z"),
    });

    expect(result.ok).toBe(false);
    expect(result.failures?.[0]?.problems.some((problem: any) => problem.code === "missing_required_files")).toBe(true);
  });

  it("fails when the fresh bundle no longer carries the required assertion set", () => {
    const repoRoot = makeTempRepo();
    writeClaimDocs(repoRoot);
    writeProofBundle(repoRoot, { missingAssertion: "shadowMode" });

    const result = verifyProofSmoke({
      repoRoot,
      now: new Date("2026-03-26T00:00:00.000Z"),
    });

    expect(result.ok).toBe(false);
    expect(result.failures?.[0]?.problems.some((problem: any) => problem.code === "validation_report_missing_assertion_keys")).toBe(true);
  });

  it("does not enforce the gate when the repo no longer advertises a proof-freshness claim", () => {
    const repoRoot = makeTempRepo();
    writeText(repoRoot, "docs/RELEASE_CONTRACT.md", "# Release Contract\n");
    writeText(repoRoot, "docs/EVIDENCE.md", "# Evidence\n");
    writeText(repoRoot, "CLAIMS.md", "# Claims\n");

    const result = verifyProofSmoke({
      repoRoot,
      now: new Date("2026-03-26T00:00:00.000Z"),
    });

    expect(result.ok).toBe(true);
    expect(result.enforced).toBe(false);
  });
});
