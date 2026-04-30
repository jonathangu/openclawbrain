import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";

export interface ReproducibilityMetadata {
  schema_version: "ocb.reproducibility.v1";
  generated_at: string;
  run_id: string;
  node_version: string;
  platform: NodeJS.Platform;
  arch: string;
  git_commit: string | null;
  git_branch: string | null;
  git_dirty: boolean | null;
  command: string[];
  trace_file_sha256: string;
  fixture_file_sha256: string;
  eval_mode: "smoke" | "production";
  external_mutation_allowed: false;
}

export async function captureReproducibilityMetadata(options: {
  runId: string;
  tracePath: string;
  fixturesPath: string;
  mode: "smoke" | "production";
  command: string[];
}): Promise<ReproducibilityMetadata> {
  return {
    schema_version: "ocb.reproducibility.v1",
    generated_at: new Date().toISOString(),
    run_id: options.runId,
    node_version: process.version,
    platform: process.platform,
    arch: process.arch,
    git_commit: gitOrNull(["rev-parse", "HEAD"]),
    git_branch: gitOrNull(["branch", "--show-current"]),
    git_dirty: gitDirty(),
    command: options.command,
    trace_file_sha256: await sha256File(options.tracePath),
    fixture_file_sha256: await sha256File(options.fixturesPath),
    eval_mode: options.mode,
    external_mutation_allowed: false,
  };
}

export async function sha256File(path: string): Promise<string> {
  const bytes = await readFile(path);
  return createHash("sha256").update(bytes).digest("hex");
}

export function stableHash(value: unknown): string {
  return createHash("sha256").update(stableStringify(value)).digest("hex");
}

function gitDirty(): boolean | null {
  try {
    const output = execFileSync("git", ["status", "--porcelain"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
    return output.trim().length > 0;
  } catch {
    return null;
  }
}

function gitOrNull(args: string[]): string | null {
  try {
    return execFileSync("git", args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  return `{${entries
    .map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`)
    .join(",")}}`;
}
