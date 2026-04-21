import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";

import Database from "better-sqlite3";

import { Ledger } from "../src/ledger/ledger.js";
import type { TaskCase } from "../src/types.js";

interface ProgressSnapshot {
  decisions: number;
  outcomes: number;
  pending: number;
}

async function main(): Promise<void> {
  const model = process.env.OCB_MODEL ?? "qwen2.5:32b-instruct";
  const ollamaHost = process.env.OCB_OLLAMA_HOST ?? "http://localhost:11434";
  const timeoutMs = Number.parseInt(process.env.OCB_TIMEOUT_MS ?? "", 10) || 600_000;
  const maxRetries = Number.parseInt(process.env.OCB_MAX_RETRIES ?? "", 10) || 2;
  const ledgerPath = process.env.OCB_LEDGER ?? "./ocb-ledger.sqlite";
  const casesPath = process.env.OCB_CASES ?? "./cases/correction-recurrence.json";
  const resultsDir = process.env.OCB_RESULTS ?? "./results";
  const runId = process.env.OCB_RUN_ID ?? randomUUID();
  const backendCount = process.env.OCB_WIRE_FULL === "1" ? 4 : 3;
  const checkEveryMs = Number.parseInt(process.env.OCB_SUPERVISOR_CHECK_MS ?? "", 10) || 15_000;
  const stallMs = Number.parseInt(process.env.OCB_SUPERVISOR_STALL_MS ?? "", 10) || 180_000;
  const maxRestarts = Number.parseInt(process.env.OCB_SUPERVISOR_MAX_RESTARTS ?? "", 10) || 20;

  const cases = JSON.parse(await readFile(casesPath, "utf8")) as TaskCase[];
  const expectedOutcomes = cases.length * backendCount;

  ensureLedger(ledgerPath);

  console.log(
    `supervisor run_id=${runId} expected_outcomes=${expectedOutcomes} check_ms=${checkEveryMs} stall_ms=${stallMs} max_restarts=${maxRestarts}`,
  );

  let restarts = 0;
  let lastProgressAt = Date.now();
  let lastOutcomeCount = -1;
  let child: ChildProcess | null = null;

  while (true) {
    const before = getProgress(ledgerPath, runId);
    if (before.outcomes >= expectedOutcomes) {
      console.log(`supervisor complete run_id=${runId} outcomes=${before.outcomes}/${expectedOutcomes}`);
      return;
    }

    if (!child) {
      if (restarts > maxRestarts) {
        throw new Error(`Supervisor restart limit exceeded for run_id=${runId}`);
      }
      child = startChild({
        runId,
        model,
        ollamaHost,
        timeoutMs,
        maxRetries,
        ledgerPath,
        casesPath,
        resultsDir,
      });
      restarts += 1;
      console.log(`supervisor launch=${restarts} run_id=${runId} outcomes=${before.outcomes}/${expectedOutcomes}`);
    }

    const status = await waitForChildOrTimeout(child, checkEveryMs);
    const after = getProgress(ledgerPath, runId);
    const progressMade = after.outcomes > lastOutcomeCount;
    if (progressMade) {
      lastOutcomeCount = after.outcomes;
      lastProgressAt = Date.now();
    }

    if (after.outcomes >= expectedOutcomes) {
      if (status === "running") {
        child.kill("SIGTERM");
      }
      console.log(`supervisor complete run_id=${runId} outcomes=${after.outcomes}/${expectedOutcomes}`);
      return;
    }

    if (status === "exited") {
      child = null;
      continue;
    }

    const stalled = after.pending > 0 && Date.now() - lastProgressAt >= stallMs;
    if (stalled) {
      console.log(
        `supervisor stall run_id=${runId} decisions=${after.decisions} outcomes=${after.outcomes} pending=${after.pending}, restarting child`,
      );
      child.kill("SIGTERM");
      await waitForChildExit(child, 30_000);
      child = null;
    }
  }
}

function startChild(args: {
  runId: string;
  model: string;
  ollamaHost: string;
  timeoutMs: number;
  maxRetries: number;
  ledgerPath: string;
  casesPath: string;
  resultsDir: string;
}): ChildProcess {
  const env = {
    ...process.env,
    OCB_RUN_ID: args.runId,
    OCB_MODEL: args.model,
    OCB_OLLAMA_HOST: args.ollamaHost,
    OCB_TIMEOUT_MS: String(args.timeoutMs),
    OCB_MAX_RETRIES: String(args.maxRetries),
    OCB_LEDGER: args.ledgerPath,
    OCB_CASES: args.casesPath,
    OCB_RESULTS: args.resultsDir,
  };

  return spawn(
    process.execPath,
    ["./node_modules/tsx/dist/cli.mjs", "scripts/run-ablation.ts"],
    {
      cwd: process.cwd(),
      env,
      stdio: "inherit",
    },
  );
}

function getProgress(ledgerPath: string, runId: string): ProgressSnapshot {
  ensureLedger(ledgerPath);
  const db = new Database(ledgerPath, { readonly: true });
  try {
    const row = db
      .prepare(
        `SELECT
           COUNT(*) AS decisions,
           (SELECT COUNT(*) FROM outcomes WHERE run_id = ?) AS outcomes
         FROM decisions
         WHERE run_id = ?`,
      )
      .get(runId, runId) as { decisions: number; outcomes: number };
    const pending = db
      .prepare(
        `SELECT COUNT(*) AS n
         FROM decisions d
         LEFT JOIN outcomes o ON o.decision_id = d.decision_id
         WHERE d.run_id = ? AND o.decision_id IS NULL`,
      )
      .get(runId) as { n: number };
    return {
      decisions: row.decisions,
      outcomes: row.outcomes,
      pending: pending.n,
    };
  } finally {
    db.close();
  }
}

function ensureLedger(ledgerPath: string): void {
  if (!existsSync(ledgerPath)) {
    const ledger = new Ledger(ledgerPath);
    ledger.close();
    return;
  }

  const db = new Database(ledgerPath, { readonly: true });
  try {
    const tables = db
      .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('decisions', 'outcomes')`)
      .all() as Array<{ name: string }>;
    if (tables.length < 2) {
      db.close();
      const ledger = new Ledger(ledgerPath);
      ledger.close();
      return;
    }
  } finally {
    if (db.open) db.close();
  }
}

function waitForChildOrTimeout(child: ChildProcess, timeoutMs: number): Promise<"exited" | "running"> {
  return new Promise((resolve) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve("running");
    }, timeoutMs);

    const onExit = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve("exited");
    };

    const cleanup = () => {
      clearTimeout(timer);
      child.off("exit", onExit);
    };

    child.once("exit", onExit);
  });
}

function waitForChildExit(child: ChildProcess, timeoutMs: number): Promise<void> {
  return new Promise((resolve) => {
    if (child.exitCode !== null || child.killed) {
      resolve();
      return;
    }

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      resolve();
    }, timeoutMs);

    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
