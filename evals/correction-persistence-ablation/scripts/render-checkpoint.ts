import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import Database from "better-sqlite3";

import { Ledger } from "../src/ledger/ledger.js";
import { generateResults } from "../src/results/generate.js";
import type { MemoryBackend, TaskCase, TurnSlice } from "../src/types.js";

async function main(): Promise<void> {
  const ledgerPath = process.env.OCB_LEDGER ?? "./ocb-ledger.sqlite";
  const casesPath = process.env.OCB_CASES ?? "./cases/correction-recurrence.json";
  const resultsDir = process.env.OCB_RESULTS ?? "./results";
  const runId = process.env.OCB_RUN_ID;

  if (!runId) {
    throw new Error("OCB_RUN_ID is required for checkpoint rendering");
  }

  const cases = JSON.parse(await readFile(casesPath, "utf8")) as TaskCase[];
  const backends: MemoryBackend[] =
    process.env.OCB_WIRE_FULL === "1"
      ? ["none", "correction-only", "correction-plus-heuristics", "full-ocb"]
      : ["none", "correction-only", "correction-plus-heuristics"];
  const slices: Array<TurnSlice | "all"> = ["all", ...new Set(cases.map((taskCase) => taskCase.slice))];
  const expectedOutcomes = cases.length * backends.length;

  const db = new Database(ledgerPath, { readonly: true });
  const progress = db
    .prepare(
      `SELECT COUNT(*) AS decisions,
              (SELECT COUNT(*) FROM outcomes WHERE run_id = ?) AS outcomes
         FROM decisions
        WHERE run_id = ?`,
    )
    .get(runId, runId) as { decisions: number; outcomes: number };
  db.close();

  const ledger = new Ledger(ledgerPath);
  try {
    const { html, json } = generateResults(ledger, {
      run_id: runId,
      backends,
      slices,
      title: "OpenClawBrain — Correction-persistence ablation checkpoint",
      notes: `Checkpoint snapshot. ${progress.outcomes}/${expectedOutcomes} outcomes are complete so far. Each denominator reflects completed outcomes in the ledger at render time, not the full preregistered suite.`,
    });

    await mkdir(resultsDir, { recursive: true });
    await writeFile(join(resultsDir, "checkpoint.html"), html);
    await writeFile(
      join(resultsDir, "checkpoint.json"),
      JSON.stringify(
        {
          run_id: runId,
          generated_at: new Date().toISOString(),
          expected_outcomes: expectedOutcomes,
          completed_outcomes: progress.outcomes,
          logged_decisions: progress.decisions,
          results: (json as { results: unknown[] }).results,
        },
        null,
        2,
      ),
    );
    console.log(`wrote ${join(resultsDir, "checkpoint.html")}`);
    console.log(`wrote ${join(resultsDir, "checkpoint.json")}`);
    console.log(`progress ${progress.outcomes}/${expectedOutcomes}`);
  } finally {
    ledger.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
