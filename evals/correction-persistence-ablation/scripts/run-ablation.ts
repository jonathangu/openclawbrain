import { mkdir, readFile, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { join } from "node:path";

import { OllamaClient } from "../src/agent/client.js";
import {
  CorrectionOnlyBackend,
  CorrectionPlusHeuristicsBackend,
  FullOcbBackend,
  NoneBackend,
} from "../src/ablation/backends.js";
import type { MemoryBackendImpl } from "../src/ablation/backends.js";
import { runAblation } from "../src/ablation/runner.js";
import { generateResults } from "../src/results/generate.js";
import { Ledger } from "../src/ledger/ledger.js";
import type { TaskCase, TurnSlice } from "../src/types.js";

async function main(): Promise<void> {
  const model = process.env.OCB_MODEL ?? "qwen2.5:32b-instruct";
  const ollamaHost = process.env.OCB_OLLAMA_HOST ?? "http://localhost:11434";
  const timeoutMs = Number.parseInt(process.env.OCB_TIMEOUT_MS ?? "", 10) || 600_000;
  const maxRetries = Number.parseInt(process.env.OCB_MAX_RETRIES ?? "", 10) || 2;
  const maxOutputTokens = Number.parseInt(process.env.OCB_MAX_OUTPUT_TOKENS ?? "", 10) || 128;
  const ledgerPath = process.env.OCB_LEDGER ?? "./ocb-ledger.sqlite";
  const casesPath = process.env.OCB_CASES ?? "./cases/correction-recurrence.json";
  const resultsDir = process.env.OCB_RESULTS ?? "./results";
  const runId = process.env.OCB_RUN_ID;

  const cases = JSON.parse(await readFile(casesPath, "utf8")) as TaskCase[];
  const ledger = new Ledger(ledgerPath);
  try {
    const agent = new OllamaClient(model, ollamaHost, timeoutMs, maxRetries, maxOutputTokens);
    const backends: MemoryBackendImpl[] = [
      new NoneBackend(),
      new CorrectionOnlyBackend(),
      new CorrectionPlusHeuristicsBackend(),
    ];

    if (process.env.OCB_WIRE_FULL === "1") {
      const { createOcbAdapter } = await import("./ocb-adapter.js");
      backends.push(new FullOcbBackend(await createOcbAdapter()));
    }

    const run_id = runId ?? randomUUID();
    console.log(`run_id=${run_id} model=${model} host=${ollamaHost} timeout_ms=${timeoutMs} max_retries=${maxRetries} max_output_tokens=${maxOutputTokens} cases=${cases.length} backends=${backends.map((backend) => backend.name).join(",")} resume=${runId ? "explicit" : "fresh"}`);

    const { summary } = await runAblation({
      run_id,
      cases,
      backends,
      agent,
      ledger,
      onProgress: (event) =>
        console.log(`[${event.i}/${event.total}] ${event.backend} on ${event.case_id}: ${event.passed ? "PASS" : "FAIL"}`),
    });

    console.log("\n=== summary ===");
    for (const row of summary) {
      console.log(
        `${row.backend.padEnd(28)} ${row.slice.padEnd(25)} pass=${(row.pass_rate * 100).toFixed(1)}% fires=${row.total_fires}/${row.total_cases} regret=${row.abstention_regret_count} harm=${row.false_fire_harm_count}`,
      );
    }

    await mkdir(resultsDir, { recursive: true });
    const slices: Array<TurnSlice | "all"> = ["all", ...new Set(cases.map((taskCase) => taskCase.slice))];
    const { html, json } = generateResults(ledger, {
      run_id,
      backends: backends.map((backend) => backend.name),
      slices,
      title: "OpenClawBrain — Correction-persistence ablation results",
    });
    await writeFile(join(resultsDir, "index.html"), html);
    await writeFile(join(resultsDir, "results.json"), JSON.stringify(json, null, 2));
    console.log(`\nwrote ${join(resultsDir, "index.html")}`);
  } finally {
    ledger.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
