#!/usr/bin/env node
import { spawn } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const runId = process.env.OCB_RUN_ID || "smoke";
const runDir = join("eval", "results", runId);
const docsResultsDir = join("docs", "results");
const warning = "NOT PRODUCT EVIDENCE / SYNTHETIC PIPELINE VALIDATION ONLY";

async function main() {
  await step("schema tests", ["pnpm", "ocb:results:schema-test"]);
  await step("trace validation", ["pnpm", "ocb:traces:validate"]);
  await step("eval run", ["pnpm", "ocb:eval:run", "--", "--mode", "smoke", "--run-id", runId]);
  await step("blind packets", ["pnpm", "ocb:eval:make-blind-packets", "--", "--run-id", runId]);
  await writeSyntheticJudgments();
  await step("judgment import", ["pnpm", "ocb:judgments:import", "--", "--mode", "smoke", "--run-id", runId, "--judgments", join(runDir, "judgments.synthetic.json")]);
  await writeRunState({ resultsGenerated: false, decisionGenerated: false });
  await step("results generation", ["pnpm", "ocb:results:generate", "--", "--ledger", join(runDir, "ledger-judged.synthetic.jsonl")]);
  await step("decision generation", ["pnpm", "ocb:decision:generate", "--", "--ledger", join(runDir, "ledger-judged.synthetic.jsonl")]);
  await writeBlockerArtifacts();
  await writeRunState({ resultsGenerated: true, decisionGenerated: true });
  await verifyArtifacts();
  console.log(JSON.stringify({ ok: true, run_id: runId, engineering_e2e_complete: true, evidence_e2e_complete: false, run_state: join(runDir, "RUN_STATE.json") }, null, 2));
}

async function step(label, command) {
  console.log(`== ${label}`);
  await run(command);
}

function run(command) {
  return new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), { stdio: "inherit", shell: false });
    child.on("exit", (code) => code === 0 ? resolve() : reject(new Error(`${command.join(" ")} exited ${code}`)));
    child.on("error", reject);
  });
}

async function writeSyntheticJudgments() {
  const packetsDir = join(runDir, "blind-judge-packets");
  const judgments = [];
  for (const file of (await readdir(packetsDir)).filter((file) => file.endsWith(".json") && !file.startsWith("_"))) {
    const packet = JSON.parse(await readFile(join(packetsDir, file), "utf8"));
    for (const candidate of packet.candidates) {
      judgments.push({
        trace_id: packet.trace_id,
        candidate_id: candidate.candidate_id,
        overall_score: syntheticScore(candidate.answer),
        judge_id: "synthetic-smoke-judge",
        notes: "Deterministic smoke judgment; validates plumbing only.",
        synthetic: true,
      });
    }
  }
  await writeFile(join(runDir, "judgments.synthetic.json"), `${JSON.stringify({ mode: "smoke", synthetic: true, warning, judgments }, null, 2)}\n`, "utf8");
}

function syntheticScore(answer) {
  const text = String(answer ?? "").toLowerCase();
  if (text.includes("correction") || text.includes("fixture") || text.includes("evidence limits")) return 4;
  return 3;
}

async function writeRunState({ resultsGenerated, decisionGenerated }) {
  const draftRows = countLines(await readFile(join(runDir, "ledger-draft.jsonl"), "utf8"));
  const judgedRows = countLines(await readFile(join(runDir, "ledger-judged.synthetic.jsonl"), "utf8"));
  const packetCount = (await readdir(join(runDir, "blind-judge-packets"))).filter((file) => file.endsWith(".json") && !file.startsWith("_")).length;
  await writeFile(join(runDir, "RUN_STATE.json"), `${JSON.stringify({
    schema_version: "ocb.run_state.v1",
    run_id: runId,
    mode: "smoke",
    warning,
    engineering_e2e_complete: true,
    evidence_e2e_complete: false,
    trace_count: packetCount,
    real_trace_count: 0,
    synthetic_trace_count: packetCount,
    all_slice_minimums_met: false,
    all_backends_run: draftRows === packetCount * 4,
    blind_packets_generated: packetCount > 0,
    judging_complete: judgedRows === draftRows && judgedRows > 0,
    ledger_valid: judgedRows === draftRows && judgedRows > 0,
    results_generated: resultsGenerated,
    decision_generated: decisionGenerated,
    judge_disagreement_within_threshold: false,
    blockers: [
      "Synthetic smoke traces are not product evidence.",
      "Need at least 40 admitted real privacy-scrubbed traces across V5 slice minimums.",
      "Need completed non-synthetic blind judgments for production evidence.",
      "Runtime decision events and candidates are candidate-only until admitted and judged.",
    ],
  }, null, 2)}\n`, "utf8");
}

async function writeBlockerArtifacts() {
  await mkdir(docsResultsDir, { recursive: true });
  await writeFile(join(docsResultsDir, "BLOCKERS.md"), `# Evidence Blockers\n\n- Synthetic smoke traces are not product evidence.\n- 0/40 admitted real privacy-scrubbed traces are present.\n- Production blind judging is not complete.\n- V5 slice minimums are not met with real admitted traces.\n- Runtime decision events and candidates remain candidate-only until admitted and judged through the production evidence gate.\n`, "utf8");
  await writeFile(join(docsResultsDir, "NEXT_DATA_NEEDED.md"), `# Next Data Needed\n\n1. Collect at least 40 admitted real privacy-scrubbed traces from actual agent turns.\n2. Cover V5 slices: direct-answer, continuation, correction-follow-up, retrieval-heavy, tool-heavy, stale-memory-conflict.\n3. Export runtime decision events into trace candidates, admit only valid real redacted traces, and keep rejected candidates out of product counts.\n4. Generate blind packets and import non-synthetic judgments.\n5. Regenerate results and apply product thresholds.\n`, "utf8");
  await writeFile(join(docsResultsDir, "PARTIAL_COMPLETION.md"), `# Partial Completion\n\nEngineering E2E is complete for smoke mode: schema tests, trace validation, four-backend eval, blind packets, synthetic judgment import, results generation, decision generation, and RUN_STATE writing.\n\nThe minimal runtime decision interface is connected to candidate-only runtime event capture and trace candidate export, so actual agent turns can now feed the V5 evidence pipeline without bypassing admission.\n\nEvidence E2E remains false by design until 40 admitted real privacy-scrubbed traces, required slice minimums, and real blind judgments exist.\n`, "utf8");
}

async function verifyArtifacts() {
  const required = [
    join(runDir, "ledger-draft.jsonl"),
    join(runDir, "ledger-judged.synthetic.jsonl"),
    join(runDir, "judgments.synthetic.json"),
    join(runDir, "RUN_STATE.json"),
    join(docsResultsDir, "index.md"),
    join(docsResultsDir, "summary.json"),
    join(docsResultsDir, "30_DAY_DECISION.blocked.md"),
    join(docsResultsDir, "BLOCKERS.md"),
    join(docsResultsDir, "NEXT_DATA_NEEDED.md"),
    join(docsResultsDir, "PARTIAL_COMPLETION.md"),
  ];
  for (const path of required) await readFile(path, "utf8");
  const state = JSON.parse(await readFile(join(runDir, "RUN_STATE.json"), "utf8"));
  if (state.engineering_e2e_complete !== true || state.evidence_e2e_complete !== false) throw new Error("RUN_STATE gate mismatch");
}

function countLines(text) { return text.split(/\r?\n/u).filter((line) => line.trim()).length; }

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
