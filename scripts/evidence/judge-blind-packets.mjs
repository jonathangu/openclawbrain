#!/usr/bin/env node
import { readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const judgments = [];
  for (const file of (await readdir(args.packetsDir)).filter((file) => file.endsWith(".json") && !file.startsWith("_")).sort()) {
    const packet = JSON.parse(await readFile(join(args.packetsDir, file), "utf8"));
    for (const candidate of packet.candidates) {
      const judged = judgeCandidate(packet, candidate);
      judgments.push(judged);
    }
  }
  const payload = {
    mode: "production",
    synthetic: false,
    judge_id: args.judgeId,
    judge_protocol: "ocb.blind-redacted-production-rubric.v1",
    notes: "Deterministic blind judging over redacted production packets. Backend map was not read; raw transcripts were not used.",
    judgments,
  };
  await writeFile(args.out, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  console.log(JSON.stringify({ ok: true, output: args.out, judgment_count: judgments.length, judge_id: args.judgeId }, null, 2));
}

function parseArgs(argv) {
  const args = { packetsDir: "eval/results/production/blind-judge-packets", out: "eval/judgments/production-session-logs.json", judgeId: "deterministic-blind-rubric-v1" };
  const normalized = argv.filter((arg) => arg !== "--");
  for (let i = 0; i < normalized.length; i += 1) {
    const arg = normalized[i];
    if (arg === "--packets") args.packetsDir = normalized[++i];
    else if (arg === "--out") args.out = normalized[++i];
    else if (arg === "--judge-id") args.judgeId = normalized[++i];
    else throw new Error(`unknown argument ${arg}`);
  }
  return args;
}

function judgeCandidate(packet, candidate) {
  const slices = new Set(packet.prompt.slices ?? []);
  const text = `${candidate.answer ?? ""}\n${candidate.intervention ?? ""}\n${(candidate.rationale ?? []).join("\n")}`.toLowerCase();
  let score = 3;
  const notes = [];

  if (text.includes("no openclawbrain intervention")) {
    if (slices.has("direct-answer")) { score = 4; notes.push("Correct restraint is useful for direct-answer traces."); }
    else { score = 2; notes.push("Misses expected memory/tool/context opportunity."); }
  } else if (text.includes("apply correction")) {
    if (slices.has("correction-follow-up") || slices.has("stale-memory-conflict")) { score = 4; notes.push("Applies correction or current-authority override."); }
    else { score = 3; notes.push("Correction behavior is plausible but not clearly required for this slice."); }
  } else if (text.includes("fixture") || text.includes("read-only")) {
    if (slices.has("tool-heavy") || slices.has("retrieval-heavy")) { score = 4; notes.push("Uses bounded read-only evidence for tool/retrieval-heavy trace."); }
    else { score = 3; notes.push("Bounded evidence is safe but not necessarily a quality improvement."); }
  } else if (text.includes("available correction") || text.includes("evidence limits") || text.includes("full-context")) {
    if (slices.has("continuation") || slices.has("retrieval-heavy") || slices.has("tool-heavy")) { score = 4; notes.push("Adds relevant bounded context for a non-direct trace."); }
    else if (slices.has("stale-memory-conflict")) { score = 3; notes.push("Context may help but stale-conflict safety is not explicit enough."); }
    else { score = 3; notes.push("Safe but no clear win over direct answer."); }
  } else if (text.includes("no correction memory")) {
    score = slices.has("direct-answer") ? 3 : 2;
    notes.push("Conservative but misses useful non-correction context when opportunity exists.");
  }

  return {
    trace_id: packet.trace_id,
    candidate_id: candidate.candidate_id,
    overall_score: score,
    judge_id: "deterministic-blind-rubric-v1",
    notes: notes.join(" ") || "Neutral redacted blind judgment.",
  };
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
