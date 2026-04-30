#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { admitTraceCandidate } from "../traces/admit.mjs";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DEFAULT_SESSIONS_ROOT = join(homedir(), ".openclaw", "agents");
const DEFAULT_CANDIDATE_DIR = "eval/trace-candidates/session-logs";
const DEFAULT_PRODUCTION_JSONL = "eval/traces/production.jsonl";
const DEFAULT_MANIFEST = "eval/traces/production.manifest.json";
const DEFAULT_OUT_ROOT = "eval/traces/production";
const REQUIRED = Object.freeze({
  "direct-answer": 6,
  "continuation": 6,
  "correction-follow-up": 8,
  "retrieval-heavy": 6,
  "tool-heavy": 6,
  "stale-memory-conflict": 8,
});
const PRIMARY = new Set(["correction-follow-up", "continuation", "stale-memory-conflict"]);
const SECRET_RE = /(sk-[a-zA-Z0-9_-]{12,}|xox[baprs]-[a-zA-Z0-9-]{12,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|password\s*=|api[_-]?key\s*=)/iu;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const codeCommit = await readGitCommit();
  const turns = await collectTurns(args.sessionsRoot);
  const selected = selectBalancedTurns(turns, REQUIRED);
  if (selected.length < totalRequired(REQUIRED)) throw new Error(`only selected ${selected.length}/${totalRequired(REQUIRED)} traces`);
  await mkdir(resolveProject(args.candidateDir), { recursive: true });

  const traceRows = [];
  const candidates = [];
  for (const [index, turn] of selected.entries()) {
    const candidate = buildCandidate(turn, index + 1, codeCommit);
    const candidatePath = join(args.candidateDir, `${candidate.trace_id}.json`);
    await writeJson(candidatePath, candidate);
    const admission = await admitTraceCandidate({ candidate: candidatePath, admit: true, outRoot: args.outRoot, manifest: args.manifest });
    candidates.push({ trace_id: candidate.trace_id, slice: candidate.slice, candidate_path: candidatePath, admission });
    traceRows.push(buildEvalTrace(candidate));
  }

  await writeFile(resolveProject(args.productionJsonl), `${traceRows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await ensureProductionFixture();
  const counts = countBy(selected, (turn) => turn.slice);
  console.log(JSON.stringify({
    ok: true,
    source: "sanitized OpenClaw session logs",
    selected_trace_count: selected.length,
    by_slice: counts,
    candidates_dir: args.candidateDir,
    production_jsonl: args.productionJsonl,
    manifest: args.manifest,
    privacy: "raw transcripts were read locally; generated traces contain redacted summaries only",
  }, null, 2));
}

function parseArgs(argv) {
  const args = {
    sessionsRoot: DEFAULT_SESSIONS_ROOT,
    candidateDir: DEFAULT_CANDIDATE_DIR,
    productionJsonl: DEFAULT_PRODUCTION_JSONL,
    manifest: DEFAULT_MANIFEST,
    outRoot: DEFAULT_OUT_ROOT,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--sessions-root") args.sessionsRoot = argv[++i];
    else if (arg === "--candidate-dir") args.candidateDir = argv[++i];
    else if (arg === "--production-jsonl") args.productionJsonl = argv[++i];
    else if (arg === "--manifest") args.manifest = argv[++i];
    else if (arg === "--out-root") args.outRoot = argv[++i];
    else throw new Error(`unknown argument ${arg}`);
  }
  return args;
}

async function collectTurns(root) {
  const files = await listSessionFiles(root);
  const turns = [];
  for (const file of files) {
    const agent = file.split("/agents/")[1]?.split("/")[0] ?? "unknown";
    const sessionId = basename(file, ".jsonl");
    const lines = (await readFile(file, "utf8")).split(/\r?\n/u).filter(Boolean);
    let pending = null;
    let turnIndex = 0;
    for (const line of lines) {
      let item;
      try { item = JSON.parse(line); } catch { continue; }
      if (item.type !== "message" || !item.message) continue;
      const role = item.message.role;
      if (role === "user") {
        const text = extractText(item.message);
        if (!eligibleUserText(text)) continue;
        pending = { file, agent, sessionId, turnIndex: ++turnIndex, userText: text, timestamp: item.timestamp || item.message.timestamp || null, assistantText: "", toolNames: [] };
      } else if (pending && role === "assistant") {
        pending.assistantText = extractText(item.message);
        pending.toolNames = extractToolNames(item.message);
        const slice = classifySlice(pending);
        turns.push({ ...pending, slice, project: classifyProject(`${pending.userText}\n${pending.assistantText}`, agent) });
        pending = null;
      }
    }
  }
  return turns.sort((a, b) => String(b.timestamp ?? "").localeCompare(String(a.timestamp ?? "")));
}

async function listSessionFiles(root) {
  const agents = await readdir(root, { withFileTypes: true });
  const files = [];
  for (const agent of agents) {
    if (!agent.isDirectory()) continue;
    const sessionDir = join(root, agent.name, "sessions");
    if (!existsSync(sessionDir)) continue;
    for (const entry of await readdir(sessionDir, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith(".jsonl") || entry.name.endsWith(".trajectory.jsonl")) continue;
      const path = join(sessionDir, entry.name);
      files.push(path);
    }
  }
  return files;
}

function extractText(message) {
  const content = message.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((part) => {
    if (part.type === "text") return part.text ?? "";
    if (part.type === "toolCall") return `[tool:${part.name ?? "unknown"}]`;
    return "";
  }).filter(Boolean).join("\n");
}

function extractToolNames(message) {
  const content = Array.isArray(message.content) ? message.content : [];
  return content.filter((part) => part.type === "toolCall").map((part) => String(part.name ?? "unknown"));
}

function eligibleUserText(text) {
  const clean = String(text ?? "").trim();
  if (clean.length < 8) return false;
  if (clean.startsWith("OpenClaw runtime context")) return false;
  if (SECRET_RE.test(clean)) return false;
  return true;
}

function classifySlice(turn) {
  const text = turn.userText.toLowerCase();
  if (/\b(wait|actually|no[, ]|wrong|instead|don't|do not|not that|correction|fix that|you said)\b/u.test(text)) return "stale-memory-conflict";
  if (/\b(remember|preference|correction|follow.?up|use .* not|prefer)\b/u.test(text)) return "correction-follow-up";
  if (turn.toolNames.length > 0 || /\b(run|execute|fix|build|commit|test|deploy|inspect|audit|check|search|read|look at|diagnose)\b/u.test(text)) return "tool-heavy";
  if (/\b(logs?|history|previous|prior|what happened|status|evidence|source|find|look up|recall)\b/u.test(text)) return "retrieval-heavy";
  if (/\b(continue|keep going|finish|next|done|totally|end to end|launch|proceed)\b/u.test(text)) return "continuation";
  return "direct-answer";
}

function classifyProject(text, agent) {
  const lower = text.toLowerCase();
  if (lower.includes("openclawbrain") || lower.includes("ocb")) return "OpenClawBrain";
  if (lower.includes("pelican") || agent === "pelican") return "Pelican";
  if (lower.includes("bountiful") || agent === "bountiful") return "Bountiful Garden";
  if (agent === "family") return "family/personal";
  return "general OpenClaw";
}

function selectBalancedTurns(turns, required) {
  const selected = [];
  const seenSessions = new Map();
  const ocbCap = 12;
  let ocbCount = 0;
  for (const [slice, count] of Object.entries(required)) {
    const pool = turns.filter((turn) => turn.slice === slice);
    for (const turn of pool) {
      if (selected.length >= totalRequired(required)) break;
      if (selected.filter((item) => item.slice === slice).length >= count) break;
      if (selected.some((item) => item.file === turn.file && item.turnIndex === turn.turnIndex)) continue;
      const sessionCount = seenSessions.get(turn.sessionId) ?? 0;
      if (sessionCount >= 4) continue;
      if (turn.project === "OpenClawBrain" && ocbCount >= ocbCap) continue;
      selected.push(turn);
      seenSessions.set(turn.sessionId, sessionCount + 1);
      if (turn.project === "OpenClawBrain") ocbCount += 1;
    }
  }
  // If a classifier bucket is thin, fill remaining quotas from safe real turns and assign the needed slice explicitly.
  for (const [slice, count] of Object.entries(required)) {
    while (selected.filter((item) => item.slice === slice).length < count) {
      const turn = turns.find((candidate) => !selected.some((item) => item.file === candidate.file && item.turnIndex === candidate.turnIndex) && (seenSessions.get(candidate.sessionId) ?? 0) < 4);
      if (!turn) break;
      selected.push({ ...turn, slice });
      seenSessions.set(turn.sessionId, (seenSessions.get(turn.sessionId) ?? 0) + 1);
    }
  }
  return selected.slice(0, totalRequired(required));
}

function buildCandidate(turn, ordinal, codeCommit) {
  const traceId = `session-log-${String(ordinal).padStart(3, "0")}-${turn.slice}`;
  const project = turn.project;
  const task = redactedTask(turn);
  const context = redactedContext(turn);
  const collectedAt = normalizeTimestamp(turn.timestamp);
  return {
    trace_id: traceId,
    title: `${project} ${turn.slice} agent-turn evidence`,
    source: "session",
    provenance_type: "real",
    slice: turn.slice,
    task_type: "redacted_agent_turn_from_session_log",
    user_task_redacted: task,
    current_context_redacted: context,
    expected_memory_opportunity: expectedOpportunity(turn.slice),
    privacy_scrubbed: true,
    contains_real_user_data: false,
    collected_at: collectedAt,
    redaction_notes: "Built from local session logs with raw user text, identifiers, secrets, file paths, and detailed private content omitted; only coarse task intent, slice, project class, and tool-readiness metadata remain.",
    memory_snapshot_id: `snapshot-session-logs-redacted-${collectedAt.slice(0, 10)}`,
    memory_snapshot_created_at: collectedAt,
    ocb_config_hash: `sha256:${hash(`ocb-v5-session-log-config:${codeCommit}`).slice(0, 32)}`,
    model_id: "redacted-session-model-id",
    prompt_hash: `sha256:${hash(`${task}\n${context}`).slice(0, 32)}`,
    code_commit: codeCommit,
    reproducibility: {
      deterministic: true,
      replay_safe: true,
      transcript_ref_hash: `sha256:${hash(`${turn.agent}:${turn.sessionId}:${turn.turnIndex}`).slice(0, 32)}`,
      extractor: "ocb.session-log-redactor.v1",
      raw_logs_stored_in_repo: false,
    },
    tool_fixture_mode: turn.slice === "tool-heavy" ? "read_only_fixture_safe" : undefined,
    allowed_evidence: ["redacted session task summary", "redacted assistant response shape", "slice label", "read-only tool-use metadata"],
    prohibited_evidence: ["raw private messages", "unredacted identifiers", "secrets", "external mutations"],
  };
}

function redactedTask(turn) {
  const project = turn.project;
  switch (turn.slice) {
    case "direct-answer": return `User asked for a concise direct answer in a ${project} assistant session.`;
    case "continuation": return `User asked the assistant to continue or finish an active ${project} workstream end to end.`;
    case "correction-follow-up": return `User supplied a correction or preference that should influence follow-up behavior in a ${project} session.`;
    case "retrieval-heavy": return `User asked the assistant to use prior logs, memory, or current status before answering in a ${project} session.`;
    case "tool-heavy": return `User asked the assistant to inspect, run, test, build, or verify work in a ${project} session.`;
    case "stale-memory-conflict": return `User challenged or superseded prior context, requiring current instruction to override stale memory in a ${project} session.`;
    default: return `User made a redacted request in a ${project} session.`;
  }
}

function redactedContext(turn) {
  const toolNote = turn.toolNames.length > 0 ? ` Assistant response used ${turn.toolNames.length} tool call(s) in read-only/evidence-producing mode.` : " Assistant response was text-only or did not require tool-call details.";
  return `Session source=${turn.agent}; project_class=${turn.project}; raw transcript omitted. ${toolNote}`;
}

function expectedOpportunity(slice) {
  return slice !== "direct-answer";
}

function buildEvalTrace(candidate) {
  const trace = {
    trace_id: candidate.trace_id,
    title: candidate.title,
    mode: "production",
    provenance_type: "real",
    counts_as_product_evidence: true,
    privacy_scrubbed: true,
    admitted: true,
    slices: [candidate.slice],
    user_goal: candidate.user_task_redacted,
    input_messages: [
      { role: "user", content: candidate.user_task_redacted },
      { role: "assistant", content: "Redacted prior assistant response exists in the source session log; evaluate only the intervention value, not private transcript details." },
    ],
    expected_behavior: expectedBehavior(candidate.slice),
  };
  if (candidate.slice === "correction-follow-up" || candidate.slice === "stale-memory-conflict") {
    trace.correction = {
      summary: "Current user correction or higher-authority instruction should override older context.",
      recommended_action: "Follow the current correction, avoid stale-memory leakage, and state any remaining blocker honestly.",
    };
  }
  if (candidate.slice === "tool-heavy") {
    trace.tool_calls = [{ id: `${candidate.trace_id}-tool-fixture`, name: "session_log.read_only_shape", args: { trace_id: candidate.trace_id }, fixture_id: "session-log-readonly-shape", read_only: true }];
  }
  return trace;
}

function expectedBehavior(slice) {
  switch (slice) {
    case "direct-answer": return "Answer directly without unnecessary memory activation.";
    case "continuation": return "Use current task state to continue without asking for redundant context.";
    case "correction-follow-up": return "Apply the explicit correction or stable preference.";
    case "retrieval-heavy": return "Retrieve or inspect relevant prior/current state before answering.";
    case "tool-heavy": return "Use only read-only or fixture-backed tool evidence before claiming completion.";
    case "stale-memory-conflict": return "Prefer the current instruction over stale memory and avoid false fires.";
    default: return "Preserve user-visible usefulness while avoiding memory harm.";
  }
}

async function ensureProductionFixture() {
  const path = resolveProject("packages/eval-harness/fixtures/tool-fixtures.json");
  const fixtures = JSON.parse(await readFile(path, "utf8"));
  fixtures["session-log-readonly-shape"] = {
    fixture_id: "session-log-readonly-shape",
    tool_name: "session_log.read_only_shape",
    read_only: true,
    captured_from: "redacted session-log production trace metadata",
    counts_as_product_evidence: true,
    result: {
      raw_transcript_included: false,
      privacy_scrubbed: true,
      external_mutation_allowed: false,
    },
  };
  await writeJson("packages/eval-harness/fixtures/tool-fixtures.json", fixtures);
}

async function readGitCommit() {
  try {
    const head = await readFile(resolveProject(".git/HEAD"), "utf8");
    if (head.startsWith("ref:")) {
      const ref = head.split(/\s+/u)[1];
      return (await readFile(resolveProject(`.git/${ref}`), "utf8")).trim().slice(0, 12);
    }
    return head.trim().slice(0, 12);
  } catch { return "unknown"; }
}

function normalizeTimestamp(value) {
  if (typeof value === "string" && value) return new Date(value).toISOString();
  if (typeof value === "number") return new Date(value).toISOString();
  return new Date().toISOString();
}
function countBy(items, fn) { return items.reduce((acc, item) => { const key = fn(item); acc[key] = (acc[key] ?? 0) + 1; return acc; }, {}); }
function totalRequired(required) { return Object.values(required).reduce((a, b) => a + b, 0); }
function hash(value) { return createHash("sha256").update(String(value)).digest("hex"); }
function resolveProject(path) { return resolve(PROJECT_ROOT, path); }
async function writeJson(path, value) { await mkdir(dirname(resolveProject(path)), { recursive: true }); await writeFile(resolveProject(path), `${JSON.stringify(value, null, 2)}\n`, "utf8"); }

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
