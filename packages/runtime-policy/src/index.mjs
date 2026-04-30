import { createHash } from 'node:crypto';

const VALID_MODES = new Set(["off", "proof-only", "conservative", "active"]);
const VALID_TURN_TYPES = new Set(["direct-answer", "correction-follow-up", "stale-memory-conflict", "continuation", "retrieval-heavy", "tool-heavy", "unknown"]);
const RAW_FIELD_RE = /(^|[_-])(raw[_-]?(messages?|transcripts?|texts?|contents?)|unredacted|private[_-]?text)$/iu;
const SECRET_FIELD_RE = /(^|[_-])(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|secret|private[_-]?key)$/iu;
const SECRET_VALUE_RE = /(-----BEGIN [A-Z ]*PRIVATE KEY-----|(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{12,}|AKIA[0-9A-Z]{12,}|API_KEY=|PASSW…)/u;
export const RUNTIME_POLICY_VERSION = "ocb.runtime-policy.selected-product.v1";

export function decideOpenClawBrainIntervention(input) {
  const normalized = normalizeRuntimePolicyInput(input);
  const proofBase = buildProofBase(normalized);
  if (normalized.runtimeMode === "off") return staySilent({ ...proofBase, reason: "runtime mode is off", reason_code: "mode_off" });
  const requested = selectedPolicyKind(normalized.redactedTurn.turnType);
  const proof = { ...proofBase, reason: selectedPolicyReason(normalized.redactedTurn.turnType, requested), reason_code: selectedReasonCode(normalized.redactedTurn.turnType, requested), candidate_memory_count: normalized.candidateMemories.length, selected_memory_ids_redacted: selectMemoryIds(normalized) };
  if (normalized.runtimeMode === "proof-only") return { kind: "proof_only", proof: { ...proof, decision: "proof_only", decisionKind: "proof_only", reason: `proof-only mode: ${proof.reason}`, reason_code: "proof_only_mode" } };
  if (requested === "stay_silent") return { kind: "stay_silent", proof: { ...proof, decision: "stay_silent", decisionKind: "stay_silent" } };
  if (requested === "correction_only") {
    const message = correctionMessage(normalized);
    if (!message) return staySilent({ ...proof, reason: "correction policy fired but no safe redacted correction candidate was available", reason_code: "no_safe_correction_candidate" });
    return { kind: "correction_only", message, proof: { ...proof, decision: "correction_only", decisionKind: "correction_only" } };
  }
  if (requested === "full_context") {
    const context = fullContextMessage(normalized);
    if (!context) return staySilent({ ...proof, reason: "context policy fired but no safe redacted context candidate was available", reason_code: "no_safe_context_candidate" });
    return { kind: "full_context", context, proof: { ...proof, decision: "full_context", decisionKind: "full_context" } };
  }
  return staySilent({ ...proof, reason: "unknown turn type defaults to silence", reason_code: "unknown_turn_silence" });
}

export function normalizeRuntimePolicyInput(input) {
  const issues = validateRuntimePolicyInput(input);
  if (issues.length > 0) { const error = new Error(`runtime policy input rejected: ${issues.join("; ")}`); error.issues = issues; throw error; }
  const runtimeMode = input.runtimeMode ?? "conservative";
  const scope = normalizeScope(input);
  return { profileId: String(input.profileId).trim(), runtimeMode, scope, redactedTurn: normalizeRedactedTurn(input.redactedTurn, scope), candidateMemories: (input.candidateMemories ?? []).map(normalizeMemoryCandidate).filter(Boolean), toolContext: input.toolContext ? normalizeToolContext(input.toolContext) : undefined };
}

export function validateRuntimePolicyInput(input) {
  const issues = [];
  if (!isRecord(input)) return ["input must be an object"];
  scanUnsafe(input, [], issues);
  if (typeof input.profileId !== "string" || input.profileId.trim() === "") issues.push("profileId must be a non-empty string");
  if (!isRecord(input.redactedTurn)) issues.push("redactedTurn must be an object");
  if (input.candidateMemories !== undefined && !Array.isArray(input.candidateMemories)) issues.push("candidateMemories must be an array when present");
  const mode = input.runtimeMode ?? "conservative";
  if (!VALID_MODES.has(mode)) issues.push("runtimeMode must be off, proof-only, conservative, or active");
  return issues;
}

export function classifyRedactedTurn(redactedTurn = {}) {
  const explicit = typeof redactedTurn.turnType === "string" ? redactedTurn.turnType.trim() : "";
  if (VALID_TURN_TYPES.has(explicit)) return explicit;
  const text = String(redactedTurn.summary ?? redactedTurn.userMessageRedacted ?? "").toLowerCase();
  if (/^(continue|go on|keep going|proceed)\b/.test(text.trim())) return "continuation";
  if (/\b(actually|correction|don't|do not|instead|preference)\b/.test(text)) return "correction-follow-up";
  if (/\b(verify|check|inspect|status|current|latest|run|test|build|doctor)\b/.test(text)) return "tool-heavy";
  if (/\b(remember|prior|previous|history|find|search|context)\b/.test(text)) return "retrieval-heavy";
  if (/^\s*(what is|calculate|convert|define|summarize in one sentence)\b/.test(text) || /\b\d+\s*[%+*/-]/.test(text)) return "direct-answer";
  return "unknown";
}

export function hashScopeValue(value) { const text = safeString(value); return text ? `sha256:${createHash('sha256').update(text).digest('hex').slice(0, 24)}` : null; }
function selectedPolicyKind(turnType) { return ({ "direct-answer":"stay_silent", unknown:"stay_silent", "correction-follow-up":"correction_only", "stale-memory-conflict":"correction_only", continuation:"full_context", "retrieval-heavy":"full_context", "tool-heavy":"full_context" })[turnType] ?? "stay_silent"; }
function selectedPolicyReason(turnType, decision) { if (decision === "stay_silent") return `${turnType} turn: conservative policy stays silent`; if (decision === "correction_only") return `${turnType} turn: inject only the relevant redacted correction`; return `${turnType} turn: bounded context is useful and allowed`; }
function selectedReasonCode(turnType, decision) { return `${turnType}.${decision}`.replace(/[^a-z0-9_.-]/gi, '_'); }
function normalizeScope(input) { const redactedTurn = input.redactedTurn ?? {}; const scope = input.scope ?? {}; return { openclawProfile: safeString(scope.openclawProfile ?? input.openclawProfile ?? redactedTurn.openclawProfile ?? input.profileId), agentId: safeString(scope.agentId ?? input.agentId ?? redactedTurn.agentId ?? input.profileId), sessionKeyHash: safeString(scope.sessionKeyHash ?? redactedTurn.sessionKeyHash ?? hashScopeValue(scope.sessionKey ?? input.sessionKey ?? redactedTurn.sessionKey)), sessionIdHash: safeString(scope.sessionIdHash ?? redactedTurn.sessionIdHash ?? hashScopeValue(scope.sessionId ?? input.sessionId ?? redactedTurn.sessionId)), runIdHash: safeString(scope.runIdHash ?? redactedTurn.runIdHash ?? hashScopeValue(scope.runId ?? input.runId ?? redactedTurn.runId)) }; }
function normalizeRedactedTurn(turn, scope) { const turnType = classifyRedactedTurn(turn); return { turnId: safeString(turn.turnId ?? turn.id ?? "turn-redacted"), turnType, summary: safeString(turn.summary ?? turn.userMessageRedacted ?? ""), channel: safeString(turn.channel ?? "unknown"), agentId: safeString(turn.agentId ?? turn.profileId ?? scope.agentId ?? "main"), openclawProfile: scope.openclawProfile, sessionKeyHash: scope.sessionKeyHash, sessionIdHash: scope.sessionIdHash, runIdHash: scope.runIdHash }; }
function normalizeMemoryCandidate(candidate) { if (!isRecord(candidate)) return null; return { id: safeString(candidate.id ?? candidate.memoryId ?? "memory-redacted"), kind: safeString(candidate.kind ?? candidate.type ?? "context"), text: safeString(candidate.text ?? candidate.summary ?? candidate.redactedText ?? ""), relevance: numberOrDefault(candidate.relevance ?? candidate.relevanceScore, 0), stale: candidate.stale === true, conflict: candidate.conflict === true }; }
function normalizeToolContext(toolContext) { return { summary: safeString(toolContext.summary ?? ""), readOnlyPreferred: toolContext.readOnlyPreferred !== false }; }
function correctionMessage(input) { const candidate = bestCandidate(input, (memory) => memory.kind.includes("correction") || memory.kind.includes("preference") || memory.conflict); return candidate?.text ? `Relevant user correction: ${candidate.text}` : ""; }
function fullContextMessage(input) { const candidate = bestCandidate(input, (memory) => !memory.stale && !memory.conflict); const parts = []; if (candidate?.text) parts.push(input.redactedTurn.turnType === "tool-heavy" ? `Verification context: ${candidate.text}` : `Continuation context: ${candidate.text}`); if (input.redactedTurn.turnType === "tool-heavy") parts.push("Tool-heavy turn: verify current facts before making a claim. Prefer read-only sources."); return parts.join("\n").trim(); }
function bestCandidate(input, predicate) { return input.candidateMemories.filter((memory) => memory.text && !memory.stale && predicate(memory)).sort((a, b) => b.relevance - a.relevance || a.id.localeCompare(b.id))[0]; }
function selectMemoryIds(input) { return input.candidateMemories.filter((memory) => !memory.stale && memory.text).map((memory) => memory.id).slice(0, 5); }
function buildProofBase(input) { return { schema: "ocb.proof_event.v1", policy_version: RUNTIME_POLICY_VERSION, profile_id: input.profileId, openclaw_profile: input.scope.openclawProfile, agent_id: input.scope.agentId, session_key_hash: input.scope.sessionKeyHash, session_id_hash: input.scope.sessionIdHash, run_id_hash: input.scope.runIdHash, runtime_mode: input.runtimeMode, turn_id_redacted: input.redactedTurn.turnId, turn_type: input.redactedTurn.turnType, raw_text_stored: false, raw_transcript_upload: false, rawTranscriptStored: false, containsRealUserData: false, contains_real_user_data: false, redacted_proof_event: true, created_at: new Date().toISOString() }; }
function staySilent(proof) { return { kind: "stay_silent", proof: { ...proof, decision: "stay_silent", decisionKind: "stay_silent" } }; }
function scanUnsafe(value, path, issues) { if (Array.isArray(value)) return value.forEach((item, index) => scanUnsafe(item, [...path, String(index)], issues)); if (isRecord(value)) { for (const [key, child] of Object.entries(value)) { const nextPath = [...path, key]; if (RAW_FIELD_RE.test(key)) issues.push(`${nextPath.join(".")}: raw/unredacted fields are not allowed`); if (SECRET_FIELD_RE.test(key)) issues.push(`${nextPath.join(".")}: secret-like fields are not allowed`); scanUnsafe(child, nextPath, issues); } return; } if (typeof value === "string" && SECRET_VALUE_RE.test(value)) issues.push(`${path.join(".") || "<root>"}: secret-like values are not allowed`); }
function numberOrDefault(value, fallback) { const number = Number(value); return Number.isFinite(number) ? number : fallback; }
function safeString(value) { return typeof value === "string" ? value.trim() : String(value ?? "").trim(); }
function isRecord(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
