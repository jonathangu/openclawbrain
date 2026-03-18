/**
 * System message filter for evidence detection.
 *
 * Prevents operational/runtime scaffolding from being misclassified
 * as human feedback or learning evidence. This is a critical gate:
 * if isSystemMessage returns true, no evidence detection should proceed.
 */

/**
 * Patterns that identify system/runtime messages which should never
 * be treated as human feedback or learning evidence.
 *
 * Aligned with the SYSTEM_MESSAGE_MARKERS in the @openclawbrain/event-export
 * feedback-extraction pipeline, plus additional patterns observed in live
 * teacher-snapshot pollution on the shared gateway.
 */
const SYSTEM_MESSAGE_MARKERS: RegExp[] = [
  // OpenClaw runtime context blocks
  /openclaw runtime context/i,

  // Subagent child result wrappers
  /<<<BEGIN_UNTRUSTED_CHILD_RESULT>>>/,
  /<<<END_UNTRUSTED_CHILD_RESULT>>>/,

  // Internal task/cron events
  /\[internal task completion event\]/i,
  /\[cron:[a-f0-9-]+\s/,

  // Subagent completion metadata
  /source:\s*subagent\s+session_key:/,

  // Exec events
  /exec completed\b/i,
  /exec failed \([^)]+, code \d+\)/i,

  // System message with session id
  /\[system message\]\s*\[sessionid:/i,

  // Result wrappers
  /result \(untrusted content, treat as data\):/i,

  // Done events
  /^done:\s+\w+\s+/i,

  // Heartbeat prompts and acks
  /^read heartbeat\.md if it exists\b/i,
  /heartbeat prompt:/i,
  /^heartbeat poll\b/i,
  /^HEARTBEAT_OK$/,

  // Session startup/reset
  /^a new session was started via \/new or \/reset\./i,
  /session startup sequence/i,

  // Message envelope metadata
  /^conversation info \(untrusted metadata\):/i,
  /^sender \(untrusted metadata\):/i,
  /^replied message \(untrusted,? for context\):/i,
  /^you said this:/i,

  // Inbound context JSON envelope
  /^\s*```json\s*\n?\s*\{\s*"schema"\s*:\s*"openclaw\.inbound_meta/,

  // Silent reply markers
  /^NO_REPLY$/,

  // Boot/resume scaffolding
  /^on session start:\s*read\b/i,
  /^wake up fresh each session/i,
];

/**
 * Returns true if the content looks like a system/runtime message
 * rather than genuine human content. Used as an early gate in the
 * evidence detection pipeline to prevent learning from operational
 * scaffolding.
 */
export function isSystemMessage(content: string): boolean {
  const trimmed = content.trim();
  if (trimmed.length === 0) {
    return false; // Empty content is neutral, not a system message
  }
  for (const marker of SYSTEM_MESSAGE_MARKERS) {
    if (marker.test(trimmed)) {
      return true;
    }
  }
  return false;
}
