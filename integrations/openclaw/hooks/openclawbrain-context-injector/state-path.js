import { homedir } from "node:os";
import * as path from "node:path";

function normalizeText(value) {
  return typeof value === "string" ? value.trim() : "";
}

export function resolveStatePath(agentId, env = process.env) {
  const override = normalizeText(env?.OPENCLAWBRAIN_STATE_PATH);
  if (override) {
    return override;
  }
  const forcedAgentId = normalizeText(env?.OPENCLAWBRAIN_AGENT_ID);
  const resolvedAgentId = forcedAgentId || normalizeText(agentId) || "main";
  return path.join(homedir(), ".openclawbrain", resolvedAgentId, "state.json");
}
