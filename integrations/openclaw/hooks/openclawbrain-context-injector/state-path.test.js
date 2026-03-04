import assert from "node:assert/strict";
import { homedir } from "node:os";
import * as path from "node:path";
import { resolveStatePath } from "./state-path.js";

const home = homedir();

assert.equal(
  resolveStatePath("agent-a", {
    OPENCLAWBRAIN_STATE_PATH: "/tmp/openclawbrain/state.json",
    OPENCLAWBRAIN_AGENT_ID: "forced",
  }),
  "/tmp/openclawbrain/state.json",
);

assert.equal(
  resolveStatePath("agent-a", {
    OPENCLAWBRAIN_AGENT_ID: "forced",
  }),
  path.join(home, ".openclawbrain", "forced", "state.json"),
);

assert.equal(
  resolveStatePath("agent-a", {
    OPENCLAWBRAIN_STATE_PATH: "  ",
  }),
  path.join(home, ".openclawbrain", "agent-a", "state.json"),
);
