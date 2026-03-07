#!/usr/bin/env node

import { runObservabilityScenario } from "./observability-smoke.mjs";

try {
  runObservabilityScenario({
    logPrefix: "observability:report"
  });
} catch (error) {
  console.error("[observability:report] failed");
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}
