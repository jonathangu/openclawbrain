#!/usr/bin/env node

import { runGraphifySchedulerCli } from "./graphify-scheduler.mjs";

try {
  process.exitCode = runGraphifySchedulerCli("delta", process.argv.slice(2));
}
catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exit(1);
}
