#!/usr/bin/env tsx

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  readAndValidateColdStartSourceIntakeRegistryV1,
} from "../src/brain-core/cold-start-source-intake.js";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const defaultRegistryPath = resolve(scriptDir, "../data/cold-start/registry.bootstrap.json");
const registryPath = process.argv[2] ? resolve(process.cwd(), process.argv[2]) : defaultRegistryPath;

const { validation, summary } = readAndValidateColdStartSourceIntakeRegistryV1(registryPath);

if (!validation.valid) {
  console.error(JSON.stringify({ registryPath, valid: false, issues: validation.issues }, null, 2));
  process.exit(1);
}

console.log(
  JSON.stringify(
    {
      registryPath,
      valid: true,
      summary,
    },
    null,
    2,
  ),
);
