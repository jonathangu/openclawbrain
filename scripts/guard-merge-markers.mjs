#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const trackedFiles = execFileSync("git", ["ls-files", "-z"], {
  cwd: rootDir,
  encoding: "utf8"
})
  .split("\0")
  .filter(Boolean);

const offenders = [];

for (const relativePath of trackedFiles) {
  const absolutePath = path.join(rootDir, relativePath);
  const source = readFileSync(absolutePath);

  if (source.includes(0)) {
    continue;
  }

  const lines = source.toString("utf8").split(/\r?\n/u);
  for (let index = 0; index < lines.length; index += 1) {
    if (/^(<<<<<<< |>>>>>>> )/u.test(lines[index])) {
      offenders.push(`${relativePath}:${index + 1}:${lines[index]}`);
    }
  }
}

if (offenders.length > 0) {
  console.error("merge conflict markers detected in tracked files:");
  for (const offender of offenders) {
    console.error(`- ${offender}`);
  }
  process.exit(1);
}
