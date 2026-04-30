#!/usr/bin/env node
import { spawnSync } from 'node:child_process';

const targets = process.argv.slice(2);
const targetToScript = new Map([
  ['packages/runtime-policy', 'test:runtime-policy'],
  ['packages/proof-store', 'test:proof-store'],
  ['packages/openclaw-plugin', 'test:openclaw-plugin'],
  ['packages/openclaw-integration', 'test:openclaw-integration'],
  ['packages/installer', 'test:installer'],
  ['packages/cli', 'test:cli'],
]);

const scripts = targets.length === 0
  ? ['test:product']
  : targets.map((target) => targetToScript.get(target.replace(/\/$/, '')) || targetToScript.get(target.replace(/^\.\//, '').replace(/\/$/, '')));

if (scripts.some((script) => !script)) {
  console.error(`Unknown test target. Known targets: ${[...targetToScript.keys()].join(', ')}`);
  process.exit(1);
}

for (const script of scripts) {
  const result = spawnSync('pnpm', [script], { stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
