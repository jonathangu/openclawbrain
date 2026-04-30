#!/usr/bin/env node
import { spawnSync } from 'node:child_process';

const checkOnly = process.argv.includes('--check');
const args = ['-p', 'tsconfig.json'];
if (checkOnly) args.push('--noEmit');
const result = spawnSync('tsc', args, { cwd: new URL('..', import.meta.url), stdio: 'inherit' });
if (result.status !== 0) process.exit(result.status ?? 1);
console.log(`${checkOnly ? 'Checked' : 'Built'} OpenClawBrain TypeScript plugin.`);
