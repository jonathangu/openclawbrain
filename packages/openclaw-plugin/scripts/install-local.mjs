#!/usr/bin/env node
import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const packageRoot = path.resolve(new URL('..', import.meta.url).pathname);
const repoRoot = path.resolve(packageRoot, '../..');
const args = process.argv.slice(2);

const build = spawnSync('node', ['scripts/build.mjs'], { cwd: packageRoot, stdio: 'inherit' });
if (build.status !== 0) process.exit(build.status ?? 1);

const homes = await resolveTargetHomes(args);
for (const openclawHome of homes) {
  await installToHome(openclawHome);
}

async function installToHome(openclawHome) {
  const installRoot = path.join(openclawHome, 'extensions', 'openclawbrain');
  const installsPath = path.join(openclawHome, 'plugins', 'installs.json');

  await rm(installRoot, { recursive: true, force: true });
  await mkdir(installRoot, { recursive: true, mode: 0o700 });
  for (const entry of ['dist', 'openclaw.plugin.json', 'package.json', 'README.md', 'LICENSE']) {
    await cp(path.join(packageRoot, entry), path.join(installRoot, entry), { recursive: true });
  }

  const installDeps = spawnSync('pnpm', ['install', '--prod', '--no-frozen-lockfile'], {
    cwd: installRoot,
    stdio: 'inherit',
  });
  if (installDeps.status !== 0) process.exit(installDeps.status ?? 1);

  await mkdir(path.dirname(installsPath), { recursive: true, mode: 0o700 });
  const manifest = JSON.parse(await readFile(path.join(packageRoot, 'openclaw.plugin.json'), 'utf8'));
  const record = {
    source: 'path',
    spec: `local:${packageRoot}`,
    sourcePath: packageRoot,
    installPath: installRoot,
    version: manifest.version,
    resolvedName: manifest.name || 'OpenClawBrain',
    resolvedVersion: manifest.version,
    resolvedSpec: `local:${packageRoot}`,
    installedAt: new Date().toISOString(),
    sourceRepo: repoRoot,
  };
  const installs = existsSync(installsPath) ? JSON.parse(await readFile(installsPath, 'utf8')) : { installRecords: {} };
  if (installs && typeof installs === 'object' && installs.installRecords && typeof installs.installRecords === 'object') {
    installs.installRecords.openclawbrain = record;
    delete installs.openclawbrain;
  } else {
    installs.openclawbrain = {
      ...(installs.openclawbrain || {}),
      id: 'openclawbrain',
      name: 'OpenClawBrain',
      ...record,
    };
  }
  await writeFile(installsPath, `${JSON.stringify(installs, null, 2)}\n`);

  console.log(`Installed OpenClawBrain ${manifest.version} to ${installRoot}`);
}

async function resolveTargetHomes(argv) {
  const explicit = [];
  let allLocalHomes = false;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--') continue;
    if (arg === '--all-local-homes') {
      allLocalHomes = true;
      continue;
    }
    if (arg === '--home') {
      const value = argv[i + 1];
      if (!value) throw new Error('--home requires a path');
      explicit.push(resolveHome(value));
      i += 1;
      continue;
    }
    if (arg.startsWith('--home=')) {
      explicit.push(resolveHome(arg.slice('--home='.length)));
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  const homes = [];
  if (allLocalHomes) homes.push(...await discoverLocalOpenClawHomes());
  homes.push(...explicit);
  if (!homes.length) homes.push(resolveHome(process.env.OPENCLAW_HOME || '~/.openclaw'));
  return [...new Set(homes.map((home) => path.resolve(home)))];
}

async function discoverLocalOpenClawHomes() {
  const root = os.homedir();
  const entries = await readdir(root, { withFileTypes: true });
  const homes = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (entry.name !== '.openclaw' && !entry.name.startsWith('.openclaw-')) continue;
    const candidate = path.join(root, entry.name);
    if (
      existsSync(path.join(candidate, 'openclaw.json')) ||
      existsSync(path.join(candidate, 'extensions', 'openclawbrain')) ||
      existsSync(path.join(candidate, 'plugins'))
    ) {
      homes.push(candidate);
    }
  }
  return homes.sort();
}

function resolveHome(value) {
  if (!value) return path.join(os.homedir(), '.openclaw');
  if (value === '~') return os.homedir();
  if (value.startsWith('~/')) return path.join(os.homedir(), value.slice(2));
  return path.resolve(value);
}
