import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const daemonSourcePath = path.join(__dirname, "..", "src", "daemon.js");
const daemonSource = readFileSync(daemonSourcePath, "utf8").replace(
  'from "./index.js"',
  'from "./index.stub.js"',
);

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function withProcessState(state, fn) {
  const previousHome = process.env.HOME;
  const previousArgv = [...process.argv];
  process.env.HOME = state.homeDir;
  process.argv = [process.execPath, state.argv1];
  try {
    return await fn();
  } finally {
    if (previousHome === undefined) {
      delete process.env.HOME;
    } else {
      process.env.HOME = previousHome;
    }
    process.argv = previousArgv;
  }
}

function captureConsole(fn) {
  const stdout = [];
  const stderr = [];
  const originalLog = console.log;
  const originalError = console.error;
  console.log = (...args) => stdout.push(args.join(" "));
  console.error = (...args) => stderr.push(args.join(" "));
  try {
    return {
      result: fn(),
      stdout: stdout.join("\n"),
      stderr: stderr.join("\n"),
    };
  } finally {
    console.log = originalLog;
    console.error = originalError;
  }
}

function createCommandRunner(overrides = {}) {
  return (command) => {
    if (command.startsWith("launchctl load -w ")) {
      return "";
    }
    if (command === "command -v launchctl") {
      return "/bin/launchctl\n";
    }
    const result = overrides[command];
    if (result instanceof Error) {
      throw result;
    }
    if (result !== undefined) {
      return result;
    }
    throw new Error(`Unexpected daemon command: ${command}`);
  };
}

function createDaemonFixture(t, options = {}) {
  const root = mkdtempSync(path.join(tmpdir(), "ocb-daemon-test-"));
  const packageRoot = path.join(root, "package");
  const distSrc = path.join(packageRoot, "dist", "src");
  mkdirSync(distSrc, { recursive: true });
  writeFileSync(
    path.join(packageRoot, "package.json"),
    JSON.stringify(
      {
        name: options.packageName ?? "@openclawbrain/cli",
        version: options.packageVersion ?? "1.2.3",
        type: "module",
      },
      null,
      2,
    ),
  );
  writeFileSync(path.join(distSrc, "daemon.js"), daemonSource, "utf8");
  writeFileSync(
    path.join(distSrc, "index.stub.js"),
    [
      "export function loadTeacherSurface() { return null; }",
      "export function resolveWatchSessionTailCursorPath() { return null; }",
      "export function resolveWatchStateRoot() { return null; }",
      "export function resolveWatchTeacherSnapshotPath() { return null; }",
      "",
    ].join("\n"),
    "utf8",
  );
  if (options.withCliScript) {
    writeFileSync(path.join(distSrc, "cli.js"), "#!/usr/bin/env node\n", "utf8");
  }
  t.after(() => {
    rmSync(root, { recursive: true, force: true });
  });
  return {
    root,
    packageRoot,
    distSrc,
    cliScriptPath: path.join(distSrc, "cli.js"),
  };
}

async function importFixtureDaemon(fixture) {
  return import(`${pathToFileURL(path.join(fixture.distSrc, "daemon.js")).href}?ts=${Date.now()}-${Math.random()}`);
}

test("daemon start emits a pinned npm exec launch command instead of an _npx cache path", async (t) => {
  const fixture = createDaemonFixture(t, { withCliScript: false, packageVersion: "9.9.9" });
  const homeDir = path.join(fixture.root, "home");
  const activationRoot = path.join(fixture.root, "activation-root");
  const npxCliPath = path.join(
    fixture.root,
    ".npm",
    "_npx",
    "abc123",
    "node_modules",
    "@openclawbrain",
    "cli",
    "dist",
    "src",
    "cli.js",
  );
  const npxBinPath = path.join(
    fixture.root,
    ".npm",
    "_npx",
    "abc123",
    "node_modules",
    ".bin",
    "openclawbrain",
  );
  mkdirSync(path.dirname(npxCliPath), { recursive: true });
  mkdirSync(path.dirname(npxBinPath), { recursive: true });
  mkdirSync(homeDir, { recursive: true });
  mkdirSync(activationRoot, { recursive: true });
  writeFileSync(npxCliPath, "#!/usr/bin/env node\n", "utf8");
  writeFileSync(npxBinPath, "#!/usr/bin/env node\n", "utf8");

  const daemon = await importFixtureDaemon(fixture);

  await withProcessState({ homeDir, argv1: npxCliPath }, async () => {
    const serviceIdentity = daemon.buildDaemonServiceIdentity(activationRoot);
    daemon.setDaemonCommandRunnerForTesting(
      createCommandRunner({
        "which -a openclawbrain": `${npxBinPath}\n`,
        "which -a npm": "/usr/local/bin/npm\n",
        "launchctl list": `- 0 ${serviceIdentity.label}\n`,
      }),
    );
    t.after(() => daemon.setDaemonCommandRunnerForTesting(null));

    const startOutcome = captureConsole(() => daemon.daemonStart(activationRoot, false));
    assert.equal(startOutcome.result, 0);
    assert.equal(startOutcome.stderr, "");

    const plist = readFileSync(serviceIdentity.plistPath, "utf8");
    assert.match(plist, /--package=@openclawbrain\/cli@9\.9\.9/);
    assert.doesNotMatch(plist, /_npx/);

    const statusOutcome = captureConsole(() => daemon.daemonStatus(activationRoot, true));
    assert.equal(statusOutcome.stderr, "");
    const statusPayload = JSON.parse(statusOutcome.stdout);
    assert.equal(statusPayload.configuredProgramArguments[0], "/usr/local/bin/npm");
    assert.deepEqual(statusPayload.configuredProgramArguments.slice(1, 6), [
      "exec",
      "--yes",
      "--package=@openclawbrain/cli@9.9.9",
      "--",
      "openclawbrain",
    ]);
    assert.equal(statusPayload.configuredRuntimePackageSpec, "@openclawbrain/cli@9.9.9");
    assert.equal(statusPayload.configuredRuntimeLooksEphemeral, false);
    assert.equal(statusPayload.hotfixBoundary.surface, "daemon_runtime");
    assert.equal(statusPayload.hotfixBoundary.separateFromInstalledHookSurface, true);
    assert.match(statusPayload.hotfixBoundary.guidance, /status --openclaw-home <path> --detailed/);
  });
});

test("daemon status shows the configured program, args, and command for a durable local cli.js runtime", async (t) => {
  const fixture = createDaemonFixture(t, { withCliScript: true, packageVersion: "3.4.5" });
  const homeDir = path.join(fixture.root, "home");
  const activationRoot = path.join(fixture.root, "activation-root");
  const npxCliPath = path.join(
    fixture.root,
    ".npm",
    "_npx",
    "def456",
    "node_modules",
    "@openclawbrain",
    "cli",
    "dist",
    "src",
    "cli.js",
  );
  mkdirSync(path.dirname(npxCliPath), { recursive: true });
  mkdirSync(homeDir, { recursive: true });
  mkdirSync(activationRoot, { recursive: true });
  writeFileSync(npxCliPath, "#!/usr/bin/env node\n", "utf8");

  const daemon = await importFixtureDaemon(fixture);

  await withProcessState({ homeDir, argv1: npxCliPath }, async () => {
    const serviceIdentity = daemon.buildDaemonServiceIdentity(activationRoot);
    const durableCliPath = realpathSync(fixture.cliScriptPath);
    daemon.setDaemonCommandRunnerForTesting(
      createCommandRunner({
        "which -a openclawbrain": "",
        "which -a npm": "/usr/local/bin/npm\n",
        "launchctl list": `123 0 ${serviceIdentity.label}\n`,
      }),
    );
    t.after(() => daemon.setDaemonCommandRunnerForTesting(null));

    const startOutcome = captureConsole(() => daemon.daemonStart(activationRoot, false));
    assert.equal(startOutcome.result, 0);

    const plist = readFileSync(serviceIdentity.plistPath, "utf8");
    assert.match(plist, new RegExp(escapeRegex(durableCliPath)));
    assert.doesNotMatch(plist, /--package=@openclawbrain\/cli@3\.4\.5/);

    const statusOutcome = captureConsole(() => daemon.daemonStatus(activationRoot, false));
    assert.equal(statusOutcome.stderr, "");
    assert.match(statusOutcome.stdout, new RegExp(`Program: ${escapeRegex(process.execPath)}`));
    assert.match(
      statusOutcome.stdout,
      new RegExp(`Args: ${escapeRegex(durableCliPath)} watch --activation-root ${escapeRegex(activationRoot)}`),
    );
    assert.match(
      statusOutcome.stdout,
      new RegExp(`Command: ${escapeRegex(process.execPath)} ${escapeRegex(durableCliPath)} watch --activation-root ${escapeRegex(activationRoot)}`),
    );
    assert.match(statusOutcome.stdout, /Runtime surface: daemon watch\/learner runtime/);
    assert.match(statusOutcome.stdout, /Hotfix boundary: Patch this daemon runtime path for background watch\/learner fixes/);
    assert.doesNotMatch(statusOutcome.stdout, /_npx/);
  });
});

test("ensureManagedLearnerServiceForActivationRoot refreshes a running legacy compatibility daemon runtime", async (t) => {
  const fixture = createDaemonFixture(t, { withCliScript: true, packageVersion: "4.5.6" });
  const homeDir = path.join(fixture.root, "home");
  const activationRoot = path.join(fixture.root, "activation-root");
  const legacyRoot = path.join(fixture.root, "legacy-compat");
  const legacyBinPath = path.join(legacyRoot, "bin", "openclawbrain.js");
  mkdirSync(path.dirname(legacyBinPath), { recursive: true });
  mkdirSync(homeDir, { recursive: true });
  mkdirSync(activationRoot, { recursive: true });
  writeFileSync(
    path.join(legacyRoot, "package.json"),
    JSON.stringify({ name: "@jonathangu/openclawbrain", version: "0.3.8", type: "module" }, null, 2),
  );
  writeFileSync(legacyBinPath, "#!/usr/bin/env node\n", "utf8");

  const daemon = await importFixtureDaemon(fixture);

  await withProcessState({ homeDir, argv1: fixture.cliScriptPath }, async () => {
    const serviceIdentity = daemon.buildDaemonServiceIdentity(activationRoot);
    mkdirSync(path.dirname(serviceIdentity.plistPath), { recursive: true });
    writeFileSync(
      serviceIdentity.plistPath,
      `<?xml version="1.0" encoding="UTF-8"?>\n` +
        `<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n` +
        `<plist version="1.0"><dict>\n` +
        `  <key>Label</key><string>${serviceIdentity.label}</string>\n` +
        `  <key>ProgramArguments</key><array>\n` +
        `    <string>${legacyBinPath}</string>\n` +
        `    <string>watch</string>\n` +
        `    <string>--activation-root</string>\n` +
        `    <string>${activationRoot}</string>\n` +
        `  </array>\n` +
        `  <key>WorkingDirectory</key><string>${activationRoot}</string>\n` +
        `  <key>StandardOutPath</key><string>${serviceIdentity.logPath}</string>\n` +
        `  <key>StandardErrorPath</key><string>${serviceIdentity.logPath}</string>\n` +
        `</dict></plist>\n`,
      "utf8",
    );

    daemon.setDaemonCommandRunnerForTesting(
      createCommandRunner({
        "command -v launchctl": "/bin/launchctl\n",
        "launchctl list": `123 0 ${serviceIdentity.label}\n`,
        [`launchctl unload ${JSON.stringify(serviceIdentity.plistPath)}`]: "",
        [`launchctl load -w ${JSON.stringify(serviceIdentity.plistPath)}`]: "",
      }),
    );
    t.after(() => daemon.setDaemonCommandRunnerForTesting(null));

    const ensureResult = daemon.ensureManagedLearnerServiceForActivationRoot(activationRoot);
    assert.equal(ensureResult.state, "refreshed");
    assert.equal(ensureResult.reason, "legacy_compat_runtime");
    assert.match(ensureResult.detail, /retired compatibility package/);

    const refreshedPlist = readFileSync(serviceIdentity.plistPath, "utf8");
    assert.match(refreshedPlist, /dist\/src\/cli\.js/);
    assert.doesNotMatch(refreshedPlist, /legacy-compat/);
  });
});
