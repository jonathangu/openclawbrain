import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildProofCommandForOpenClawHome, buildProofCommandHelpSection, captureOperatorProofBundle, parseProofCliArgs } from "../src/proof-command.js";

function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-proof-cli-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}

function makeCapture(command, args, overrides = {}) {
    return {
        command,
        argv: args,
        shellCommand: [command, ...args].join(" "),
        startedAt: "2026-03-23T01:00:00.000Z",
        endedAt: "2026-03-23T01:00:01.000Z",
        durationMs: 1_000,
        exitCode: 0,
        signal: null,
        stdout: "",
        stderr: "",
        error: null,
        ...overrides
    };
}

test("proof parser resolves paths and command strings for the public operator lane", (t) => {
    const root = createTempRoot(t);
    const openclawHome = path.join(root, ".openclaw Tern");
    mkdirSync(openclawHome, { recursive: true });
    const parsed = parseProofCliArgs([
        "--openclaw-home",
        openclawHome,
        "--activation-root",
        path.join(root, "activation"),
        "--output-dir",
        path.join(root, "proof-bundle"),
        "--skip-install",
        "--skip-restart",
        "--plugin-id",
        "openclawbrain-alt",
        "--timeout-ms",
        "1500",
        "--json"
    ]);
    assert.equal(parsed.command, "proof");
    assert.equal(parsed.openclawHome, path.resolve(openclawHome));
    assert.equal(parsed.activationRoot, path.resolve(root, "activation"));
    assert.equal(parsed.outputDir, path.resolve(root, "proof-bundle"));
    assert.equal(parsed.skipInstall, true);
    assert.equal(parsed.skipRestart, true);
    assert.equal(parsed.pluginId, "openclawbrain-alt");
    assert.equal(parsed.timeoutMs, 1500);
    assert.equal(parsed.json, true);
    assert.equal(buildProofCommandForOpenClawHome(openclawHome), `openclawbrain proof --openclaw-home '${path.resolve(openclawHome)}'`);
});

test("proof help stays discoverable without requiring an OpenClaw home", () => {
    const parsed = parseProofCliArgs(["--help"]);
    const help = buildProofCommandHelpSection();
    assert.equal(parsed.command, "proof");
    assert.equal(parsed.help, true);
    assert.equal(parsed.openclawHome, "");
    assert.match(help.usage, /openclawbrain proof --openclaw-home <path>/);
    assert.match(help.lifecycle, /durable operator proof bundle/);
    assert.match(help.advanced, /startup breadcrumbs/);
});

test("proof capture writes one durable bundle with proof artifacts and profile-scoped gateway steps", (t) => {
    const root = createTempRoot(t);
    const openclawHome = path.join(root, ".openclaw-Tern");
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    const bundleDir = path.join(root, "artifacts", "proof-bundle");
    const gatewayLogPath = path.join(root, "gateway.log");
    const runtimeLoadProofPath = path.join(activationRoot, "attachment-truth", "runtime-load-proofs.json");
    mkdirSync(openclawHome, { recursive: true });
    mkdirSync(path.dirname(runtimeLoadProofPath), { recursive: true });
    writeFileSync(path.join(openclawHome, "openclaw.json"), JSON.stringify({
        profile: "Tern"
    }, null, 2));
    writeFileSync(gatewayLogPath, `${JSON.stringify({
        _meta: { date: "2999-01-01T00:00:00.000Z" },
        message: "[openclawbrain] BRAIN LOADED"
    })}\n`, "utf8");
    writeFileSync(runtimeLoadProofPath, `${JSON.stringify({
        contract: "openclaw_profile_runtime_load_proofs.v1",
        runtimeOwner: "openclaw",
        activationRoot,
        updatedAt: "2026-03-23T01:00:05.000Z",
        profiles: [
            {
                openclawHome: path.resolve(openclawHome),
                loadedAt: "2026-03-23T01:00:05.000Z"
            }
        ]
    }, null, 2)}\n`, "utf8");
    const captures = [];
    const runCapture = (command, args, options = {}) => {
        captures.push({ command, args, label: options.label });
        switch (options.label) {
            case "install":
                return makeCapture(command, args, { stdout: "install ok\n" });
            case "gateway restart":
                return makeCapture(command, args, { stdout: "restart ok\n" });
            case "gateway status":
                return makeCapture(command, args, {
                    stdout: `Runtime: running\nRPC probe: ok\nFile logs: ${gatewayLogPath}\n`
                });
            case "plugin inspect":
                return makeCapture(command, args, {
                    stdout: "Status: loaded\nSource: /tmp/node_modules/@openclawbrain/openclaw/dist/extension/index.js\n"
                });
            case "detailed status":
                return makeCapture(command, args, {
                    stdout: [
                        "STATUS ok",
                        `target activation=${activationRoot} boundary=current_profile`,
                        "attachTruth current=current_profile runtime=proven hook=present config=allows_load",
                        "hook        install=installed loadable=loadable",
                        "serve       state=serving_active_pack",
                        "routeFn     available=yes",
                        `attachedSet current_profile@${path.resolve(openclawHome)} proofPath=${runtimeLoadProofPath} proofError=none`,
                        "loadProof=status_probe_ready"
                    ].join("\n")
                });
            default:
                throw new Error(`unexpected label: ${options.label}`);
        }
    };
    const result = captureOperatorProofBundle({
        openclawHome,
        activationRoot,
        outputDir: bundleDir,
        timeoutMs: 2_000,
        pluginId: "openclawbrain",
        skipInstall: false,
        skipRestart: false,
        cliInvocation: {
            command: "openclawbrain",
            args: []
        },
        runCapture,
        cwd: root
    });
    assert.equal(result.verdict.verdict, "success_and_proven");
    assert.equal(result.gatewayProfile, "Tern");
    assert.ok(existsSync(path.join(bundleDir, "summary.md")));
    assert.ok(existsSync(path.join(bundleDir, "steps.json")));
    assert.ok(existsSync(path.join(bundleDir, "verdict.json")));
    assert.ok(existsSync(path.join(bundleDir, "runtime-load-proof.json")));
    assert.ok(existsSync(path.join(bundleDir, "extracted-startup-breadcrumbs.log")));
    assert.ok(existsSync(path.join(bundleDir, "01-install.stdout.log")));
    assert.ok(existsSync(path.join(bundleDir, "05-detailed-status.stdout.log")));
    const stepsPayload = JSON.parse(readFileSync(path.join(bundleDir, "steps.json"), "utf8"));
    const verdictPayload = JSON.parse(readFileSync(path.join(bundleDir, "verdict.json"), "utf8"));
    const summary = readFileSync(path.join(bundleDir, "summary.md"), "utf8");
    const breadcrumbsLog = readFileSync(path.join(bundleDir, "extracted-startup-breadcrumbs.log"), "utf8");
    const runtimeProofSnapshot = JSON.parse(readFileSync(path.join(bundleDir, "runtime-load-proof.json"), "utf8"));
    assert.equal(stepsPayload.gatewayProfile, "Tern");
    assert.equal(stepsPayload.steps.length, 5);
    assert.equal(verdictPayload.verdict.verdict, "success_and_proven");
    assert.match(summary, /bundle verdict: \*\*success_and_proven\*\*/);
    assert.match(summary, /startup log contained a post-bundle \[openclawbrain\] BRAIN LOADED breadcrumb/);
    assert.match(breadcrumbsLog, /BRAIN LOADED/);
    assert.equal(runtimeProofSnapshot.exists, true);
    assert.equal(runtimeProofSnapshot.path, runtimeLoadProofPath);
    assert.deepEqual(captures[1]?.args, ["gateway", "restart", "--profile", "Tern"]);
    assert.deepEqual(captures[2]?.args, ["gateway", "status", "--profile", "Tern"]);
});
