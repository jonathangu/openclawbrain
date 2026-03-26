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

function createProofFixture(t, options = {}) {
    const root = createTempRoot(t);
    const openclawHome = path.join(root, ".openclaw-Tern");
    const activationRoot = path.join(root, ".openclawbrain", "activation");
    const bundleDir = path.join(root, "artifacts", "proof-bundle");
    const gatewayLogPath = path.join(root, "gateway.log");
    const runtimeLoadProofPath = path.join(activationRoot, "attachment-truth", "runtime-load-proofs.json");
    mkdirSync(openclawHome, { recursive: true });
    writeFileSync(path.join(openclawHome, "openclaw.json"), JSON.stringify({
        profile: "Tern"
    }, null, 2));
    if (options.gatewayLogText !== null) {
        writeFileSync(gatewayLogPath, options.gatewayLogText ?? `${JSON.stringify({
            _meta: { date: "2999-01-01T00:00:00.000Z" },
            message: "[openclawbrain] BRAIN LOADED"
        })}\n`, "utf8");
    }
    if (options.runtimeLoadProofSnapshot !== null) {
        mkdirSync(path.dirname(runtimeLoadProofPath), { recursive: true });
        writeFileSync(runtimeLoadProofPath, `${JSON.stringify(options.runtimeLoadProofSnapshot ?? {
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
    }
    return {
        root,
        openclawHome,
        activationRoot,
        bundleDir,
        gatewayLogPath,
        runtimeLoadProofPath
    };
}

function createDetailedStatusText(fixture, overrides = {}) {
    return [
        overrides.statusLine ?? "STATUS ok",
        `target activation=${overrides.activationRoot ?? fixture.activationRoot} boundary=current_profile`,
        `attachTruth current=current_profile runtime=${overrides.runtimeTruth ?? "proven"} hook=present config=allows_load`,
        overrides.hookLine ?? "hook        install=installed loadable=loadable",
        `serve       state=${overrides.serveState ?? "serving_active_pack"}`,
        `routeFn     available=${overrides.routeFnAvailable ?? "yes"}`,
        overrides.pathLine ?? "path        source=materialized_candidate pg=v2 method=policy_gradient_v2 target=trajectory_reconstruction connect=4 trajectories=12 bindingQuality=exact_only",
        overrides.attributionLine ?? "attribution quality=exact_only source=latest_materialization/watch_snapshot nonZero=1 exact=1 heuristic=0 unmatched=0 ambiguous=0 modes=decision:1|digest:0|compile:0|heuristic:0",
        `attachedSet current_profile@${path.resolve(fixture.openclawHome)} proofPath=${overrides.proofPath ?? fixture.runtimeLoadProofPath} proofError=${overrides.proofError ?? "none"}`,
        `loadProof=${overrides.loadProof ?? "status_probe_ready"}`
    ].join("\n");
}

function createHealthyLabelOutputs(fixture, overrides = {}) {
    return {
        "install": { stdout: "install ok\n" },
        "gateway restart": { stdout: "restart ok\n" },
        "gateway status": {
            stdout: `Runtime: running\nRPC probe: ok\nFile logs: ${fixture.gatewayLogPath}\n`
        },
        "plugin inspect": {
            stdout: "Status: loaded\nSource: /tmp/node_modules/@openclawbrain/openclaw/dist/extension/index.js\n"
        },
        "detailed status": {
            stdout: createDetailedStatusText(fixture)
        },
        ...overrides
    };
}

function captureProofScenario(fixture, labelOutputs, overrides = {}) {
    const captures = [];
    const runCapture = (command, args, options = {}) => {
        captures.push({ command, args, label: options.label });
        const response = labelOutputs[options.label];
        if (response === undefined) {
            throw new Error(`unexpected label: ${options.label}`);
        }
        return makeCapture(command, args, typeof response === "function"
            ? response({ command, args, options, fixture, captures })
            : response);
    };
    const result = captureOperatorProofBundle({
        openclawHome: fixture.openclawHome,
        activationRoot: fixture.activationRoot,
        outputDir: fixture.bundleDir,
        timeoutMs: 2_000,
        pluginId: "openclawbrain",
        skipInstall: false,
        skipRestart: false,
        cliInvocation: {
            command: "openclawbrain",
            args: []
        },
        runCapture,
        cwd: fixture.root,
        ...overrides
    });
    return { result, captures };
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
    const fixture = createProofFixture(t);
    const { result, captures } = captureProofScenario(fixture, createHealthyLabelOutputs(fixture));
    assert.equal(result.verdict.verdict, "success_and_proven");
    assert.equal(result.gatewayProfile, "Tern");
    assert.equal(result.runtimeLoadProofPath, fixture.runtimeLoadProofPath);
    assert.ok(existsSync(path.join(fixture.bundleDir, "summary.md")));
    assert.ok(existsSync(path.join(fixture.bundleDir, "steps.json")));
    assert.ok(existsSync(path.join(fixture.bundleDir, "verdict.json")));
    assert.ok(existsSync(path.join(fixture.bundleDir, "runtime-load-proof.json")));
    assert.ok(existsSync(path.join(fixture.bundleDir, "extracted-startup-breadcrumbs.log")));
    assert.ok(existsSync(path.join(fixture.bundleDir, "01-install.stdout.log")));
    assert.ok(existsSync(path.join(fixture.bundleDir, "05-detailed-status.stdout.log")));
    const stepsPayload = JSON.parse(readFileSync(path.join(fixture.bundleDir, "steps.json"), "utf8"));
    const verdictPayload = JSON.parse(readFileSync(path.join(fixture.bundleDir, "verdict.json"), "utf8"));
    const summary = readFileSync(path.join(fixture.bundleDir, "summary.md"), "utf8");
    const breadcrumbsLog = readFileSync(path.join(fixture.bundleDir, "extracted-startup-breadcrumbs.log"), "utf8");
    const runtimeProofSnapshot = JSON.parse(readFileSync(path.join(fixture.bundleDir, "runtime-load-proof.json"), "utf8"));
    assert.equal(stepsPayload.gatewayProfile, "Tern");
    assert.equal(stepsPayload.steps.length, 5);
    assert.equal(verdictPayload.verdict.verdict, "success_and_proven");
    assert.match(summary, /bundle verdict: \*\*success_and_proven\*\*/);
    assert.match(summary, /## Learning Attribution/);
    assert.match(summary, /attribution quality=exact_only/);
    assert.match(summary, /path        source=materialized_candidate/);
    assert.match(summary, /## Warnings/);
    assert.match(summary, /- none/);
    assert.match(summary, /startup log contained a post-bundle \[openclawbrain\] BRAIN LOADED breadcrumb/);
    assert.match(breadcrumbsLog, /BRAIN LOADED/);
    assert.equal(runtimeProofSnapshot.exists, true);
    assert.equal(runtimeProofSnapshot.path, fixture.runtimeLoadProofPath);
    assert.equal(verdictPayload.attributionLine, "attribution quality=exact_only source=latest_materialization/watch_snapshot nonZero=1 exact=1 heuristic=0 unmatched=0 ambiguous=0 modes=decision:1|digest:0|compile:0|heuristic:0");
    assert.equal(verdictPayload.learningPathLine, "path        source=materialized_candidate pg=v2 method=policy_gradient_v2 target=trajectory_reconstruction connect=4 trajectories=12 bindingQuality=exact_only");
    assert.deepEqual(captures[1]?.args, ["gateway", "restart", "--profile", "Tern"]);
    assert.deepEqual(captures[2]?.args, ["gateway", "status", "--profile", "Tern"]);
});

test("proof capture reports warnings when proof support is incomplete but runtime truth is healthy", (t) => {
    const fixture = createProofFixture(t, {
        gatewayLogText: null,
        runtimeLoadProofSnapshot: null
    });
    const { result } = captureProofScenario(fixture, createHealthyLabelOutputs(fixture));
    const summary = readFileSync(path.join(fixture.bundleDir, "summary.md"), "utf8");
    assert.equal(result.verdict.verdict, "success_but_proof_incomplete");
    assert.equal(result.verdict.severity, "degraded");
    assert.deepEqual(result.verdict.missingProofs, [
        "startup_breadcrumb",
        "runtime_load_proof_record"
    ]);
    assert.deepEqual(result.verdict.warnings, [
        "startup log did not contain a post-bundle [openclawbrain] BRAIN LOADED breadcrumb",
        "runtime-load-proof snapshot was missing"
    ]);
    assert.match(summary, /bundle verdict: \*\*success_but_proof_incomplete\*\*/);
    assert.match(summary, /startup log did not contain a post-bundle \[openclawbrain\] BRAIN LOADED breadcrumb/);
    assert.match(summary, /runtime-load-proof snapshot was missing/);
});

test("proof capture treats ambiguous plugin inspection as a proof warning when status runtime truth stays healthy", (t) => {
    const fixture = createProofFixture(t);
    const { result } = captureProofScenario(fixture, createHealthyLabelOutputs(fixture, {
        "plugin inspect": {
            stdout: "Status: loaded\nSource: /tmp/plugins/openclawbrain-dev/index.js\n"
        }
    }));
    assert.equal(result.verdict.verdict, "success_but_proof_incomplete");
    assert.equal(result.verdict.severity, "degraded");
    assert.ok(result.verdict.missingProofs.includes("packaged_hook_path"));
    assert.ok(result.verdict.warnings.includes("plugin inspect did not confirm the packaged hook source"));
    assert.ok(!result.verdict.missingProofs.includes("runtime_proven"));
});

test("proof capture treats partial detailed-status proof failures as warnings when stdout still proves runtime truth", (t) => {
    const fixture = createProofFixture(t);
    const { result } = captureProofScenario(fixture, createHealthyLabelOutputs(fixture, {
        "detailed status": {
            stdout: createDetailedStatusText(fixture),
            exitCode: null,
            error: "Error: command timed out after 2000ms"
        }
    }));
    const detailedStatusStep = result.steps.find((step) => step.stepId === "05-detailed-status");
    assert.equal(result.verdict.verdict, "success_but_proof_incomplete");
    assert.ok(result.verdict.missingProofs.includes("step:05-detailed-status:timed_out:partial"));
    assert.ok(result.verdict.warnings.includes("05-detailed-status ended as timed_out with partial capture"));
    assert.equal(detailedStatusStep?.resultClass, "timed_out");
    assert.equal(detailedStatusStep?.captureState, "partial");
});

test("proof capture stays blocking when detailed status fails to prove runtime truth", (t) => {
    const fixture = createProofFixture(t);
    const { result } = captureProofScenario(fixture, createHealthyLabelOutputs(fixture, {
        "detailed status": {
            stdout: createDetailedStatusText(fixture, {
                statusLine: "STATUS degraded",
                runtimeTruth: "missing",
                serveState: "seed_only",
                routeFnAvailable: "no",
                loadProof: "pending"
            })
        }
    }));
    assert.equal(result.verdict.verdict, "degraded_or_failed_proof");
    assert.equal(result.verdict.severity, "blocking");
    assert.ok(result.verdict.missingProofs.includes("status_ok"));
    assert.ok(result.verdict.missingProofs.includes("load_proof"));
    assert.ok(result.verdict.missingProofs.includes("runtime_proven"));
    assert.ok(result.verdict.missingProofs.includes("serve_active_pack"));
    assert.ok(result.verdict.missingProofs.includes("route_fn"));
});
