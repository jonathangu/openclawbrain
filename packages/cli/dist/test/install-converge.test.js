import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildOpenClawBrainConvergeRestartPlan, classifyOpenClawBrainConvergeVerification, describeOpenClawBrainConvergeChangeReasons, diffOpenClawBrainConvergeRuntimeFingerprint, finalizeOpenClawBrainConvergeResult, planOpenClawBrainConvergePluginAction, shouldReplaceOpenClawBrainInstallBeforeConverge } from "../src/install-converge.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test("converge plans plugin install when the authoritative split-package plugin is absent", () => {
    const plan = planOpenClawBrainConvergePluginAction({
        selectedInstall: null
    });
    assert.equal(plan.action, "install");
    assert.equal(plan.packageSpec, "@openclawbrain/openclaw");
    assert.match(plan.reason, /no authoritative OpenClawBrain plugin install/);
});

test("converge knows when an existing non-native install must be replaced before migrating onto the canonical plugin lane", () => {
    assert.equal(shouldReplaceOpenClawBrainInstallBeforeConverge({ selectedInstall: null }), false);
    assert.equal(shouldReplaceOpenClawBrainInstallBeforeConverge({
        selectedInstall: {
            packageName: "@openclawbrain/openclaw"
        }
    }), false);
    assert.equal(shouldReplaceOpenClawBrainInstallBeforeConverge({
        selectedInstall: {
            packageName: "@jonathangu/openclawbrain"
        }
    }), true);
    assert.equal(shouldReplaceOpenClawBrainInstallBeforeConverge({
        selectedInstall: {
            packageName: "openclawbrain"
        }
    }), true);
});

test("converge diff and restart plan skip restart when no runtime-affecting install state changed", () => {
    const before = {
        selectedInstall: {
            extensionDir: "/tmp/openclaw/extensions/@openclawbrain/openclaw",
            manifestId: "openclawbrain",
            installId: "openclaw",
            packageName: "@openclawbrain/openclaw"
        },
        installLayout: "native_package_plugin",
        hookPath: "/tmp/openclaw/extensions/@openclawbrain/openclaw/dist/extension/index.js",
        hookState: "installed",
        loadability: "loadable",
        activationRoot: "/tmp/.openclawbrain/activation",
        loaderSource: "const ACTIVATION_ROOT = \"/tmp/.openclawbrain/activation\";",
        runtimeGuardSource: "guard",
        pluginsConfig: "{\"allow\":[\"openclawbrain\"]}",
        daemonRuntimePath: "/tmp/openclawbrain/cli.js",
        daemonRuntimePackageName: "@openclawbrain/cli",
        daemonRuntimePackageVersion: "1.2.3",
        daemonRuntimePackageSpec: null,
        daemonRuntimeSource: "managed_service"
    };
    const diff = diffOpenClawBrainConvergeRuntimeFingerprint(before, {
        ...before
    });
    const restart = buildOpenClawBrainConvergeRestartPlan({
        profileName: "Tern",
        changeReasons: diff.reasons
    });
    assert.equal(diff.changed, false);
    assert.equal(describeOpenClawBrainConvergeChangeReasons(diff.reasons), "no runtime-affecting install changes detected");
    assert.equal(restart.required, false);
    assert.equal(restart.automatic, false);
});

test("converge diff tracks daemon runtime changes without forcing a gateway restart", () => {
    const before = {
        selectedInstall: {
            extensionDir: "/tmp/openclaw/extensions/@openclawbrain/openclaw",
            manifestId: "openclawbrain",
            installId: "openclaw",
            packageName: "@openclawbrain/openclaw"
        },
        installLayout: "native_package_plugin",
        hookPath: "/tmp/openclaw/extensions/@openclawbrain/openclaw/dist/extension/index.js",
        hookState: "installed",
        loadability: "loadable",
        activationRoot: "/tmp/.openclawbrain/activation",
        loaderSource: "const ACTIVATION_ROOT = \"/tmp/.openclawbrain/activation\";",
        runtimeGuardSource: "guard",
        pluginsConfig: "{\"allow\":[\"openclawbrain\"]}",
        daemonRuntimePath: "/tmp/openclawbrain/cli-old.js",
        daemonRuntimePackageName: "@openclawbrain/cli",
        daemonRuntimePackageVersion: "1.2.2",
        daemonRuntimePackageSpec: null,
        daemonRuntimeSource: "managed_service"
    };
    const diff = diffOpenClawBrainConvergeRuntimeFingerprint(before, {
        ...before,
        daemonRuntimePath: "/tmp/openclawbrain/cli.js",
        daemonRuntimePackageVersion: "1.2.3"
    });
    const restart = buildOpenClawBrainConvergeRestartPlan({
        profileName: "Tern",
        changeReasons: diff.reasons
    });
    assert.equal(diff.changed, true);
    assert.ok(diff.reasons.includes("daemon_runtime_path"));
    assert.ok(diff.reasons.includes("daemon_runtime_identity"));
    assert.equal(restart.required, false);
    assert.equal(restart.reason, "daemon_only_runtime_changes");
    assert.match(restart.detail, /daemon runtime surface/);
});

test("converge classifies manual-action and warning outcomes truthfully from status facts", () => {
    const manualVerification = classifyOpenClawBrainConvergeVerification({
        installLayout: "generated_shadow_extension",
        installState: "installed",
        loadability: "loadable",
        displayedStatus: "ok",
        runtimeLoad: "proven",
        loadProof: "status_probe_ready",
        serveState: "serving_active_pack",
        routeFnAvailable: true,
        awaitingFirstExport: false,
        restartRequired: true,
        restartPerformed: false
    });
    const manualVerdict = finalizeOpenClawBrainConvergeResult({
        stepFailure: null,
        verification: manualVerification,
        warnings: []
    });
    assert.equal(manualVerification.state, "failed");
    assert.equal(manualVerdict.verdict, "manual_action_required");
    assert.match(manualVerdict.why, /split-package native package plugin is not authoritative/);
    const halfConvergedVerification = classifyOpenClawBrainConvergeVerification({
        installLayout: "native_package_plugin",
        installState: "installed",
        loadability: "loadable",
        displayedStatus: "ok",
        runtimeLoad: "proven",
        loadProof: "status_probe_ready",
        serveState: "serving_active_pack",
        routeFnAvailable: true,
        awaitingFirstExport: false,
        restartRequired: false,
        restartPerformed: false,
        surfaceBoundary: "split_surfaces",
        surfaceSkew: "split_path_version_skew",
        surfaceConvergeState: "half_converged",
        selectedOpenClawHome: "/tmp/.openclaw-Tern",
        daemonPackage: "@openclawbrain/cli@1.2.2",
        hookPackage: "@openclawbrain/openclaw@1.2.3"
    });
    const halfConvergedVerdict = finalizeOpenClawBrainConvergeResult({
        stepFailure: null,
        verification: halfConvergedVerification,
        warnings: []
    });
    assert.equal(halfConvergedVerification.state, "failed");
    assert.equal(halfConvergedVerdict.verdict, "manual_action_required");
    assert.match(halfConvergedVerdict.why, /half-converged/);
    assert.match(halfConvergedVerdict.why, /split_surfaces\/split_path_version_skew/);
    const restartWarningVerification = classifyOpenClawBrainConvergeVerification({
        installLayout: "native_package_plugin",
        installState: "installed",
        loadability: "loadable",
        displayedStatus: "ok",
        runtimeLoad: "proven",
        loadProof: "status_probe_ready",
        serveState: "serving_active_pack",
        routeFnAvailable: true,
        awaitingFirstExport: false,
        restartRequired: true,
        restartPerformed: false
    });
    const restartWarningVerdict = finalizeOpenClawBrainConvergeResult({
        stepFailure: null,
        verification: restartWarningVerification,
        warnings: []
    });
    assert.equal(restartWarningVerification.state, "warning");
    assert.equal(restartWarningVerdict.verdict, "converged_with_warnings");
    assert.match(restartWarningVerdict.why, /automatic restart was not performed/);
    const restartWarningWithStatusWarn = classifyOpenClawBrainConvergeVerification({
        installLayout: "native_package_plugin",
        installState: "installed",
        loadability: "loadable",
        displayedStatus: "warn",
        runtimeLoad: "proven",
        loadProof: "status_probe_ready",
        serveState: "serving_active_pack",
        routeFnAvailable: true,
        awaitingFirstExport: true,
        restartRequired: true,
        restartPerformed: false
    });
    const restartWarningWithStatusWarnVerdict = finalizeOpenClawBrainConvergeResult({
        stepFailure: null,
        verification: restartWarningWithStatusWarn,
        warnings: []
    });
    assert.equal(restartWarningWithStatusWarn.state, "warning");
    assert.equal(restartWarningWithStatusWarnVerdict.verdict, "converged_with_warnings");
    // When runtime proof is green (proven + status_probe_ready), proof-promotion
    // skips the stale "status is warn" warning. The remaining warnings are the
    // restart-not-performed note and awaiting-first-export.
    assert.match(restartWarningWithStatusWarnVerdict.why, /automatic restart was not performed/);
    assert.match(restartWarningWithStatusWarnVerdict.why, /first export/);
    const unverifiedSurfaceVerification = classifyOpenClawBrainConvergeVerification({
        installLayout: "native_package_plugin",
        installState: "installed",
        loadability: "loadable",
        displayedStatus: "ok",
        runtimeLoad: "proven",
        loadProof: "status_probe_ready",
        serveState: "serving_active_pack",
        routeFnAvailable: true,
        awaitingFirstExport: false,
        restartRequired: false,
        restartPerformed: false,
        surfaceConvergeState: "unverified"
    });
    assert.equal(unverifiedSurfaceVerification.state, "warning");
    assert.match(unverifiedSurfaceVerification.warnings.join("; "), /not fully proven/);
    const warningVerification = classifyOpenClawBrainConvergeVerification({
        installLayout: "native_package_plugin",
        installState: "installed",
        loadability: "loadable",
        displayedStatus: "ok",
        runtimeLoad: "not_proven",
        loadProof: "status_probe_ready",
        serveState: "seed_state_authoritative",
        routeFnAvailable: false,
        awaitingFirstExport: true,
        restartRequired: false,
        restartPerformed: false
    });
    const warningVerdict = finalizeOpenClawBrainConvergeResult({
        stepFailure: null,
        verification: warningVerification,
        warnings: []
    });
    assert.equal(warningVerification.state, "warning");
    assert.equal(warningVerdict.verdict, "converged_with_warnings");
    assert.match(warningVerdict.why, /runtime load is not_proven/);
    assert.match(warningVerdict.why, /serve state is seed_state_authoritative/);
});

test("converge fails loudly when daemon and installed hook surfaces are half-converged", () => {
    const verification = classifyOpenClawBrainConvergeVerification({
        installLayout: "native_package_plugin",
        installState: "installed",
        loadability: "loadable",
        displayedStatus: "ok",
        runtimeLoad: "proven",
        loadProof: "status_probe_ready",
        serveState: "serving_active_pack",
        routeFnAvailable: true,
        awaitingFirstExport: false,
        surfaceBoundary: "split_surfaces",
        surfaceConvergeState: "half_converged",
        surfaceSkew: "split_path_version_skew",
        daemonPackage: "@openclawbrain/cli@0.4.29",
        hookPackage: "@openclawbrain/openclaw@0.4.28",
        selectedOpenClawHome: "/tmp/.openclaw-Tern",
        restartRequired: false,
        restartPerformed: false
    });
    const verdict = finalizeOpenClawBrainConvergeResult({
        stepFailure: null,
        verification,
        warnings: []
    });
    assert.equal(verification.state, "failed");
    assert.equal(verdict.verdict, "manual_action_required");
    assert.match(verdict.why, /half-converged/);
    assert.match(verdict.why, /split_path_version_skew/);
});

test("install writes a live runtime config manifest instead of an empty stub", () => {
    const cliSource = readFileSync(path.join(__dirname, "..", "src", "cli.js"), "utf8");

    const requiredLiterals = [
        "brainEmbeddingBaseUrl",
        "brainMaxCompileMs",
        "brainBudgetFraction",
        "brainMaxHops",
        "brainMaxFanoutPerNode",
        "brainMaxFrontierSize",
        "brainMaxSeeds",
        "brainSemanticThreshold",
        "brainShadowMode",
        "brainWorkerMode",
        "brainWorkerHeartbeatTimeoutMs",
        "brainWorkerRestartDelayMs",
    ];

    for (const literal of requiredLiterals) {
        assert.match(cliSource, new RegExp(`\\b${literal}\\b`));
    }

    assert.doesNotMatch(
        cliSource,
        /function buildExtensionPluginManifest\(\) \{[\s\S]*properties:\s*\{\}[\s\S]*\}/,
    );

    assert.match(cliSource, /function readOpenClawRuntimePackageMetadata\(/);
    assert.match(cliSource, /\[runtimePackageMetadata\.name\]: runtimePackageMetadata\.version/);
});
