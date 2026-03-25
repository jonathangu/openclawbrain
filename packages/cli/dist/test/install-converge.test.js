import test from "node:test";
import assert from "node:assert/strict";
import { buildOpenClawBrainConvergeRestartPlan, classifyOpenClawBrainConvergeVerification, describeOpenClawBrainConvergeChangeReasons, diffOpenClawBrainConvergeRuntimeFingerprint, finalizeOpenClawBrainConvergeResult, planOpenClawBrainConvergePluginAction } from "../src/install-converge.js";

test("converge plans plugin install when the authoritative split-package plugin is absent", () => {
    const plan = planOpenClawBrainConvergePluginAction({
        selectedInstall: null
    });
    assert.equal(plan.action, "install");
    assert.equal(plan.packageSpec, "@openclawbrain/openclaw");
    assert.match(plan.reason, /no authoritative OpenClawBrain plugin install/);
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
        pluginsConfig: "{\"allow\":[\"openclawbrain\"]}"
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
