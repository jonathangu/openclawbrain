import test from "node:test";
import assert from "node:assert/strict";
import { describeOpenClawBrainHotfixBoundary, summarizeOpenClawBrainHookLoad } from "../src/openclaw-hook-truth.js";

function buildInspection(overrides = {}) {
    return {
        scope: "exact_openclaw_home",
        openclawHome: "/tmp/.openclaw",
        extensionDir: "/tmp/.openclaw/extensions/openclawbrain",
        hookPath: "/tmp/.openclaw/extensions/openclawbrain/index.js",
        runtimeGuardPath: "/tmp/.openclaw/extensions/openclawbrain/runtime-guard.js",
        manifestPath: "/tmp/.openclaw/extensions/openclawbrain/openclaw.plugin.json",
        packageJsonPath: "/tmp/.openclaw/extensions/openclawbrain/package.json",
        manifestId: "openclawbrain",
        installId: "openclawbrain",
        packageName: "@openclawbrain/openclaw",
        packageVersion: "1.2.3",
        installLayout: "native_package_plugin",
        additionalInstallCount: 0,
        installState: "installed",
        loadability: "loadable",
        pluginAllowlistState: "allowed",
        desynced: false,
        detail: "profile hook is installed via native package plugin",
        ...overrides,
    };
}

test("openclaw hook load summary reports no-op safety when the hook is loadable", () => {
    const summary = summarizeOpenClawBrainHookLoad(buildInspection(), true);
    assert.equal(summary.guardSeverity, "none");
    assert.equal(summary.guardActionability, "none");
    assert.equal(summary.guardSummary, "profile hook is installed and loadable");
    assert.equal(summary.guardAction, "none");
    assert.equal(summary.loadProof, "status_probe_ready");
});

test("openclaw hotfix boundary marks daemon and installed hook as separate surfaces even on matching versions", () => {
    const boundary = describeOpenClawBrainHotfixBoundary({
        hookInspection: buildInspection(),
        daemonInspection: {
            configuredRuntimePath: "/tmp/openclawbrain/cli.js",
            configuredRuntimePackageName: "@openclawbrain/cli",
            configuredRuntimePackageVersion: "1.2.3",
            configuredRuntimePackageSpec: null,
        },
    });
    assert.equal(boundary.boundary, "split_surfaces");
    assert.equal(boundary.skew, "split_path_same_version");
    assert.equal(boundary.convergeState, "converged");
    assert.match(boundary.detail, /daemon background watch runs from/);
    assert.match(boundary.guidance, /Patch the daemon runtime path/);
});

test("openclaw hotfix boundary calls out version skew when daemon and installed hook diverge", () => {
    const boundary = describeOpenClawBrainHotfixBoundary({
        hookInspection: buildInspection(),
        daemonInspection: {
            configuredRuntimePath: "/tmp/openclawbrain/cli.js",
            configuredRuntimePackageName: "@openclawbrain/cli",
            configuredRuntimePackageVersion: "9.9.9",
            configuredRuntimePackageSpec: null,
        },
    });
    assert.equal(boundary.boundary, "split_surfaces");
    assert.equal(boundary.skew, "split_path_version_skew");
    assert.equal(boundary.convergeState, "half_converged");
});

test("openclaw hotfix boundary treats a blocked installed hook as half-converged", () => {
    const boundary = describeOpenClawBrainHotfixBoundary({
        hookInspection: buildInspection({
            loadability: "blocked",
            desynced: true,
            detail: "profile hook is present but OpenClaw will not load it",
        }),
        daemonInspection: {
            configuredRuntimePath: "/tmp/openclawbrain/cli.js",
            configuredRuntimePackageName: "@openclawbrain/cli",
            configuredRuntimePackageVersion: "1.2.3",
            configuredRuntimePackageSpec: null,
        },
    });
    assert.equal(boundary.skew, "split_path_same_version");
    assert.equal(boundary.convergeState, "half_converged");
});
