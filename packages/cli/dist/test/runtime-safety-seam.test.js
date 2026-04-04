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

test("hook load summary reports no-op safety when the hook is loadable", () => {
    const summary = summarizeOpenClawBrainHookLoad(buildInspection(), true);
    assert.equal(summary.guardSeverity, "none");
    assert.equal(summary.guardActionability, "none");
    assert.equal(summary.guardSummary, "profile hook is installed and loadable");
    assert.equal(summary.guardAction, "none");
    assert.equal(summary.loadProof, "status_probe_ready");
});

test("hook load summary reports blocking repair guidance when the hook is broken", () => {
    const summary = summarizeOpenClawBrainHookLoad(buildInspection({
        loadability: "blocked",
        desynced: true,
        detail: "profile hook is present but OpenClaw will not load it",
    }), false);
    assert.equal(summary.guardSeverity, "blocking");
    assert.equal(summary.guardActionability, "repair_install");
    assert.match(summary.guardSummary, /will not load it/);
    assert.match(summary.guardAction, /openclawbrain install --openclaw-home/);
    assert.equal(summary.loadProof, "not_ready");
});

test("hook load summary treats activation-root-only reads as degraded scope advice", () => {
    const summary = summarizeOpenClawBrainHookLoad(buildInspection({
        scope: "activation_root_only",
        openclawHome: null,
        extensionDir: null,
        hookPath: null,
        runtimeGuardPath: null,
        manifestPath: null,
        packageJsonPath: null,
        manifestId: null,
        installId: null,
        packageName: null,
        installLayout: null,
        installState: "unverified",
        loadability: "unverified",
        pluginAllowlistState: "unverified",
        detail: "profile hook state is unknown from activation-root-only status",
    }), false);
    assert.equal(summary.guardSeverity, "degraded");
    assert.equal(summary.guardActionability, "pin_openclaw_home");
    assert.match(summary.guardSummary, /not self-proven/);
    assert.match(summary.guardAction, /--openclaw-home/);
});

test("hotfix boundary marks daemon and installed hook as separate surfaces even on matching versions", () => {
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
    assert.equal(boundary.daemonSource, "managed_service");
    assert.match(boundary.detail, /daemon background watch runs from/);
    assert.match(boundary.guidance, /Patch the daemon runtime path/);
    assert.match(boundary.guidance, /installed hook\/runtime-guard/);
});

test("hotfix boundary calls out version skew when daemon and installed hook diverge", () => {
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

test("hotfix boundary refuses to imply hook truth from activation-root-only status", () => {
    const boundary = describeOpenClawBrainHotfixBoundary({
        hookInspection: buildInspection({
            scope: "activation_root_only",
            openclawHome: null,
            extensionDir: null,
            hookPath: null,
            runtimeGuardPath: null,
            manifestPath: null,
            packageJsonPath: null,
            manifestId: null,
            installId: null,
            packageName: null,
            packageVersion: null,
            installLayout: null,
            installState: "unverified",
            loadability: "unverified",
            pluginAllowlistState: "unverified",
            detail: "profile hook state is unknown from activation-root-only status",
        }),
        daemonInspection: {
            configuredRuntimePath: "/tmp/openclawbrain/cli.js",
            configuredRuntimePackageName: "@openclawbrain/cli",
            configuredRuntimePackageVersion: "1.2.3",
            configuredRuntimePackageSpec: null,
        },
    });
    assert.equal(boundary.boundary, "hook_surface_unverified");
    assert.equal(boundary.skew, "unverified");
    assert.equal(boundary.convergeState, "unverified");
    assert.match(boundary.guidance, /Pin --openclaw-home/);
});

test("hotfix boundary treats a blocked installed hook as half-converged even when paths are visible", () => {
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
    assert.ok(boundary.convergeReasons.includes("installed_surface_not_loadable"));
});
