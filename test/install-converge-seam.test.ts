import { describe, expect, it } from "vitest";
import {
  planOpenClawBrainConvergePluginAction,
  classifyOpenClawBrainConvergeVerification,
  finalizeOpenClawBrainConvergeResult,
  // @ts-ignore — JS module, not typed
} from "../packages/cli/dist/src/install-converge.js";

const LEGACY_COMPAT_PACKAGE_NAME = "@jonathangu/openclawbrain";

describe("planOpenClawBrainConvergePluginAction — compat-package upgrade", () => {
  it("forces full install when the selected install is the retired compat package", () => {
    const plan = planOpenClawBrainConvergePluginAction({
      selectedInstall: {
        extensionDir: "/tmp/openclaw/extensions/openclawbrain",
        manifestId: "openclawbrain",
        installId: "openclawbrain",
        packageName: LEGACY_COMPAT_PACKAGE_NAME,
        installLayout: "generated_shadow_extension",
      },
    });

    expect(plan.action).toBe("install");
    expect(plan.packageSpec).toBe("@openclawbrain/openclaw");
    expect(plan.reason).toContain(LEGACY_COMPAT_PACKAGE_NAME);
    expect(plan.reason).toContain("retired compatibility package");
  });

  it("forces full install even if compat package claims native_package_plugin layout", () => {
    const plan = planOpenClawBrainConvergePluginAction({
      selectedInstall: {
        extensionDir: "/tmp/openclaw/extensions/openclawbrain",
        manifestId: "openclawbrain",
        installId: "openclawbrain",
        packageName: LEGACY_COMPAT_PACKAGE_NAME,
        installLayout: "native_package_plugin",
      },
    });

    // Even if layout says native, the compat package must trigger install not update
    expect(plan.action).toBe("install");
    expect(plan.reason).toContain(LEGACY_COMPAT_PACKAGE_NAME);
  });

  it("treats the canonical native package plugin as a no-op when it matches the daemon/runtime version", () => {
    const plan = planOpenClawBrainConvergePluginAction({
      selectedInstall: {
        extensionDir: "/tmp/openclaw/extensions/@openclawbrain/openclaw",
        manifestId: "openclawbrain",
        installId: "openclaw",
        packageName: "@openclawbrain/openclaw",
        packageVersion: "1.2.3",
        installLayout: "native_package_plugin",
      },
      daemonRuntimePackageVersion: "1.2.3",
    });

    expect(plan.action).toBe("noop");
  });

  it("refreshes the canonical native package plugin when the installed hook version lags the daemon/runtime version", () => {
    const plan = planOpenClawBrainConvergePluginAction({
      selectedInstall: {
        extensionDir: "/tmp/openclaw/extensions/@openclawbrain/openclaw",
        manifestId: "openclawbrain",
        installId: "openclaw",
        packageName: "@openclawbrain/openclaw",
        packageVersion: "1.2.2",
        installLayout: "native_package_plugin",
      },
      daemonRuntimePackageVersion: "1.2.3",
    });

    expect(plan.action).toBe("update");
    expect(plan.reason).toContain("1.2.2");
    expect(plan.reason).toContain("1.2.3");
  });
});

describe("classifyOpenClawBrainConvergeVerification — compat-package blocking", () => {
  it("blocks verification when the installed plugin is the retired compat package", () => {
    const verification = classifyOpenClawBrainConvergeVerification({
      installedPackageName: LEGACY_COMPAT_PACKAGE_NAME,
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
    });

    expect(verification.state).toBe("failed");
    expect(verification.blockingReasons.join("; ")).toContain(LEGACY_COMPAT_PACKAGE_NAME);
  });

  it("does not block when the installed plugin is the canonical package", () => {
    const verification = classifyOpenClawBrainConvergeVerification({
      installedPackageName: "@openclawbrain/openclaw",
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
    });

    expect(verification.state).toBe("healthy");
    expect(verification.blockingReasons).toHaveLength(0);
  });

  it("blocks verification when daemon and installed hook surfaces are half-converged", () => {
    const verification = classifyOpenClawBrainConvergeVerification({
      installedPackageName: "@openclawbrain/openclaw",
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
      selectedOpenClawHome: "/tmp/.openclaw-example",
      daemonPackage: "@openclawbrain/cli@1.2.2",
      hookPackage: "@openclawbrain/openclaw@1.2.3",
    });

    expect(verification.state).toBe("failed");
    expect(verification.blockingReasons.join("; ")).toContain("half-converged");
    expect(verification.blockingReasons.join("; ")).toContain("split_surfaces/split_path_version_skew");
  });
});

describe("classifyOpenClawBrainConvergeVerification — proof promotion", () => {
  it("promotes to healthy when runtime proof is green even if displayedStatus lags", () => {
    const verification = classifyOpenClawBrainConvergeVerification({
      installLayout: "native_package_plugin",
      installState: "installed",
      loadability: "loadable",
      displayedStatus: "warn",
      runtimeLoad: "proven",
      loadProof: "status_probe_ready",
      serveState: "serving_active_pack",
      routeFnAvailable: true,
      awaitingFirstExport: false,
      restartRequired: false,
      restartPerformed: false,
    });

    // With runtimeTruthAlreadyProven = true, the stale displayedStatus should
    // NOT produce a warning — proof surface should be promoted
    expect(verification.state).toBe("healthy");
    expect(verification.warnings).not.toContain(expect.stringContaining("status is warn"));
  });

  it("still warns about awaiting first export even when runtime proof is green", () => {
    const verification = classifyOpenClawBrainConvergeVerification({
      installLayout: "native_package_plugin",
      installState: "installed",
      loadability: "loadable",
      displayedStatus: "ok",
      runtimeLoad: "proven",
      loadProof: "status_probe_ready",
      serveState: "serving_active_pack",
      routeFnAvailable: true,
      awaitingFirstExport: true,
      restartRequired: false,
      restartPerformed: false,
    });

    expect(verification.state).toBe("warning");
    expect(verification.warnings.join("; ")).toContain("first export");
  });

  it("warns when surface convergence is still unverified even if runtime proof is green", () => {
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
      restartRequired: false,
      restartPerformed: false,
      surfaceConvergeState: "unverified",
    });

    expect(verification.state).toBe("warning");
    expect(verification.warnings.join("; ")).toContain("not fully proven");
  });

  it("produces degraded warnings without proof promotion when runtime is not proven", () => {
    const verification = classifyOpenClawBrainConvergeVerification({
      installLayout: "native_package_plugin",
      installState: "installed",
      loadability: "loadable",
      displayedStatus: "warn",
      runtimeLoad: "not_proven",
      loadProof: "unverified",
      serveState: "seed_state_authoritative",
      routeFnAvailable: false,
      awaitingFirstExport: true,
      restartRequired: false,
      restartPerformed: false,
    });

    expect(verification.state).toBe("warning");
    expect(verification.warnings.join("; ")).toContain("status is warn");
    expect(verification.warnings.join("; ")).toContain("runtime load is not_proven");
    expect(verification.warnings.join("; ")).toContain("route_fn availability");
  });

  it("finalizeOpenClawBrainConvergeResult returns converged when proof is green and healthy", () => {
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
      restartRequired: false,
      restartPerformed: false,
    });

    const result = finalizeOpenClawBrainConvergeResult({
      stepFailure: null,
      verification,
      warnings: [],
    });

    expect(result.verdict).toBe("converged");
  });

  it("blocks half-converged daemon-vs-installed-hook state loudly", () => {
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
      daemonPackage: "@openclawbrain/cli@0.4.30",
      hookPackage: "@openclawbrain/openclaw@0.4.28",
      selectedOpenClawHome: "/tmp/.openclaw-example",
      restartRequired: false,
      restartPerformed: false,
    });

    const result = finalizeOpenClawBrainConvergeResult({
      stepFailure: null,
      verification,
      warnings: [],
    });

    expect(verification.state).toBe("failed");
    expect(result.verdict).toBe("manual_action_required");
    expect(result.why).toContain("half-converged");
    expect(result.why).toContain("split_path_version_skew");
  });
});
