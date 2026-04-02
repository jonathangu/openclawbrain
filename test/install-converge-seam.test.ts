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

  it("plans update for the canonical native package plugin", () => {
    const plan = planOpenClawBrainConvergePluginAction({
      selectedInstall: {
        extensionDir: "/tmp/openclaw/extensions/@openclawbrain/openclaw",
        manifestId: "openclawbrain",
        installId: "openclaw",
        packageName: "@openclawbrain/openclaw",
        installLayout: "native_package_plugin",
      },
    });

    expect(plan.action).toBe("update");
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
});
