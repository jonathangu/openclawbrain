import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function loadFunction({ file, startMarker, endMarker, prelude = "" }) {
    const source = readFileSync(path.join(__dirname, "..", "src", file), "utf8");
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start);
    if (start === -1 || end === -1) {
        throw new Error(`failed to locate ${startMarker} in ${file}`);
    }
    const block = source.slice(start, end).replace(/^export\s+/gmu, "");
    const match = /function\s+([A-Za-z0-9_]+)/u.exec(startMarker);
    if (match === null) {
        throw new Error(`failed to extract function name from ${startMarker}`);
    }
    return new Function(`${prelude}\n${block}\nreturn ${match[1]};`)();
}

test("current profile status with report reuses one live operator snapshot", () => {
    const describeCurrentProfileBrainStatusWithReport = loadFunction({
        file: "index.js",
        startMarker: "export function describeCurrentProfileBrainStatusWithReport",
        endMarker: "export function describeCurrentProfileBrainStatus(input)",
        prelude: `
            let buildCalls = 0;
            function buildOperatorSurfaceReport() {
                buildCalls += 1;
                return {
                    marker: buildCalls,
                    manyProfile: {
                        declaredAttachmentPolicy: "dedicated"
                    }
                };
            }
            function buildCurrentProfileBrainStatusFromReport(report, policyMode, profileId) {
                return {
                    marker: report.marker,
                    policyMode,
                    profileId
                };
            }
            function normalizeOptionalString(value) {
                return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
            }
            globalThis.__ocbBuildCalls = () => buildCalls;
        `
    });

    const result = describeCurrentProfileBrainStatusWithReport({
        profileId: " current_profile "
    });

    assert.equal(globalThis.__ocbBuildCalls(), 1);
    assert.equal(result.report.marker, 1);
    assert.deepEqual(result.status, {
        marker: 1,
        policyMode: "dedicated",
        profileId: "current_profile"
    });

    delete globalThis.__ocbBuildCalls;
});

test("current profile status preserves hook packageVersion for hotfix-boundary reporting", () => {
    const buildCurrentProfileBrainStatusFromReport = loadFunction({
        file: "index.js",
        startMarker: "function buildCurrentProfileBrainStatusFromReport",
        endMarker: "export function buildOperatorSurfaceReport",
        prelude: `
            const CURRENT_PROFILE_BRAIN_STATUS_CONTRACT = "current_profile_brain_status.v1";
            function isAwaitingFirstExportSlot() { return false; }
            function summarizeCurrentProfilePassiveLearning() {
                return { learnerRunning: true, exportState: "latest_export_visible", backlogState: "caught_up", pendingLive: 0, pendingBackfill: 0, firstExportOccurred: true, currentServingPackId: "pack-123", lastObservedDelta: { explanation: "none" } };
            }
            function summarizeCurrentProfileBrainStatusLevel() { return "ok"; }
            function summarizeCurrentProfileLogRoot(activationRoot) { return activationRoot + "/logs"; }
            function summarizeCurrentProfileLastLearningUpdateAt() { return "2026-04-04T14:00:00.000Z"; }
            function summarizeCurrentProfileBrainSummary() { return "summary"; }
            function buildCurrentProfileAttachmentPolicy(policyMode) { return { mode: policyMode }; }
            function buildCurrentProfileTurnAttributionFromReport() { return { source: "stub" }; }
        `
    });

    const result = buildCurrentProfileBrainStatusFromReport({
        generatedAt: "2026-04-04T14:00:00.000Z",
        activationRoot: "/tmp/activation",
        activation: { state: "active", detail: "ok" },
        brain: { activePackId: "pack-123", state: "pg_promoted_pack_authoritative", detail: "ok" },
        servePath: {
            routerIdentity: "pack-123:route_fn",
            refreshStatus: "updated",
            activePackId: "pack-123",
            state: "serving_active_pack",
            usedLearnedRouteFn: true,
            fallbackToStaticContext: false,
            structuralDecision: null,
            timing: null
        },
        learnedRouting: {
            routerIdentity: "pack-123:route_fn",
            initMode: "fast_boot_defaults",
            routerChecksum: "sha256-test"
        },
        supervision: { exportedAt: "2026-04-04T14:00:00.000Z" },
        promotion: { lastPromotion: { at: "2026-04-04T14:00:00.000Z" } },
        learning: {},
        teacherLoop: { observationBinding: {} },
        hook: {
            scope: "exact_openclaw_home",
            openclawHome: "/tmp/.openclaw",
            extensionDir: "/tmp/.openclaw/extensions/openclawbrain",
            hookPath: "/tmp/.openclaw/extensions/openclawbrain/dist/extension/index.js",
            runtimeGuardPath: "/tmp/.openclaw/extensions/openclawbrain/dist/extension/runtime-guard.js",
            manifestPath: "/tmp/.openclaw/extensions/openclawbrain/openclaw.plugin.json",
            packageJsonPath: "/tmp/.openclaw/extensions/openclawbrain/package.json",
            manifestId: "openclawbrain",
            installId: "openclaw",
            packageName: "@openclawbrain/openclaw",
            packageVersion: "0.4.30",
            installLayout: "native_package_plugin",
            additionalInstallCount: 0,
            installState: "installed",
            loadability: "loadable",
            loadProof: "status_probe_ready",
            guardSeverity: "none",
            guardActionability: "none",
            guardSummary: "profile hook is installed and loadable",
            guardAction: "none",
            desynced: false,
            detail: "ok"
        },
        attachmentTruth: {
            state: "attached",
            activationRoot: "/tmp/activation",
            servingSlot: "active",
            proofState: "proven",
            watchOnly: false,
            detail: "attached"
        },
        active: { packId: "pack-123", routerIdentity: "pack-123:route_fn" }
    }, "dedicated", "current_profile");

    assert.equal(result.hook.packageName, "@openclawbrain/openclaw");
    assert.equal(result.hook.packageVersion, "0.4.30");
});
