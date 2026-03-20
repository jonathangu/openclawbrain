import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildTracedLearningStatusSurface, loadTracedLearningBridge, resolveTracedLearningBridgePath, writeTracedLearningBridge } from "../src/traced-learning-bridge.js";

function createTempActivationRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-traced-learning-bridge-"));
    const activationRoot = path.join(root, "activation-root");
    mkdirSync(activationRoot, { recursive: true });
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return activationRoot;
}

test("traced-learning bridge round-trips learn counters under activation-root/watch", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const bridgePath = writeTracedLearningBridge(activationRoot, {
        updatedAt: "2026-03-20T04:20:00.000Z",
        routeTraceCount: 12,
        supervisionCount: 5,
        routerUpdateCount: 3,
        teacherArtifactCount: 4,
        pgVersionRequested: "v2",
        pgVersionUsed: "v2",
        decisionLogCount: 12,
        fallbackReason: null,
        routerNoOpReason: null,
        materializedPackId: "pack-123",
        promoted: true,
        baselinePersisted: true,
        source: {
            command: "learn",
            exportDigest: "digest-123"
        }
    });
    assert.equal(bridgePath, resolveTracedLearningBridgePath(activationRoot));
    const rawBridge = JSON.parse(readFileSync(bridgePath, "utf8"));
    assert.equal(rawBridge.contract, "openclawbrain.traced-learning-bridge.v1");
    const loaded = loadTracedLearningBridge(activationRoot);
    assert.equal(loaded.error, null);
    assert.equal(loaded.bridge?.routeTraceCount, 12);
    assert.equal(loaded.bridge?.supervisionCount, 5);
    assert.equal(loaded.bridge?.routerUpdateCount, 3);
    assert.equal(loaded.bridge?.materializedPackId, "pack-123");
    const surface = buildTracedLearningStatusSurface(activationRoot);
    assert.equal(surface.present, true);
    assert.equal(surface.updatedAt, "2026-03-20T04:20:00.000Z");
    assert.equal(surface.routeTraceCount, 12);
    assert.equal(surface.supervisionCount, 5);
    assert.equal(surface.routerUpdateCount, 3);
    assert.equal(surface.teacherArtifactCount, 4);
    assert.equal(surface.pgVersionUsed, "v2");
    assert.equal(surface.materializedPackId, "pack-123");
    assert.match(surface.detail, /source=learn/);
    assert.match(surface.detail, /promoted=yes/);
});

test("traced-learning bridge fails open when the persisted file is malformed", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const bridgePath = resolveTracedLearningBridgePath(activationRoot);
    mkdirSync(path.dirname(bridgePath), { recursive: true });
    writeFileSync(bridgePath, "{ not-valid-json\n", "utf8");
    const loaded = loadTracedLearningBridge(activationRoot);
    assert.equal(loaded.bridge, null);
    assert.match(loaded.error ?? "", /Unexpected token|Expected property name|JSON/);
    const surface = buildTracedLearningStatusSurface(activationRoot);
    assert.equal(surface.present, false);
    assert.equal(surface.routeTraceCount, 0);
    assert.equal(surface.supervisionCount, 0);
    assert.equal(surface.routerUpdateCount, 0);
    assert.equal(surface.detail, "bridge_unreadable");
    assert.match(surface.error ?? "", /Unexpected token|Expected property name|JSON/);
});
