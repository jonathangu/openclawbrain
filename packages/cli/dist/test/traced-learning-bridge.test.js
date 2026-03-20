import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { buildTracedLearningStatusSurface, loadBrainStoreTracedLearningBridge, loadTracedLearningBridge, mergeTracedLearningBridgePayload, resolveTracedLearningBridgePath, writeTracedLearningBridge } from "../src/traced-learning-bridge.js";

function createTempRoot(t) {
    const root = mkdtempSync(path.join(os.tmpdir(), "openclawbrain-traced-learning-bridge-"));
    t.after(() => {
        rmSync(root, { recursive: true, force: true });
    });
    return root;
}
function createTempActivationRoot(t) {
    const activationRoot = path.join(createTempRoot(t), "activation-root");
    mkdirSync(activationRoot, { recursive: true });
    return activationRoot;
}
function createBrainStore(root) {
    const brainRoot = path.join(root, "brain-root");
    mkdirSync(brainRoot, { recursive: true });
    const dbPath = path.join(brainRoot, "state.db");
    const db = new DatabaseSync(dbPath);
    db.exec(`
      CREATE TABLE brain_traces (id TEXT PRIMARY KEY);
      CREATE TABLE brain_trace_supervision (id TEXT PRIMARY KEY);
      CREATE TABLE brain_training_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    `);
    return { brainRoot, db, dbPath };
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

test("brain-store traced-learning bridge truthfully lifts supervision and update counters", () => {
    const { brainRoot, db, dbPath } = createBrainStore(mkdtempSync(path.join(os.tmpdir(), "openclawbrain-traced-learning-store-")));
    try {
        db.exec(`
          INSERT INTO brain_traces (id) VALUES ('bt_1'), ('bt_2');
          INSERT INTO brain_trace_supervision (id) VALUES ('ts_1'), ('ts_2');
        `);
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?), (?, ?)`).run(
            "last_pg_candidate_update_json",
            JSON.stringify({
                generatedAt: Date.parse("2026-03-20T04:21:00.000Z"),
                updateCount: 4,
                routeUpdateCount: 3,
                teacherLabelCount: 1
            }),
            "last_pg_candidate_pack_version",
            "9"
        );
        const persisted = loadBrainStoreTracedLearningBridge({
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(persisted.path, dbPath);
        assert.equal(persisted.error, null);
        assert.equal(persisted.bridge?.routeTraceCount, 2);
        assert.equal(persisted.bridge?.supervisionCount, 2);
        assert.equal(persisted.bridge?.routerUpdateCount, 3);
        assert.equal(persisted.bridge?.teacherArtifactCount, 1);
        assert.equal(persisted.bridge?.updatedAt, "2026-03-20T04:21:00.000Z");
        assert.equal(persisted.bridge?.source?.command, "brain-store");
        assert.equal(persisted.bridge?.source?.bridge, "brain_store_state");
        assert.equal(persisted.bridge?.source?.candidatePackVersion, 9);
        const merged = mergeTracedLearningBridgePayload({
            updatedAt: "2026-03-20T04:22:00.000Z",
            routeTraceCount: 0,
            supervisionCount: 0,
            routerUpdateCount: 0,
            teacherArtifactCount: 0,
            pgVersionRequested: "v2",
            pgVersionUsed: "v2",
            decisionLogCount: 0,
            fallbackReason: null,
            routerNoOpReason: "no outcomes found for serve-time decisions",
            materializedPackId: null,
            promoted: false,
            baselinePersisted: false,
            source: {
                command: "learn"
            }
        }, persisted);
        assert.equal(merged.routeTraceCount, 2);
        assert.equal(merged.supervisionCount, 2);
        assert.equal(merged.routerUpdateCount, 3);
        assert.equal(merged.teacherArtifactCount, 1);
        assert.equal(merged.routerNoOpReason, null);
        assert.equal(merged.source?.bridge, "brain_store_state");
        assert.equal(merged.source?.bridgedRuntime?.path, dbPath);
        assert.equal(merged.source?.bridgedRuntime?.routerUpdateCount, 3);
    }
    finally {
        db.close();
        rmSync(path.dirname(brainRoot), { recursive: true, force: true });
    }
});

test("traced-learning bridge fails open when the persisted file is malformed", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const bridgePath = resolveTracedLearningBridgePath(activationRoot);
    mkdirSync(path.dirname(bridgePath), { recursive: true });
    writeFileSync(bridgePath, "{not-json", "utf8");
    const loaded = loadTracedLearningBridge(activationRoot);
    assert.equal(loaded.bridge, null);
    assert.match(loaded.error ?? "", /Expected property name|JSON|Unexpected token/);
    const surface = buildTracedLearningStatusSurface(activationRoot);
    assert.equal(surface.present, false);
    assert.equal(surface.routeTraceCount, 0);
    assert.equal(surface.supervisionCount, 0);
    assert.equal(surface.routerUpdateCount, 0);
    assert.equal(surface.detail, "bridge_unreadable");
    assert.match(surface.error ?? "", /Expected property name|JSON|Unexpected token/);
});
