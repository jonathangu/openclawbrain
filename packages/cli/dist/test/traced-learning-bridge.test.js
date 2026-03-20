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
function createMissingBrainStoreEnv(t) {
    return {
        OPENCLAWBRAIN_ROOT: path.join(createTempRoot(t), "brain-root")
    };
}
function createBrainStore(t) {
    const root = createTempRoot(t);
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
    const surface = buildTracedLearningStatusSurface(activationRoot, {
        env: createMissingBrainStoreEnv(t)
    });
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

test("brain-store traced-learning bridge truthfully lifts supervision and update counters", (t) => {
    const { brainRoot, db, dbPath } = createBrainStore(t);
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
    }
});

test("status surface prefers canonical brain-store truth when the runtime bridge is missing", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db, dbPath } = createBrainStore(t);
    try {
        db.exec(`
          INSERT INTO brain_traces (id) VALUES ('bt_1'), ('bt_2'), ('bt_3');
          INSERT INTO brain_trace_supervision (id) VALUES ('ts_1');
        `);
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?), (?, ?)`).run(
            "last_pg_candidate_update_json",
            JSON.stringify({
                generatedAt: Date.parse("2026-03-20T04:23:00.000Z"),
                updateCount: 5,
                routeUpdateCount: 4,
                teacherLabelCount: 2
            }),
            "last_pg_candidate_pack_version",
            "11"
        );
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, dbPath);
        assert.equal(surface.present, true);
        assert.equal(surface.updatedAt, "2026-03-20T04:23:00.000Z");
        assert.equal(surface.routeTraceCount, 3);
        assert.equal(surface.supervisionCount, 1);
        assert.equal(surface.routerUpdateCount, 4);
        assert.equal(surface.teacherArtifactCount, 2);
        assert.equal(surface.materializedPackId, null);
        assert.equal(surface.promoted, false);
        assert.equal(surface.source?.command, "brain-store");
        assert.equal(surface.source?.bridge, "brain_store_state");
        assert.equal(surface.source?.runtimeMaterialized, undefined);
        assert.match(surface.detail, /source=brain-store/);
        assert.match(surface.detail, /bridge=brain_store_state/);
        assert.match(surface.detail, /runtime=missing/);
    }
    finally {
        db.close();
    }
});

test("status surface keeps runtime materialization metadata but canonical counts win on disagreement", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db, dbPath } = createBrainStore(t);
    try {
        db.exec(`
          INSERT INTO brain_traces (id) VALUES ('bt_1'), ('bt_2');
          INSERT INTO brain_trace_supervision (id) VALUES ('ts_1');
        `);
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?), (?, ?)`).run(
            "last_pg_candidate_update_json",
            JSON.stringify({
                generatedAt: Date.parse("2026-03-20T04:24:00.000Z"),
                updateCount: 4,
                routeUpdateCount: 3,
                teacherLabelCount: 1
            }),
            "last_pg_candidate_pack_version",
            "12"
        );
        const runtimePath = writeTracedLearningBridge(activationRoot, {
            updatedAt: "2026-03-20T04:30:00.000Z",
            routeTraceCount: 99,
            supervisionCount: 77,
            routerUpdateCount: 55,
            teacherArtifactCount: 44,
            pgVersionRequested: "v7",
            pgVersionUsed: "v7",
            decisionLogCount: 6,
            fallbackReason: "runtime-bridge-used",
            routerNoOpReason: "runtime-no-op",
            materializedPackId: "pack-999",
            promoted: true,
            baselinePersisted: true,
            source: {
                command: "learn",
                exportDigest: "digest-999"
            }
        });
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, dbPath);
        assert.equal(surface.present, true);
        assert.equal(surface.updatedAt, "2026-03-20T04:24:00.000Z");
        assert.equal(surface.routeTraceCount, 2);
        assert.equal(surface.supervisionCount, 1);
        assert.equal(surface.routerUpdateCount, 3);
        assert.equal(surface.teacherArtifactCount, 1);
        assert.equal(surface.pgVersionRequested, "v7");
        assert.equal(surface.pgVersionUsed, "v7");
        assert.equal(surface.decisionLogCount, 6);
        assert.equal(surface.materializedPackId, "pack-999");
        assert.equal(surface.promoted, true);
        assert.equal(surface.baselinePersisted, true);
        assert.equal(surface.source?.command, "brain-store");
        assert.equal(surface.source?.bridge, "brain_store_state");
        assert.equal(surface.source?.runtimeMaterialized?.path, runtimePath);
        assert.equal(surface.source?.runtimeMaterialized?.routeTraceCount, 99);
        assert.equal(surface.source?.runtimeMaterialized?.supervisionCount, 77);
        assert.equal(surface.source?.runtimeMaterialized?.routerUpdateCount, 55);
        assert.equal(surface.source?.runtimeMaterialized?.teacherArtifactCount, 44);
        assert.equal(surface.source?.runtimeMaterialized?.materializedPackId, "pack-999");
        assert.match(surface.detail, /runtime=present/);
        assert.match(surface.detail, /fallback=runtime-bridge-used/);
        assert.match(surface.detail, /noOp=runtime-no-op/);
    }
    finally {
        db.close();
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
    const surface = buildTracedLearningStatusSurface(activationRoot, {
        env: createMissingBrainStoreEnv(t)
    });
    assert.equal(surface.present, false);
    assert.equal(surface.routeTraceCount, 0);
    assert.equal(surface.supervisionCount, 0);
    assert.equal(surface.routerUpdateCount, 0);
    assert.equal(surface.detail, "bridge_unreadable");
    assert.match(surface.error ?? "", /Expected property name|JSON|Unexpected token/);
});
