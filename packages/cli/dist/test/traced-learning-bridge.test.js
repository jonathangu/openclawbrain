import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { buildTracedLearningBridgePayloadFromRuntime, buildTracedLearningStatusSurface, loadBrainStoreTracedLearningBridge, loadTracedLearningBridge, mergeTracedLearningBridgePayload, persistBrainStoreTracedLearningBridge, persistTracedLearningBridgeState, resolveTracedLearningBridgePath, writeTracedLearningBridge } from "../src/traced-learning-bridge.js";

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
      CREATE TABLE brain_traces (
        id TEXT PRIMARY KEY,
        created_at INTEGER,
        route_trace_json TEXT
      );
      CREATE TABLE brain_trace_supervision (
        id TEXT PRIMARY KEY,
        trace_id TEXT,
        episode_id TEXT,
        conversation_id INTEGER,
        source TEXT,
        kind TEXT,
        value REAL,
        confidence REAL,
        reason TEXT,
        content_snippet TEXT,
        resolution TEXT,
        label_id TEXT,
        evidence_id TEXT,
        metadata TEXT,
        created_at INTEGER
      );
      CREATE TABLE brain_observations (
        id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL UNIQUE,
        conversation_id INTEGER,
        trace_id TEXT,
        query_text TEXT NOT NULL DEFAULT '',
        retrieved_context_json TEXT NOT NULL DEFAULT '[]',
        route_metadata_json TEXT NOT NULL DEFAULT '{}',
        assistant_response TEXT NOT NULL DEFAULT '',
        tool_results_json TEXT NOT NULL DEFAULT '[]',
        follow_up_text TEXT,
        status TEXT NOT NULL DEFAULT 'pending_followup',
        teacher_evaluation_json TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        evaluated_at INTEGER
      );
      CREATE TABLE brain_training_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    `);
    return { brainRoot, db, dbPath };
}
function readPersistedStatusSurface(db) {
    const row = db.prepare(`SELECT value FROM brain_training_state WHERE key = ?`).get("traced_learning_status_surface_json");
    if (row === undefined || typeof row.value !== "string") {
        return null;
    }
    return JSON.parse(row.value);
}
function makeLastAssemblyDecision(overrides = {}) {
    return {
        mode: "partial_deadline_after_query",
        brainDropReason: "deadline_after_query",
        interruptionStage: "query",
        interruptionReason: "soft_compile_deadline",
        servedPartial: true,
        interruptionAccounting: {
            droppedFrontierNodeIds: ["node-a", "node-b"],
            droppedProposalCount: 3,
            budgetUtilization: 0.625
        },
        ...overrides
    };
}
function writeActivePackFixture(activationRoot, options = {}) {
    const packId = options.packId ?? "pack-active";
    const packRoot = path.join(activationRoot, "packs", packId);
    const manifestPath = path.join(packRoot, "manifest.json");
    const routerPath = path.join(packRoot, "router", "model.json");
    mkdirSync(path.dirname(routerPath), { recursive: true });
    writeFileSync(path.join(activationRoot, "activation-pointers.json"), `${JSON.stringify({
        contract: "activation_pointers.v1",
        active: {
            slot: "active",
            packId,
            packRootDir: packRoot,
            manifestPath
        },
        candidate: null,
        previous: null
    }, null, 2)}\n`);
    writeFileSync(manifestPath, `${JSON.stringify({
        contract: "artifact_manifest.v1",
        packId,
        runtimeAssets: {
            router: {
                kind: "artifact",
                identity: `${packId}:route_fn`,
                artifactPath: "router/model.json"
            }
        }
    }, null, 2)}\n`);
    writeFileSync(routerPath, `${JSON.stringify({
        routerIdentity: `${packId}:route_fn`,
        strategy: "learned_route_fn_v1",
        trainedAt: "2026-04-02T02:50:43.775Z",
        requiresLearnedRouting: true,
        training: {
            method: "policy_gradient_v1",
            status: "updated",
            routeTraceCount: options.routeTraceCount ?? 5,
            supervisionCount: options.supervisionCount ?? 2,
            updateCount: 3
        },
        traces: options.traces ?? [
            {
                traceId: "trace-1",
                sourceEventId: "evt-1",
                supervisionKind: "route_trace",
                reward: 0
            },
            {
                traceId: "trace-2",
                sourceEventId: "evt-2",
                supervisionKind: "route_trace",
                reward: 0
            },
            {
                traceId: "trace-3",
                sourceEventId: "evt-3",
                supervisionKind: "human_feedback",
                reward: 1
            },
            {
                traceId: "trace-4",
                sourceEventId: "evt-4",
                supervisionKind: "human_feedback",
                reward: 0
            },
            {
                traceId: "trace-5",
                sourceEventId: "evt-5",
                supervisionKind: "human_feedback",
                reward: -1
            }
        ]
    }, null, 2)}\n`);
}
function writeWatchTeacherSnapshotFixture(activationRoot, notes) {
    const snapshotPath = path.join(activationRoot, "watch", "teacher-snapshot.json");
    mkdirSync(path.dirname(snapshotPath), { recursive: true });
    writeFileSync(snapshotPath, `${JSON.stringify({ notes }, null, 2)}\n`);
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

test("brain-store traced-learning surface persists surfaced learn truth", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db, dbPath } = createBrainStore(t);
    try {
        const persisted = persistBrainStoreTracedLearningBridge({
            updatedAt: "2026-03-20T04:26:00.000Z",
            routeTraceCount: 264,
            supervisionCount: 18,
            routerUpdateCount: 7,
            teacherArtifactCount: 32,
            pgVersionRequested: "v2",
            pgVersionUsed: "v2",
            decisionLogCount: 264,
            fallbackReason: null,
            routerNoOpReason: null,
            materializedPackId: "pack-ed1142ba",
            promoted: true,
            baselinePersisted: true,
            source: {
                command: "learn",
                exportDigest: "digest-264"
            }
        }, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(persisted.path, dbPath);
        assert.equal(persisted.persisted, true);
        assert.equal(persisted.error, null);
        const raw = readPersistedStatusSurface(db);
        assert.equal(raw?.contract, "openclawbrain.traced-learning-status-surface.v1");
        assert.equal(raw?.routeTraceCount, 264);
        assert.equal(raw?.teacherArtifactCount, 32);
        const loaded = loadBrainStoreTracedLearningBridge({
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(loaded.path, dbPath);
        assert.equal(loaded.bridge?.routeTraceCount, 264);
        assert.equal(loaded.bridge?.supervisionCount, 18);
        assert.equal(loaded.bridge?.routerUpdateCount, 7);
        assert.equal(loaded.bridge?.teacherArtifactCount, 32);
        assert.equal(loaded.bridge?.materializedPackId, "pack-ed1142ba");
        assert.equal(loaded.bridge?.source?.command, "brain-store");
        assert.equal(loaded.bridge?.source?.bridge, "brain_store_traced_learning_status_surface");
        assert.equal(loaded.bridge?.source?.surfacedFrom?.command, "learn");
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, dbPath);
        assert.equal(surface.routeTraceCount, 264);
        assert.equal(surface.supervisionCount, 18);
        assert.equal(surface.routerUpdateCount, 7);
        assert.equal(surface.teacherArtifactCount, 32);
        assert.equal(surface.materializedPackId, "pack-ed1142ba");
        assert.equal(surface.promoted, true);
        assert.match(surface.detail, /source=brain-store/);
        assert.match(surface.detail, /bridge=brain_store_traced_learning_status_surface/);
        assert.match(surface.detail, /runtime=missing/);
    }
    finally {
        db.close();
    }
});

test("brain-store traced-learning surface derives live feedback and attribution coverage truth from state.db", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db, dbPath } = createBrainStore(t);
    const now = Date.now();
    try {
        db.prepare(`INSERT INTO brain_traces (id, created_at, route_trace_json) VALUES (?, ?, ?)`).run(
            "trace-1",
            now - 5_000,
            JSON.stringify({ agentIdentity: { agentId: "main", lane: "subagent" } })
        );
        db.prepare(`INSERT INTO brain_traces (id, created_at, route_trace_json) VALUES (?, ?, ?)`).run(
            "trace-2",
            now - 4_000,
            JSON.stringify({ agentIdentity: { agentId: "main", lane: "main" } })
        );
        db.prepare(`
          INSERT INTO brain_trace_supervision (
            id, trace_id, episode_id, conversation_id, source, kind, value, confidence, reason, content_snippet, resolution, label_id, evidence_id, metadata, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
            "sup-1",
            "trace-1",
            "ep-1",
            1,
            "teacher_review",
            "teacher_review",
            0.9,
            1,
            "helpful",
            "",
            "promoted_to_label",
            null,
            null,
            JSON.stringify({ agentIdentity: { agentId: "main", lane: "subagent" } }),
            now - 2_000
        );
        db.prepare(`
          INSERT INTO brain_trace_supervision (
            id, trace_id, episode_id, conversation_id, source, kind, value, confidence, reason, content_snippet, resolution, label_id, evidence_id, metadata, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
            "sup-2",
            "trace-2",
            "ep-2",
            1,
            "teacher_review",
            "teacher_review",
            -0.5,
            1,
            "harmful",
            "",
            "promoted_to_label",
            null,
            null,
            "{}",
            now - 1_000
        );
        db.prepare(`
          INSERT INTO brain_observations (
            id, episode_id, conversation_id, trace_id, tool_results_json, follow_up_text, status, teacher_evaluation_json, created_at, updated_at, evaluated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
            "obs-completed",
            "ep-completed",
            1,
            "trace-1",
            "[]",
            null,
            "completed",
            null,
            now - 6_000,
            now - 6_000,
            now - 5_500
        );
        db.prepare(`
          INSERT INTO brain_observations (
            id, episode_id, conversation_id, trace_id, tool_results_json, follow_up_text, status, teacher_evaluation_json, created_at, updated_at, evaluated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
            "obs-ready-old",
            "ep-ready-old",
            1,
            "trace-1",
            "[]",
            null,
            "pending_followup",
            null,
            now - 5_000,
            now - 5_000,
            null
        );
        db.prepare(`
          INSERT INTO brain_observations (
            id, episode_id, conversation_id, trace_id, tool_results_json, follow_up_text, status, teacher_evaluation_json, created_at, updated_at, evaluated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
            "obs-ready-followup",
            "ep-ready-followup",
            1,
            "trace-2",
            "[]",
            "operator follow-up",
            "pending_teacher",
            null,
            now - 800,
            now - 800,
            null
        );
        db.prepare(`
          INSERT INTO brain_observations (
            id, episode_id, conversation_id, trace_id, tool_results_json, follow_up_text, status, teacher_evaluation_json, created_at, updated_at, evaluated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).run(
            "obs-delayed",
            "ep-delayed",
            1,
            "trace-2",
            "[]",
            null,
            "pending_followup",
            null,
            now - 200,
            now - 200,
            null
        );
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?)`).run(
            "last_teacher_evaluation_cycle_json",
            JSON.stringify({
                generatedAt: now - 100,
                budgetPerTick: 1,
                delayMs: 1_000
            })
        );
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, dbPath);
        assert.equal(surface.present, true);
        assert.deepEqual(surface.feedbackSummary, {
            visible: true,
            helpfulCount: 1,
            irrelevantCount: 0,
            harmfulCount: 1,
            supervisedTraceCount: 2,
            routeTraceCount: 2,
            latestAgentIdentity: { agentId: "main", lane: "main" },
            latestLabel: "main",
            detail: "1 helpful, 0 irrelevant, 1 harmful; 2/2 traced routes are supervised"
        });
        assert.deepEqual(surface.attributionCoverage, {
            visible: true,
            gatingVisible: true,
            completedWithoutEvaluationCount: 1,
            readyCount: 2,
            delayedCount: 1,
            budgetDeferredCount: 1,
            detail: "completed_without_evaluation=1; ready=2, delayed=1, budget_deferred=1"
        });
    }
    finally {
        db.close();
    }
});

test("status surface falls back to active-pack supervision and watch teacher queue when brain tables are empty", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db } = createBrainStore(t);
    try {
        persistBrainStoreTracedLearningBridge({
            updatedAt: "2026-04-03T01:23:34.349Z",
            routeTraceCount: 3111,
            supervisionCount: 58,
            routerUpdateCount: 65,
            teacherArtifactCount: 748,
            pgVersionRequested: "v1",
            pgVersionUsed: "v1",
            decisionLogCount: 0,
            fallbackReason: "serve_time_decision_log_tail_truncated+oversized_lines_skipped",
            routerNoOpReason: null,
            materializedPackId: null,
            promoted: false,
            baselinePersisted: false,
            source: {
                command: "watch"
            }
        }, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        writeActivePackFixture(activationRoot);
        writeWatchTeacherSnapshotFixture(activationRoot, [
            "teacher_queue_depth=0",
            "teacher_freshness=fresh",
            "teacher_budget=32",
            "teacher_delay_ms=0",
            "teacher_feedback_eligible=83",
            "teacher_feedback_delayed=2",
            "teacher_feedback_budgeted_out=51"
        ]);
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.deepEqual(surface.feedbackSummary, {
            visible: true,
            helpfulCount: 1,
            irrelevantCount: 1,
            harmfulCount: 1,
            supervisedTraceCount: 3,
            routeTraceCount: 5,
            latestAgentIdentity: null,
            latestLabel: null,
            detail: "1 helpful, 1 irrelevant, 1 harmful; 3/5 active-pack traced routes are supervised"
        });
        assert.deepEqual(surface.attributionCoverage, {
            visible: true,
            gatingVisible: true,
            completedWithoutEvaluationCount: 0,
            readyCount: 83,
            delayedCount: 2,
            budgetDeferredCount: 51,
            detail: "watch sparse-feedback queue: completed_without_evaluation=0, ready=83, delayed=2, budget_deferred=51"
        });
    }
    finally {
        db.close();
    }
});

test("brain-store traced-learning bridge derives last assembly interruption summary from state.db", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db, dbPath } = createBrainStore(t);
    try {
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?)`).run(
            "last_assembly_decision_json",
            JSON.stringify(makeLastAssemblyDecision())
        );
        const loaded = loadBrainStoreTracedLearningBridge({
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(loaded.path, dbPath);
        assert.deepEqual(loaded.bridge?.lastInterruptionSummary, {
            reason: "deadline_after_query",
            stage: "query",
            servedPartial: true,
            droppedFrontierCount: 2,
            droppedProposalCount: 3,
            budgetUtilization: 0.625
        });
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, dbPath);
        assert.equal(surface.present, true);
        assert.deepEqual(surface.lastInterruptionSummary, {
            reason: "deadline_after_query",
            stage: "query",
            servedPartial: true,
            droppedFrontierCount: 2,
            droppedProposalCount: 3,
            budgetUtilization: 0.625
        });
        assert.match(surface.detail, /interrupt=deadline_after_query/);
        assert.match(surface.detail, /partial=yes/);
        assert.match(surface.detail, /frontier=2/);
        assert.match(surface.detail, /proposals=3/);
        assert.match(surface.detail, /budget=63%/);
    }
    finally {
        db.close();
    }
});

test("persisted traced-learning bridge keeps the last assembly interruption summary", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db } = createBrainStore(t);
    try {
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?)`).run(
            "last_assembly_decision_json",
            JSON.stringify(makeLastAssemblyDecision({
                brainDropReason: "deadline_before_injection",
                interruptionStage: "injection",
                interruptionAccounting: {
                    droppedFrontierNodeIds: ["node-z"],
                    droppedProposalCount: 1,
                    budgetUtilization: 0.4
                }
            }))
        );
        const bridge = buildTracedLearningBridgePayloadFromRuntime({
            updatedAt: "2026-03-22T09:14:00.000Z",
            teacherArtifactCount: 4,
            serveTimeLearning: {
                pgVersion: "v2",
                decisionLogCount: 12,
                fallbackReason: null
            },
            materializedPackId: "pack-watch-123",
            promoted: false,
            baselinePersisted: false,
            source: {
                command: "watch",
                scanRoot: path.join(activationRoot, "event-exports")
            }
        });
        const persisted = persistTracedLearningBridgeState(activationRoot, bridge, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.deepEqual(persisted.lastInterruptionSummary, {
            reason: "deadline_before_injection",
            stage: "injection",
            servedPartial: true,
            droppedFrontierCount: 1,
            droppedProposalCount: 1,
            budgetUtilization: 0.4
        });
        assert.deepEqual(readPersistedStatusSurface(db)?.lastInterruptionSummary, {
            reason: "deadline_before_injection",
            stage: "injection",
            servedPartial: true,
            droppedFrontierCount: 1,
            droppedProposalCount: 1,
            budgetUtilization: 0.4
        });
        assert.deepEqual(loadTracedLearningBridge(activationRoot).bridge?.lastInterruptionSummary, {
            reason: "deadline_before_injection",
            stage: "injection",
            servedPartial: true,
            droppedFrontierCount: 1,
            droppedProposalCount: 1,
            budgetUtilization: 0.4
        });
    }
    finally {
        db.close();
    }
});

test("watch bridge helper persists traced-learning status surface for daemon readers", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db, dbPath } = createBrainStore(t);
    try {
        const watchBridge = buildTracedLearningBridgePayloadFromRuntime({
            updatedAt: "2026-03-22T09:14:00.000Z",
            lastMaterialization: {
                candidate: {
                    summary: {
                        packId: "pack-watch-123",
                        learnedRouter: {
                            routeTraceCount: 12,
                            supervisionCount: 5,
                            updateCount: 3,
                            noOpReason: null
                        }
                    },
                    routingBuild: {
                        learnedRoutingPath: "policy_gradient_v2",
                        pgVersionRequested: "v2",
                        pgVersionUsed: "v2",
                        decisionLogCount: 12,
                        fallbackReason: null,
                        updatedBaseline: null
                    }
                }
            },
            teacherArtifactCount: 4,
            serveTimeLearning: {
                pgVersion: "v2",
                decisionLogCount: 12,
                fallbackReason: null
            },
            materializedPackId: "pack-watch-123",
            promoted: false,
            baselinePersisted: false,
            source: {
                command: "watch",
                scanRoot: path.join(activationRoot, "event-exports"),
                teacherSnapshotPath: path.join(activationRoot, "watch", "teacher-snapshot.json")
            }
        });
        const surfaced = persistTracedLearningBridgeState(activationRoot, watchBridge, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surfaced.routeTraceCount, 12);
        assert.equal(surfaced.supervisionCount, 5);
        assert.equal(surfaced.routerUpdateCount, 3);
        assert.equal(surfaced.materializedPackId, "pack-watch-123");
        assert.equal(surfaced.source?.command, "watch");
        const runtime = loadTracedLearningBridge(activationRoot);
        assert.equal(runtime.bridge?.source?.command, "watch");
        assert.equal(runtime.bridge?.routeTraceCount, 12);
        const canonical = loadBrainStoreTracedLearningBridge({
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(canonical.path, dbPath);
        assert.equal(canonical.bridge?.routeTraceCount, 12);
        assert.equal(canonical.bridge?.supervisionCount, 5);
        assert.equal(canonical.bridge?.routerUpdateCount, 3);
        assert.equal(canonical.bridge?.materializedPackId, "pack-watch-123");
        assert.equal(canonical.bridge?.source?.bridge, "brain_store_traced_learning_status_surface");
        assert.equal(canonical.bridge?.source?.surfacedFrom?.command, "watch");
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, dbPath);
        assert.equal(surface.present, true);
        assert.equal(surface.routeTraceCount, 12);
        assert.equal(surface.supervisionCount, 5);
        assert.equal(surface.routerUpdateCount, 3);
        assert.equal(surface.teacherArtifactCount, 4);
        assert.equal(surface.pgVersionUsed, "v2");
        assert.equal(surface.materializedPackId, "pack-watch-123");
        assert.equal(surface.promoted, false);
        assert.equal(surface.source?.command, "brain-store");
        assert.equal(surface.source?.surfacedFrom?.command, "watch");
        assert.match(surface.detail, /bridge=brain_store_traced_learning_status_surface/);
        assert.match(surface.detail, /runtime=present/);
    }
    finally {
        db.close();
    }
});

test("repeated persisted status surfaces stay JSON-serializable instead of recursively inflating provenance", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db } = createBrainStore(t);
    try {
        const watchBridge = buildTracedLearningBridgePayloadFromRuntime({
            updatedAt: "2026-03-20T04:24:00.000Z",
            teacherArtifactCount: 4,
            materializedPackId: "pack-watch-123",
            lastMaterialization: {
                candidate: {
                    summary: {
                        packId: "pack-watch-123",
                        learnedRouter: {
                            routeTraceCount: 12,
                            supervisionCount: 5,
                            updateCount: 3,
                            noOpReason: null
                        }
                    },
                    routingBuild: {
                        learnedRoutingPath: "policy_gradient_v2",
                        pgVersionRequested: "v2",
                        pgVersionUsed: "v2",
                        decisionLogCount: 12,
                        fallbackReason: null
                    }
                }
            },
            source: {
                command: "watch",
                teacherSnapshotPath: path.join(activationRoot, "watch", "teacher-snapshot.json")
            }
        });
        persistTracedLearningBridgeState(activationRoot, watchBridge, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        persistTracedLearningBridgeState(activationRoot, watchBridge, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.doesNotThrow(() => JSON.stringify(surface));
        assert.equal(surface.source?.command, "brain-store");
        assert.equal(surface.source?.surfacedFrom?.command, "watch");
        assert.equal(surface.source?.surfacedFrom?.bridgedRuntime, undefined);
        assert.equal(surface.source?.runtimeMaterialized?.source?.command, "watch");
        assert.equal(surface.source?.runtimeMaterialized?.source?.bridgedRuntime, undefined);
    }
    finally {
        db.close();
    }
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
        assert.equal(merged.source?.bridgedRuntime?.source?.bridge, "brain_store_state");
        assert.equal(merged.source?.bridgedRuntime?.source?.command, "brain-store");
        assert.equal(merged.source?.bridgedRuntime?.source?.candidatePackVersion, 9);
        assert.equal(JSON.parse(JSON.stringify(merged)).source?.bridgedRuntime?.source?.bridge, "brain_store_state");
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
        persistBrainStoreTracedLearningBridge({
            updatedAt: "2026-03-20T04:24:00.000Z",
            routeTraceCount: 2,
            supervisionCount: 1,
            routerUpdateCount: 3,
            teacherArtifactCount: 1,
            pgVersionRequested: "v2",
            pgVersionUsed: "v2",
            decisionLogCount: 4,
            fallbackReason: null,
            routerNoOpReason: null,
            materializedPackId: "pack-123",
            promoted: true,
            baselinePersisted: true,
            source: {
                command: "learn",
                exportDigest: "digest-123"
            }
        }, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
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
        assert.equal(surface.pgVersionRequested, "v2");
        assert.equal(surface.pgVersionUsed, "v2");
        assert.equal(surface.decisionLogCount, 4);
        assert.equal(surface.materializedPackId, "pack-123");
        assert.equal(surface.promoted, true);
        assert.equal(surface.baselinePersisted, true);
        assert.equal(surface.source?.command, "brain-store");
        assert.equal(surface.source?.bridge, "brain_store_traced_learning_status_surface");
        assert.equal(surface.source?.runtimeMaterialized?.path, runtimePath);
        assert.equal(surface.source?.runtimeMaterialized?.routeTraceCount, 99);
        assert.equal(surface.source?.runtimeMaterialized?.supervisionCount, 77);
        assert.equal(surface.source?.runtimeMaterialized?.routerUpdateCount, 55);
        assert.equal(surface.source?.runtimeMaterialized?.teacherArtifactCount, 44);
        assert.equal(surface.source?.runtimeMaterialized?.materializedPackId, "pack-999");
        assert.equal(surface.source?.runtimeMaterialized?.fallbackReason, "runtime-bridge-used");
        assert.equal(surface.source?.runtimeMaterialized?.routerNoOpReason, "runtime-no-op");
        assert.match(surface.detail, /runtime=present/);
    }
    finally {
        db.close();
    }
});

test("status surface falls back to runtime materialization when canonical brain-store has no traced-learning signal", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db } = createBrainStore(t);
    try {
        const runtimePath = writeTracedLearningBridge(activationRoot, {
            updatedAt: "2026-03-20T04:31:00.000Z",
            routeTraceCount: 264,
            supervisionCount: 12,
            routerUpdateCount: 6,
            teacherArtifactCount: 32,
            pgVersionRequested: "v2",
            pgVersionUsed: "v2",
            decisionLogCount: 264,
            fallbackReason: null,
            routerNoOpReason: null,
            materializedPackId: "pack-ed1142ba",
            promoted: true,
            baselinePersisted: true,
            source: {
                command: "learn",
                exportDigest: "digest-runtime"
            }
        });
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, runtimePath);
        assert.equal(surface.present, true);
        assert.equal(surface.routeTraceCount, 264);
        assert.equal(surface.supervisionCount, 12);
        assert.equal(surface.routerUpdateCount, 6);
        assert.equal(surface.teacherArtifactCount, 32);
        assert.equal(surface.source?.command, "learn");
        assert.match(surface.detail, /source=learn/);
    }
    finally {
        db.close();
    }
});

test("status surface fails open to runtime materialization when the canonical summary is malformed", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db } = createBrainStore(t);
    try {
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?)`).run(
            "traced_learning_status_surface_json",
            "{not-json"
        );
        const runtimePath = writeTracedLearningBridge(activationRoot, {
            updatedAt: "2026-03-20T04:32:00.000Z",
            routeTraceCount: 264,
            supervisionCount: 12,
            routerUpdateCount: 6,
            teacherArtifactCount: 32,
            pgVersionRequested: "v2",
            pgVersionUsed: "v2",
            decisionLogCount: 264,
            fallbackReason: null,
            routerNoOpReason: null,
            materializedPackId: "pack-ed1142ba",
            promoted: true,
            baselinePersisted: true,
            source: {
                command: "learn",
                exportDigest: "digest-runtime"
            }
        });
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, runtimePath);
        assert.equal(surface.present, true);
        assert.equal(surface.routeTraceCount, 264);
        assert.equal(surface.source?.command, "learn");
    }
    finally {
        db.close();
    }
});

test("status surface fails open to runtime materialization when the canonical summary is structurally malformed", (t) => {
    const activationRoot = createTempActivationRoot(t);
    const { brainRoot, db } = createBrainStore(t);
    try {
        db.prepare(`INSERT INTO brain_training_state (key, value) VALUES (?, ?)`).run(
            "traced_learning_status_surface_json",
            JSON.stringify({})
        );
        const runtimePath = writeTracedLearningBridge(activationRoot, {
            updatedAt: "2026-03-20T04:33:00.000Z",
            routeTraceCount: 264,
            supervisionCount: 12,
            routerUpdateCount: 6,
            teacherArtifactCount: 32,
            pgVersionRequested: "v2",
            pgVersionUsed: "v2",
            decisionLogCount: 264,
            fallbackReason: null,
            routerNoOpReason: null,
            materializedPackId: "pack-ed1142ba",
            promoted: true,
            baselinePersisted: true,
            source: {
                command: "learn",
                exportDigest: "digest-runtime"
            }
        });
        const surface = buildTracedLearningStatusSurface(activationRoot, {
            env: {
                OPENCLAWBRAIN_ROOT: brainRoot
            }
        });
        assert.equal(surface.path, runtimePath);
        assert.equal(surface.present, true);
        assert.equal(surface.routeTraceCount, 264);
        assert.equal(surface.source?.command, "learn");
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
