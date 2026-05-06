// OpenClawBrain v0.2 — SQLite memory store
// Schema, migrations, CRUD, FTS5 search, graph edges, injections,
// route decisions, proof events, job queue, audit rows.
import path from 'node:path';
import os from 'node:os';
import { mkdirSync } from 'node:fs';
import { openDatabase } from './sqlite-driver.js';
import { filterMemoriesForScope } from './scope.js';
// ── Schema version ────────────────────────────────────────────────────────────
const SCHEMA_VERSION = 8;
// ── Schema SQL ────────────────────────────────────────────────────────────────
const MIGRATIONS = {
    1: `
    CREATE TABLE IF NOT EXISTS schema_meta (
      version INTEGER PRIMARY KEY,
      applied_at TEXT NOT NULL
    );

    -- Memory graph
    CREATE TABLE IF NOT EXISTS memory_nodes (
      rowid INTEGER PRIMARY KEY,
      id TEXT NOT NULL UNIQUE,
      agent_id TEXT NOT NULL,
      type TEXT NOT NULL CHECK (type IN ('correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'recall_rule', 'outcome', 'context')),
      content TEXT NOT NULL,
      positive TEXT,
      negative TEXT,
      scope_kind TEXT NOT NULL DEFAULT 'agent',
      scope_key TEXT,
      normalized_key TEXT,
      tags_json TEXT NOT NULL DEFAULT '[]',
      importance REAL NOT NULL DEFAULT 0.25,
      freshness REAL NOT NULL DEFAULT 1.0,
      confidence REAL NOT NULL DEFAULT 0.5,
      use_count INTEGER NOT NULL DEFAULT 0,
      useful_count INTEGER NOT NULL DEFAULT 0,
      capture_count INTEGER NOT NULL DEFAULT 1,
      distilled_by_model TEXT,
      distiller_prompt_version TEXT,
      distillation_confidence REAL,
      evidence_kind TEXT,
      evidence_hash TEXT,
      source_hook TEXT,
      source_turn_id TEXT,
      source_session_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_seen_at TEXT NOT NULL,
      last_used_at TEXT,
      superseded_by TEXT,
      deleted_at TEXT,
      UNIQUE(agent_id, type, normalized_key, scope_kind, scope_key)
    );

    CREATE INDEX IF NOT EXISTS idx_memory_nodes_agent ON memory_nodes(agent_id);
    CREATE INDEX IF NOT EXISTS idx_memory_nodes_type ON memory_nodes(type);
    CREATE INDEX IF NOT EXISTS idx_memory_nodes_key ON memory_nodes(normalized_key);
    CREATE INDEX IF NOT EXISTS idx_memory_nodes_active ON memory_nodes(agent_id, deleted_at);

    -- Memory edges
    CREATE TABLE IF NOT EXISTS memory_edges (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      from_id TEXT NOT NULL,
      to_id TEXT NOT NULL,
      relation TEXT NOT NULL CHECK (
        relation IN ('related', 'contradicts', 'supersedes', 'extends', 'used_with', 'supports_workflow')
      ),
      weight REAL NOT NULL DEFAULT 0.5,
      evidence_count INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(agent_id, from_id, to_id, relation)
    );

    CREATE INDEX IF NOT EXISTS idx_edges_from ON memory_edges(from_id);
    CREATE INDEX IF NOT EXISTS idx_edges_to ON memory_edges(to_id);

    -- FTS5 virtual table
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
      content, tags, normalized_key,
      content='memory_nodes',
      content_rowid='rowid',
      tokenize='porter unicode61'
    );

    -- FTS5 triggers
    CREATE TRIGGER IF NOT EXISTS memory_nodes_ai AFTER INSERT ON memory_nodes
    WHEN new.deleted_at IS NULL
    BEGIN
      INSERT INTO memory_search(rowid, content, tags, normalized_key)
      VALUES (new.rowid, new.content, new.tags_json, COALESCE(new.normalized_key, ''));
    END;

    CREATE TRIGGER IF NOT EXISTS memory_nodes_ad AFTER DELETE ON memory_nodes
    BEGIN
      INSERT INTO memory_search(memory_search, rowid, content, tags, normalized_key)
      VALUES ('delete', old.rowid, old.content, old.tags_json, COALESCE(old.normalized_key, ''));
    END;

    CREATE TRIGGER IF NOT EXISTS memory_nodes_au AFTER UPDATE ON memory_nodes
    BEGIN
      INSERT INTO memory_search(memory_search, rowid, content, tags, normalized_key)
      VALUES ('delete', old.rowid, old.content, old.tags_json, COALESCE(old.normalized_key, ''));
      INSERT INTO memory_search(rowid, content, tags, normalized_key)
      SELECT new.rowid, new.content, new.tags_json, COALESCE(new.normalized_key, '')
      WHERE new.deleted_at IS NULL;
    END;

    -- Injection events
    CREATE TABLE IF NOT EXISTS memory_injections (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      memory_id TEXT NOT NULL,
      route_decision_id TEXT,
      run_id TEXT,
      turn_id TEXT,
      session_id TEXT,
      query TEXT NOT NULL,
      rank INTEGER NOT NULL,
      score REAL NOT NULL,
      injected_at TEXT NOT NULL,
      resolved_at TEXT,
      outcome TEXT CHECK (
        outcome IN ('pending', 'helped', 'accepted', 'ignored', 'assistant_failed_to_use',
                     'user_corrected', 'harmful', 'tool_success', 'tool_failure', 'unknown')
      ) DEFAULT 'pending',
      correction_signal TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_injections_agent ON memory_injections(agent_id);
    CREATE INDEX IF NOT EXISTS idx_injections_outcome ON memory_injections(outcome);

    -- Route decisions
    CREATE TABLE IF NOT EXISTS route_decisions (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      session_id TEXT,
      turn_id TEXT,
      run_id TEXT,
      route TEXT NOT NULL,
      confidence REAL NOT NULL,
      latency_tier TEXT NOT NULL,
      sync_llm_used INTEGER NOT NULL DEFAULT 0,
      sync_latency_ms INTEGER,
      fallback_used INTEGER NOT NULL DEFAULT 0,
      turn_frame_json TEXT NOT NULL,
      retrieval_plan_json TEXT NOT NULL,
      injection_plan_json TEXT NOT NULL,
      selected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
      omitted_memory_ids_json TEXT NOT NULL DEFAULT '[]',
      model TEXT,
      prompt_version TEXT,
      policy_snapshot_id TEXT,
      outcome TEXT DEFAULT 'pending',
      reward REAL DEFAULT 0,
      created_at TEXT NOT NULL,
      resolved_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_route_agent ON route_decisions(agent_id);
    CREATE INDEX IF NOT EXISTS idx_route_outcome ON route_decisions(outcome);

    -- Route examples
    CREATE TABLE IF NOT EXISTS route_examples (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      turn_frame_json TEXT NOT NULL,
      route_decision_json TEXT NOT NULL,
      outcome TEXT NOT NULL,
      reward REAL NOT NULL,
      lesson TEXT NOT NULL,
      tags_json TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL
    );

    -- Route policy snapshots
    CREATE TABLE IF NOT EXISTS route_policy_snapshots (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      policy_text TEXT NOT NULL,
      examples_json TEXT NOT NULL DEFAULT '[]',
      model TEXT,
      prompt_version TEXT,
      created_at TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_policy_active ON route_policy_snapshots(agent_id, active);

    -- Distillation runs (LLM audit)
    CREATE TABLE IF NOT EXISTS distillation_runs (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      session_id TEXT,
      turn_id TEXT,
      run_id TEXT,
      phase TEXT NOT NULL,
      model TEXT NOT NULL,
      prompt_version TEXT NOT NULL,
      input_hash TEXT NOT NULL,
      redacted_input_summary TEXT,
      output_json TEXT NOT NULL,
      validation_status TEXT NOT NULL,
      validation_error TEXT,
      latency_ms INTEGER,
      created_at TEXT NOT NULL
    );

    -- Job queue
    CREATE TABLE IF NOT EXISTS background_jobs (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',
      priority INTEGER NOT NULL DEFAULT 0,
      payload_json TEXT NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      max_attempts INTEGER NOT NULL DEFAULT 3,
      available_at TEXT NOT NULL,
      started_at TEXT,
      finished_at TEXT,
      error TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_jobs_status ON background_jobs(status, available_at);
    CREATE INDEX IF NOT EXISTS idx_jobs_agent ON background_jobs(agent_id);

    -- Proof events (v0.2 SQLite-backed)
    CREATE TABLE IF NOT EXISTS proof_events (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      created_at TEXT NOT NULL,
      source_hook TEXT,
      turn_id TEXT,
      session_id TEXT,
      run_id TEXT,
      memory_id TEXT,
      injection_id TEXT,
      route_decision_id TEXT,
      distillation_run_id TEXT,
      raw_transcript_stored INTEGER NOT NULL DEFAULT 0,
      payload_json TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_proof_agent ON proof_events(agent_id);
  `,
    2: `
    CREATE TABLE IF NOT EXISTS status_snapshots (
      agent_id TEXT PRIMARY KEY,
      status_json TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
  `,
    3: `
    DROP TRIGGER IF EXISTS memory_nodes_ai;
    DROP TRIGGER IF EXISTS memory_nodes_ad;
    DROP TRIGGER IF EXISTS memory_nodes_au;
    DROP TABLE IF EXISTS memory_search;

    CREATE TABLE IF NOT EXISTS memory_nodes_v3 (
      rowid INTEGER PRIMARY KEY,
      id TEXT NOT NULL UNIQUE,
      agent_id TEXT NOT NULL,
      type TEXT NOT NULL CHECK (type IN ('correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'recall_rule', 'outcome', 'context')),
      content TEXT NOT NULL,
      positive TEXT,
      negative TEXT,
      scope_kind TEXT NOT NULL DEFAULT 'agent',
      scope_key TEXT,
      normalized_key TEXT,
      tags_json TEXT NOT NULL DEFAULT '[]',
      importance REAL NOT NULL DEFAULT 0.25,
      freshness REAL NOT NULL DEFAULT 1.0,
      confidence REAL NOT NULL DEFAULT 0.5,
      use_count INTEGER NOT NULL DEFAULT 0,
      useful_count INTEGER NOT NULL DEFAULT 0,
      capture_count INTEGER NOT NULL DEFAULT 1,
      distilled_by_model TEXT,
      distiller_prompt_version TEXT,
      distillation_confidence REAL,
      evidence_kind TEXT,
      evidence_hash TEXT,
      source_hook TEXT,
      source_turn_id TEXT,
      source_session_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_seen_at TEXT NOT NULL,
      last_used_at TEXT,
      superseded_by TEXT,
      deleted_at TEXT,
      UNIQUE(agent_id, type, normalized_key, scope_kind, scope_key)
    );

    INSERT OR IGNORE INTO memory_nodes_v3 (
      rowid, id, agent_id, type, content, positive, negative,
      scope_kind, scope_key, normalized_key, tags_json,
      importance, freshness, confidence, use_count, useful_count, capture_count,
      distilled_by_model, distiller_prompt_version, distillation_confidence,
      evidence_kind, evidence_hash, source_hook, source_turn_id, source_session_id,
      created_at, updated_at, last_seen_at, last_used_at, superseded_by, deleted_at
    )
    SELECT rowid, id, agent_id, type, content, positive, negative,
      scope_kind, scope_key, normalized_key, tags_json,
      importance, freshness, confidence, use_count, useful_count, capture_count,
      distilled_by_model, distiller_prompt_version, distillation_confidence,
      evidence_kind, evidence_hash, source_hook, source_turn_id, source_session_id,
      created_at, updated_at, last_seen_at, last_used_at, superseded_by, deleted_at
    FROM memory_nodes;

    DROP TABLE IF EXISTS memory_nodes;
    ALTER TABLE memory_nodes_v3 RENAME TO memory_nodes;

    CREATE INDEX IF NOT EXISTS idx_memory_nodes_agent ON memory_nodes(agent_id);
    CREATE INDEX IF NOT EXISTS idx_memory_nodes_type ON memory_nodes(type);
    CREATE INDEX IF NOT EXISTS idx_memory_nodes_key ON memory_nodes(normalized_key);
    CREATE INDEX IF NOT EXISTS idx_memory_nodes_active ON memory_nodes(agent_id, deleted_at);

    CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
      content, tags, normalized_key,
      content='memory_nodes',
      content_rowid='rowid',
      tokenize='porter unicode61'
    );

    CREATE TRIGGER IF NOT EXISTS memory_nodes_ai AFTER INSERT ON memory_nodes
    WHEN new.deleted_at IS NULL
    BEGIN
      INSERT INTO memory_search(rowid, content, tags, normalized_key)
      VALUES (new.rowid, new.content, new.tags_json, COALESCE(new.normalized_key, ''));
    END;

    CREATE TRIGGER IF NOT EXISTS memory_nodes_ad AFTER DELETE ON memory_nodes
    BEGIN
      INSERT INTO memory_search(memory_search, rowid, content, tags, normalized_key)
      VALUES ('delete', old.rowid, old.content, old.tags_json, COALESCE(old.normalized_key, ''));
    END;

    CREATE TRIGGER IF NOT EXISTS memory_nodes_au AFTER UPDATE ON memory_nodes
    BEGIN
      INSERT INTO memory_search(memory_search, rowid, content, tags, normalized_key)
      VALUES ('delete', old.rowid, old.content, old.tags_json, COALESCE(old.normalized_key, ''));
      INSERT INTO memory_search(rowid, content, tags, normalized_key)
      SELECT new.rowid, new.content, new.tags_json, COALESCE(new.normalized_key, '')
      WHERE new.deleted_at IS NULL;
    END;

    INSERT INTO memory_search(rowid, content, tags, normalized_key)
    SELECT rowid, content, tags_json, COALESCE(normalized_key, '')
    FROM memory_nodes
    WHERE deleted_at IS NULL;

    CREATE TABLE IF NOT EXISTS capture_audit (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      turn_id TEXT,
      session_id TEXT,
      run_id TEXT,
      created_at TEXT NOT NULL,
      retrieval_intent_json TEXT NOT NULL,
      capture_intent_json TEXT NOT NULL,
      capture_job_created INTEGER NOT NULL DEFAULT 0,
      distiller_ran INTEGER NOT NULL DEFAULT 0,
      distiller_model TEXT,
      distiller_latency_ms INTEGER,
      fallback_ran INTEGER NOT NULL DEFAULT 0,
      candidate_count INTEGER NOT NULL DEFAULT 0,
      stored_count INTEGER NOT NULL DEFAULT 0,
      rejected_count INTEGER NOT NULL DEFAULT 0,
      rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
      safe_candidate_preview TEXT,
      evidence_hash TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_capture_audit_agent ON capture_audit(agent_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_capture_audit_turn ON capture_audit(turn_id);
  `,
    4: `
    UPDATE memory_nodes
    SET scope_key = ''
    WHERE scope_key IS NULL
      AND NOT EXISTS (
        SELECT 1 FROM memory_nodes AS other
        WHERE other.agent_id = memory_nodes.agent_id
          AND other.type = memory_nodes.type
          AND COALESCE(other.normalized_key, '') = COALESCE(memory_nodes.normalized_key, '')
          AND other.scope_kind = memory_nodes.scope_kind
          AND other.scope_key = ''
          AND other.id != memory_nodes.id
      );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_active_dedupe
      ON background_jobs(agent_id, kind, dedupe_key)
      WHERE dedupe_key IS NOT NULL AND status IN ('pending', 'running');
  `,
    5: `
    CREATE TABLE IF NOT EXISTS route_graph_snapshots (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      route_decision_id TEXT NOT NULL UNIQUE,
      query_set_json TEXT NOT NULL DEFAULT '[]',
      candidate_memory_ids_json TEXT NOT NULL DEFAULT '[]',
      candidate_summaries_json TEXT NOT NULL DEFAULT '[]',
      graph_stats_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_route_graph_agent ON route_graph_snapshots(agent_id, created_at);

    CREATE TABLE IF NOT EXISTS route_teacher_runs (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      route_decision_id TEXT NOT NULL,
      model TEXT NOT NULL,
      prompt_version TEXT NOT NULL,
      input_hash TEXT NOT NULL,
      output_hash TEXT NOT NULL,
      verdict TEXT NOT NULL,
      teacher_route TEXT NOT NULL,
      teacher_memory_ids_json TEXT NOT NULL DEFAULT '[]',
      teacher_queries_json TEXT NOT NULL DEFAULT '[]',
      teacher_graph_depth INTEGER NOT NULL DEFAULT 0,
      sync_planner_worth_it INTEGER NOT NULL DEFAULT 0,
      confidence REAL NOT NULL DEFAULT 0,
      rationale TEXT NOT NULL DEFAULT '',
      validated INTEGER NOT NULL DEFAULT 0,
      rejection_reason TEXT,
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_route_teacher_agent ON route_teacher_runs(agent_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_route_teacher_decision ON route_teacher_runs(route_decision_id);

    CREATE TABLE IF NOT EXISTS route_counterfactuals (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      route_teacher_run_id TEXT NOT NULL,
      route_decision_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      memory_ids_json TEXT NOT NULL DEFAULT '[]',
      memory_types_json TEXT NOT NULL DEFAULT '[]',
      graph_depth INTEGER NOT NULL DEFAULT 0,
      estimated_outcome TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0,
      rationale TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_route_counterfactuals_decision ON route_counterfactuals(route_decision_id);
    CREATE INDEX IF NOT EXISTS idx_route_counterfactuals_teacher ON route_counterfactuals(route_teacher_run_id);

    CREATE TABLE IF NOT EXISTS route_training_examples_v2 (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      route_decision_id TEXT NOT NULL,
      route_teacher_run_id TEXT,
      example_kind TEXT NOT NULL,
      task_type TEXT NOT NULL,
      turn_signals_json TEXT NOT NULL DEFAULT '[]',
      route TEXT NOT NULL,
      memory_types_json TEXT NOT NULL DEFAULT '[]',
      query_templates_json TEXT NOT NULL DEFAULT '[]',
      graph_depth INTEGER NOT NULL DEFAULT 0,
      confidence REAL NOT NULL DEFAULT 0,
      support_count INTEGER NOT NULL DEFAULT 1,
      harm_count INTEGER NOT NULL DEFAULT 0,
      source TEXT NOT NULL,
      evidence_ids_json TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_route_examples_v2_agent ON route_training_examples_v2(agent_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_route_examples_v2_decision ON route_training_examples_v2(route_decision_id);

    CREATE TABLE IF NOT EXISTS route_policy_snapshots_v2 (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      version TEXT NOT NULL DEFAULT 'route-policy-v2',
      status TEXT NOT NULL DEFAULT 'candidate',
      rules_json TEXT NOT NULL DEFAULT '[]',
      global_budgets_json TEXT NOT NULL DEFAULT '{}',
      eval_summary_json TEXT,
      example_ids_json TEXT NOT NULL DEFAULT '[]',
      model TEXT,
      prompt_version TEXT,
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_policy_v2_active ON route_policy_snapshots_v2(agent_id, status, created_at);
  `,
    6: `
    CREATE TABLE IF NOT EXISTS route_frames (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      session_key_hash TEXT,
      turn_hash TEXT NOT NULL,
      redacted_turn_summary TEXT NOT NULL,
      task_type TEXT NOT NULL,
      turn_signals_json TEXT NOT NULL DEFAULT '[]',
      intent_signals_json TEXT NOT NULL DEFAULT '[]',
      safety_signals_json TEXT NOT NULL DEFAULT '[]',
      project_hint TEXT,
      repo_hint TEXT,
      latency_budget_ms INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_route_frames_agent ON route_frames(agent_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_route_frames_turn_hash ON route_frames(turn_hash);
  `,
    7: `
    CREATE TABLE IF NOT EXISTS route_frames_v3 (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      route_decision_id TEXT NOT NULL,
      route_frame_id TEXT,
      redacted_turn_summary TEXT NOT NULL,
      task_type TEXT NOT NULL,
      turn_signals_json TEXT NOT NULL DEFAULT '[]',
      project_hint TEXT,
      repo_hint TEXT,
      tool_hints_json TEXT NOT NULL DEFAULT '[]',
      route_hint_flags_json TEXT NOT NULL DEFAULT '[]',
      chosen_action_id TEXT NOT NULL,
      chosen_route TEXT NOT NULL,
      chosen_memory_types_json TEXT NOT NULL DEFAULT '[]',
      chosen_graph_depth INTEGER NOT NULL DEFAULT 0,
      chosen_sync_planner TEXT NOT NULL DEFAULT 'no',
      policy_snapshot_id TEXT,
      policy_rule_id TEXT,
      outcome TEXT,
      reward REAL NOT NULL DEFAULT 0,
      reward_components_json TEXT,
      payload_hash TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_route_frames_v3_agent ON route_frames_v3(agent_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_route_frames_v3_decision ON route_frames_v3(route_decision_id);
    CREATE INDEX IF NOT EXISTS idx_route_frames_v3_action ON route_frames_v3(agent_id, chosen_action_id, created_at);

    CREATE TABLE IF NOT EXISTS route_action_prototypes_v3 (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      route TEXT NOT NULL,
      memory_types_json TEXT NOT NULL DEFAULT '[]',
      graph_depth INTEGER NOT NULL DEFAULT 0,
      sync_planner TEXT NOT NULL DEFAULT 'no',
      query_template_family_json TEXT NOT NULL DEFAULT '[]',
      sparse_signature_json TEXT NOT NULL DEFAULT '[]',
      dense_embedding_json TEXT NOT NULL DEFAULT '[]',
      support_prior REAL NOT NULL DEFAULT 0,
      harm_prior REAL NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'active',
      provenance TEXT NOT NULL DEFAULT 'learned',
      source_example_ids_json TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_route_action_prototypes_v3_agent ON route_action_prototypes_v3(agent_id, updated_at);

    CREATE TABLE IF NOT EXISTS route_pair_examples_v3 (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      frame_id TEXT NOT NULL,
      positive_action_id TEXT NOT NULL,
      negative_action_id TEXT NOT NULL,
      label_source TEXT NOT NULL,
      margin_weight REAL NOT NULL DEFAULT 1,
      evidence_ids_json TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_route_pair_examples_v3_agent ON route_pair_examples_v3(agent_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_route_pair_examples_v3_frame ON route_pair_examples_v3(frame_id);

    CREATE TABLE IF NOT EXISTS route_bandit_feedback_v3 (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      frame_id TEXT NOT NULL,
      chosen_action_id TEXT NOT NULL,
      reward REAL NOT NULL DEFAULT 0,
      reward_components_json TEXT NOT NULL DEFAULT '{}',
      cost REAL NOT NULL DEFAULT 0,
      latency_ms INTEGER NOT NULL DEFAULT 0,
      outcome_label TEXT NOT NULL DEFAULT 'ambiguous',
      learning_bucket INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_route_bandit_feedback_v3_agent ON route_bandit_feedback_v3(agent_id, created_at);

    CREATE TABLE IF NOT EXISTS route_bandit_state_v3 (
      agent_id TEXT PRIMARY KEY,
      learner_version TEXT NOT NULL,
      feature_schema_version TEXT NOT NULL,
      exploration_alpha REAL NOT NULL DEFAULT 0.35,
      shared_weights_json TEXT NOT NULL DEFAULT '[]',
      action_stats_json TEXT NOT NULL DEFAULT '{}',
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS route_policy_snapshots_v3 (
      id TEXT PRIMARY KEY,
      agent_id TEXT NOT NULL,
      version TEXT NOT NULL DEFAULT 'route-policy-v3',
      status TEXT NOT NULL DEFAULT 'candidate',
      rules_json TEXT NOT NULL DEFAULT '[]',
      action_priors_json TEXT NOT NULL DEFAULT '{}',
      global_budgets_json TEXT NOT NULL DEFAULT '{}',
      eval_summary_json TEXT,
      calibration_json TEXT,
      lineage_json TEXT,
      source_frame_ids_json TEXT NOT NULL DEFAULT '[]',
      source_prototype_ids_json TEXT NOT NULL DEFAULT '[]',
      model TEXT,
      prompt_version TEXT,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_policy_v3_active ON route_policy_snapshots_v3(agent_id, status, created_at);
  `,
    8: `
    -- v8: persist route-policy-v3 calibration + lineage JSON
  `,
};
// ── UUID helper ───────────────────────────────────────────────────────────────
import { randomUUID } from 'node:crypto';
export const uuid = () => randomUUID();
export const now = () => new Date().toISOString();
export class MemoryStore {
    db;
    dbPath;
    ownerAgentId;
    constructor(options) {
        this.ownerAgentId = options.agentId;
        this.dbPath = dbPathForAgent(options.activationRoot, options.agentId);
        this.db = openDb(this.dbPath);
        this.migrate();
    }
    // ── Migration ───────────────────────────────────────────────────────────────
    migrate() {
        const current = this.db.pragma('user_version', { simple: true });
        for (let v = current + 1; v <= SCHEMA_VERSION; v++) {
            const sql = MIGRATIONS[v];
            if (!sql)
                continue;
            if (v === 4 && !tableHasColumn(this.db, 'background_jobs', 'dedupe_key')) {
                this.db.exec('ALTER TABLE background_jobs ADD COLUMN dedupe_key TEXT;');
            }
            if (v === 6) {
                if (!tableHasColumn(this.db, 'route_decisions', 'route_frame_id'))
                    this.db.exec('ALTER TABLE route_decisions ADD COLUMN route_frame_id TEXT;');
                if (!tableHasColumn(this.db, 'route_decisions', 'policy_rule_id'))
                    this.db.exec('ALTER TABLE route_decisions ADD COLUMN policy_rule_id TEXT;');
                if (!tableHasColumn(this.db, 'route_decisions', 'candidate_count'))
                    this.db.exec('ALTER TABLE route_decisions ADD COLUMN candidate_count INTEGER DEFAULT 0;');
                if (!tableHasColumn(this.db, 'route_decisions', 'reason_code'))
                    this.db.exec('ALTER TABLE route_decisions ADD COLUMN reason_code TEXT;');
                if (!tableHasColumn(this.db, 'route_decisions', 'injection_payload_hash'))
                    this.db.exec('ALTER TABLE route_decisions ADD COLUMN injection_payload_hash TEXT;');
            }
            if (v === 8) {
                if (!tableHasColumn(this.db, 'route_policy_snapshots_v3', 'calibration_json'))
                    this.db.exec('ALTER TABLE route_policy_snapshots_v3 ADD COLUMN calibration_json TEXT;');
                if (!tableHasColumn(this.db, 'route_policy_snapshots_v3', 'lineage_json'))
                    this.db.exec('ALTER TABLE route_policy_snapshots_v3 ADD COLUMN lineage_json TEXT;');
            }
            this.db.exec(sql);
            this.db.pragma(`user_version = ${v}`);
            this.db.prepare('INSERT OR REPLACE INTO schema_meta (version, applied_at) VALUES (?, ?)').run(v, now());
        }
    }
    close() {
        this.db.close();
    }
    // ── Memory nodes ────────────────────────────────────────────────────────────
    insertMemory(node) {
        const id = node.id || uuid();
        const ts = now();
        this.db.prepare(`
      INSERT INTO memory_nodes (
        id, agent_id, type, content, positive, negative,
        scope_kind, scope_key, normalized_key, tags_json,
        importance, freshness, confidence, use_count, useful_count, capture_count,
        distilled_by_model, distiller_prompt_version, distillation_confidence,
        evidence_kind, evidence_hash, source_hook, source_turn_id, source_session_id,
        created_at, updated_at, last_seen_at, last_used_at, superseded_by, deleted_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, node.agentId, node.type, node.content, node.positive ?? null, node.negative ?? null, node.scopeKind, node.scopeKey ?? '', node.normalizedKey, JSON.stringify(node.tags), node.importance, node.freshness, node.confidence, node.useCount, node.usefulCount, node.captureCount, node.distilledByModel ?? null, node.distillerPromptVersion ?? null, node.distillationConfidence ?? null, node.evidenceKind ?? null, node.evidenceHash ?? null, node.sourceHook ?? null, node.sourceTurnId ?? null, node.sourceSessionId ?? null, ts, ts, ts, null, null, null);
        return this.getMemory(id);
    }
    getMemory(id) {
        const row = this.db.prepare('SELECT * FROM memory_nodes WHERE id = ?').get(id);
        return row ? rowToMemory(row) : null;
    }
    findMemoryByNormalizedKey(agentId, normalizedKey, scopeKind, scopeKey) {
        const row = this.db.prepare(`
      SELECT * FROM memory_nodes
      WHERE agent_id = ? AND normalized_key = ? AND scope_kind = ? AND (scope_key = ? OR (scope_key IS NULL AND ? = '')) AND deleted_at IS NULL
      LIMIT 1
    `).get(agentId, normalizedKey, scopeKind, scopeKey ?? '', scopeKey ?? '');
        return row ? rowToMemory(row) : null;
    }
    updateMemory(id, updates) {
        const existing = this.getMemory(id);
        if (!existing)
            return null;
        const merged = { ...existing, ...updates, updatedAt: now() };
        this.db.prepare(`
      UPDATE memory_nodes SET
        content = ?, positive = ?, negative = ?,
        tags_json = ?, importance = ?, freshness = ?, confidence = ?,
        use_count = ?, useful_count = ?, capture_count = ?,
        distilled_by_model = ?, distiller_prompt_version = ?, distillation_confidence = ?,
        evidence_kind = ?, evidence_hash = ?, source_hook = ?, source_turn_id = ?, source_session_id = ?,
        updated_at = ?, last_seen_at = ?, last_used_at = ?,
        superseded_by = ?, deleted_at = ?
      WHERE id = ?
    `).run(merged.content, merged.positive ?? null, merged.negative ?? null, JSON.stringify(merged.tags), merged.importance, merged.freshness, merged.confidence, merged.useCount, merged.usefulCount, merged.captureCount, merged.distilledByModel ?? null, merged.distillerPromptVersion ?? null, merged.distillationConfidence ?? null, merged.evidenceKind ?? null, merged.evidenceHash ?? null, merged.sourceHook ?? null, merged.sourceTurnId ?? null, merged.sourceSessionId ?? null, merged.updatedAt, merged.lastSeenAt, merged.lastUsedAt ?? null, merged.supersededBy ?? null, merged.deletedAt ?? null, id);
        return this.getMemory(id);
    }
    supersedeMemory(existingId, supersededById) {
        this.db.prepare('UPDATE memory_nodes SET superseded_by = ?, updated_at = ? WHERE id = ?').run(supersededById, now(), existingId);
    }
    softDeleteMemory(id) {
        this.db.prepare('UPDATE memory_nodes SET deleted_at = ?, updated_at = ? WHERE id = ?').run(now(), now(), id);
    }
    searchMemories(query, agentId, opts = {}) {
        const limit = Math.min(50, Math.max(1, opts.limit ?? 20));
        const sqlLimit = opts.scopeContext ? Math.min(500, limit * 10) : limit;
        const offset = opts.offset ?? 0;
        const trimmed = query.trim();
        let rows;
        if (!trimmed) {
            rows = this.db.prepare(`
        SELECT * FROM memory_nodes
        WHERE agent_id = ? AND deleted_at IS NULL AND superseded_by IS NULL
        ORDER BY importance DESC, updated_at DESC
        LIMIT ? OFFSET ?
      `).all(agentId, sqlLimit, offset);
        }
        else {
            try {
                rows = this.db.prepare(`
          SELECT mn.* FROM memory_search
          JOIN memory_nodes mn ON mn.rowid = memory_search.rowid
          WHERE memory_search MATCH ? AND mn.agent_id = ? AND mn.deleted_at IS NULL AND mn.superseded_by IS NULL
          ORDER BY rank
          LIMIT ? OFFSET ?
        `).all(trimmed, agentId, sqlLimit, offset);
            }
            catch {
                const lowerTokens = trimmed.toLowerCase().split(/[^a-z0-9_]+/i).filter(Boolean);
                rows = this.db.prepare(`
          SELECT * FROM memory_nodes
          WHERE agent_id = ? AND deleted_at IS NULL AND superseded_by IS NULL
          ORDER BY importance DESC, updated_at DESC
        `).all(agentId).filter((row) => {
                    const haystack = `${row.content} ${row.normalized_key ?? ''} ${row.tags_json ?? ''}`.toLowerCase();
                    return lowerTokens.every((token) => haystack.includes(token));
                }).slice(offset, offset + sqlLimit);
            }
        }
        return filterMemoriesForScope(rows.map(rowToMemory), opts.scopeContext).slice(0, limit);
    }
    listMemories(agentId, opts = {}) {
        const limit = Math.min(200, Math.max(1, opts.limit ?? 50));
        const sqlLimit = opts.scopeContext ? Math.min(1000, limit * 10) : limit;
        if (opts.type) {
            return filterMemoriesForScope(this.db.prepare('SELECT * FROM memory_nodes WHERE agent_id = ? AND type = ? AND deleted_at IS NULL ORDER BY importance DESC LIMIT ?').all(agentId, opts.type, sqlLimit).map(rowToMemory), opts.scopeContext).slice(0, limit);
        }
        return filterMemoriesForScope(this.db.prepare('SELECT * FROM memory_nodes WHERE agent_id = ? AND deleted_at IS NULL ORDER BY importance DESC LIMIT ?').all(agentId, sqlLimit).map(rowToMemory), opts.scopeContext).slice(0, limit);
    }
    countMemories(agentId, type) {
        if (type) {
            const row = this.db.prepare('SELECT COUNT(*) as cnt FROM memory_nodes WHERE agent_id = ? AND type = ? AND deleted_at IS NULL').get(agentId, type);
            return row?.cnt ?? 0;
        }
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM memory_nodes WHERE agent_id = ? AND deleted_at IS NULL').get(agentId);
        return row?.cnt ?? 0;
    }
    // ── Memory edges ────────────────────────────────────────────────────────────
    insertEdge(edge) {
        const id = edge.id || uuid();
        const ts = now();
        this.db.prepare(`
      INSERT INTO memory_edges (id, agent_id, from_id, to_id, relation, weight, evidence_count, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, edge.agentId, edge.fromId, edge.toId, edge.relation, edge.weight, edge.evidenceCount, ts, ts);
        return { ...edge, id, createdAt: ts, updatedAt: ts };
    }
    upsertEdge(agentId, fromId, toId, relation) {
        const existing = this.db.prepare('SELECT * FROM memory_edges WHERE agent_id = ? AND from_id = ? AND to_id = ? AND relation = ?').get(agentId, fromId, toId, relation);
        if (existing) {
            const ts = now();
            this.db.prepare('UPDATE memory_edges SET evidence_count = evidence_count + 1, weight = ?, updated_at = ? WHERE id = ?')
                .run(Math.min(1.0, existing.weight + 0.1), ts, existing.id);
            return rowToEdge(this.db.prepare('SELECT * FROM memory_edges WHERE id = ?').get(existing.id));
        }
        return this.insertEdge({ agentId, fromId, toId, relation, weight: 0.5, evidenceCount: 1 });
    }
    getEdges(memoryId, relation) {
        if (relation) {
            return this.db.prepare('SELECT * FROM memory_edges WHERE (from_id = ? OR to_id = ?) AND relation = ?').all(memoryId, memoryId, relation).map(rowToEdge);
        }
        return this.db.prepare('SELECT * FROM memory_edges WHERE from_id = ? OR to_id = ?').all(memoryId, memoryId).map(rowToEdge);
    }
    // ── Injection events ────────────────────────────────────────────────────────
    insertInjection(inj) {
        const id = inj.id || uuid();
        const ts = inj.injectedAt || now();
        const outcome = inj.outcome ?? 'pending';
        this.db.prepare(`
      INSERT INTO memory_injections (id, agent_id, memory_id, route_decision_id, run_id, turn_id, session_id, query, rank, score, injected_at, resolved_at, outcome, correction_signal)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, inj.agentId, inj.memoryId, inj.routeDecisionId ?? null, inj.runId ?? null, inj.turnId ?? null, inj.sessionId ?? null, inj.query, inj.rank, inj.score, ts, null, outcome, inj.correctionSignal ?? null);
        return { ...inj, id, injectedAt: ts, outcome };
    }
    resolveInjectionOutcome(injectionId, outcome, correctionSignal, scope = {}) {
        const clauses = ['id = ?'];
        const params = [injectionId];
        if (scope.agentId) {
            clauses.push('agent_id = ?');
            params.push(scope.agentId);
        }
        if (scope.runId) {
            clauses.push('run_id = ?');
            params.push(scope.runId);
        }
        if (scope.turnId) {
            clauses.push('turn_id = ?');
            params.push(scope.turnId);
        }
        if (scope.sessionId) {
            clauses.push('session_id = ?');
            params.push(scope.sessionId);
        }
        const info = this.db.prepare(`UPDATE memory_injections SET outcome = ?, correction_signal = ?, resolved_at = ? WHERE ${clauses.join(' AND ')}`)
            .run(outcome, correctionSignal ?? null, now(), ...params);
        return Number(info?.changes || 0);
    }
    getPendingInjections(agentId) {
        return this.db.prepare('SELECT * FROM memory_injections WHERE agent_id = ? AND outcome = ? ORDER BY injected_at DESC LIMIT 100').all(agentId, 'pending').map(rowToInjection);
    }
    getInjectionsForRouteDecision(routeDecisionId) {
        return this.db.prepare('SELECT * FROM memory_injections WHERE route_decision_id = ? ORDER BY rank ASC, injected_at ASC').all(routeDecisionId).map(rowToInjection);
    }
    // ── Route frames and decisions ──────────────────────────────────────────────
    insertRouteFrame(frame) {
        const id = frame.id || uuid();
        const ts = frame.createdAt || now();
        this.db.prepare(`
      INSERT INTO route_frames (
        id, agent_id, session_key_hash, turn_hash, redacted_turn_summary, task_type,
        turn_signals_json, intent_signals_json, safety_signals_json, project_hint,
        repo_hint, latency_budget_ms, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, frame.agentId, frame.sessionKeyHash ?? null, frame.turnHash, frame.redactedTurnSummary, frame.taskType, JSON.stringify(frame.turnSignals ?? []), JSON.stringify(frame.intentSignals ?? []), JSON.stringify(frame.safetySignals ?? []), frame.projectHint ?? null, frame.repoHint ?? null, frame.latencyBudgetMs, ts);
        return { ...frame, id, createdAt: ts };
    }
    getRouteFrame(id) {
        const row = this.db.prepare('SELECT * FROM route_frames WHERE id = ?').get(id);
        return row ? rowToRouteFrame(row) : null;
    }
    insertRouteFrameV3(frame) {
        const id = frame.id || uuid();
        const ts = frame.createdAt || now();
        this.db.prepare(`
      INSERT INTO route_frames_v3 (
        id, agent_id, route_decision_id, route_frame_id, redacted_turn_summary, task_type,
        turn_signals_json, project_hint, repo_hint, tool_hints_json, route_hint_flags_json,
        chosen_action_id, chosen_route, chosen_memory_types_json, chosen_graph_depth,
        chosen_sync_planner, policy_snapshot_id, policy_rule_id, outcome, reward,
        reward_components_json, payload_hash, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, frame.agentId, frame.routeDecisionId, frame.routeFrameId ?? null, frame.redactedTurnSummary, frame.taskType, JSON.stringify(frame.turnSignals ?? []), frame.projectHint ?? null, frame.repoHint ?? null, JSON.stringify(frame.toolHints ?? []), JSON.stringify(frame.routeHintFlags ?? []), frame.chosenActionId, frame.chosenRoute, JSON.stringify(frame.chosenMemoryTypes ?? []), frame.chosenGraphDepth, frame.chosenSyncPlanner, frame.policySnapshotId ?? null, frame.policyRuleId ?? null, frame.outcome ?? null, frame.reward, frame.rewardComponents ? JSON.stringify(frame.rewardComponents) : null, frame.payloadHash, ts);
        return { ...frame, id, createdAt: ts };
    }
    listRouteFramesV3(agentId, limit = 100) {
        return this.db.prepare('SELECT * FROM route_frames_v3 WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, Math.min(1000, Math.max(1, limit))).map(rowToRouteFrameV3);
    }
    upsertRouteActionPrototypeV3(prototype) {
        const existing = this.db.prepare('SELECT * FROM route_action_prototypes_v3 WHERE id = ?').get(prototype.id);
        const createdAt = existing?.created_at || prototype.createdAt || now();
        const updatedAt = prototype.updatedAt || now();
        if (existing) {
            const mergedSupport = Number(existing.support_prior ?? 0) + Number(prototype.supportPrior ?? 0);
            const mergedHarm = Number(existing.harm_prior ?? 0) + Number(prototype.harmPrior ?? 0);
            const mergedExamples = [...new Set([...(JSON.parse(existing.source_example_ids_json ?? '[]')), ...(prototype.sourceExampleIds ?? [])])];
            this.db.prepare(`
        UPDATE route_action_prototypes_v3 SET
          route = ?, memory_types_json = ?, graph_depth = ?, sync_planner = ?,
          query_template_family_json = ?, sparse_signature_json = ?, dense_embedding_json = ?,
          support_prior = ?, harm_prior = ?, status = ?, provenance = ?, source_example_ids_json = ?, updated_at = ?
        WHERE id = ?
      `).run(prototype.route, JSON.stringify(prototype.memoryTypes ?? []), prototype.graphDepth, prototype.syncPlanner, JSON.stringify(prototype.queryTemplateFamily ?? []), JSON.stringify(prototype.sparseSignature ?? []), JSON.stringify(prototype.denseEmbedding ?? []), mergedSupport, mergedHarm, prototype.status, prototype.provenance, JSON.stringify(mergedExamples), updatedAt, prototype.id);
        }
        else {
            this.db.prepare(`
        INSERT INTO route_action_prototypes_v3 (
          id, agent_id, route, memory_types_json, graph_depth, sync_planner,
          query_template_family_json, sparse_signature_json, dense_embedding_json,
          support_prior, harm_prior, status, provenance, source_example_ids_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(prototype.id, prototype.agentId, prototype.route, JSON.stringify(prototype.memoryTypes ?? []), prototype.graphDepth, prototype.syncPlanner, JSON.stringify(prototype.queryTemplateFamily ?? []), JSON.stringify(prototype.sparseSignature ?? []), JSON.stringify(prototype.denseEmbedding ?? []), prototype.supportPrior ?? 0, prototype.harmPrior ?? 0, prototype.status, prototype.provenance, JSON.stringify(prototype.sourceExampleIds ?? []), createdAt, updatedAt);
        }
        return rowToRouteActionPrototypeV3(this.db.prepare('SELECT * FROM route_action_prototypes_v3 WHERE id = ?').get(prototype.id));
    }
    listRouteActionPrototypesV3(agentId, limit = 100) {
        return this.db.prepare('SELECT * FROM route_action_prototypes_v3 WHERE agent_id = ? ORDER BY updated_at DESC LIMIT ?').all(agentId, Math.min(500, Math.max(1, limit))).map(rowToRouteActionPrototypeV3);
    }
    setRouteActionPrototypeStatusV3(id, status) {
        this.db.prepare('UPDATE route_action_prototypes_v3 SET status = ?, updated_at = ? WHERE id = ?').run(status, now(), id);
        const row = this.db.prepare('SELECT * FROM route_action_prototypes_v3 WHERE id = ?').get(id);
        return row ? rowToRouteActionPrototypeV3(row) : null;
    }
    insertRoutePairExampleV3(example) {
        const id = example.id || uuid();
        const ts = example.createdAt || now();
        this.db.prepare(`
      INSERT INTO route_pair_examples_v3 (
        id, agent_id, frame_id, positive_action_id, negative_action_id, label_source, margin_weight, evidence_ids_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, example.agentId, example.frameId, example.positiveActionId, example.negativeActionId, example.labelSource, example.marginWeight, JSON.stringify(example.evidenceIds ?? []), ts);
        return { ...example, id, createdAt: ts };
    }
    listRoutePairExamplesV3(agentId, limit = 200) {
        return this.db.prepare('SELECT * FROM route_pair_examples_v3 WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, Math.min(2000, Math.max(1, limit))).map(rowToRoutePairExampleV3);
    }
    insertRouteBanditFeedbackV3(feedback) {
        const id = feedback.id || uuid();
        const ts = feedback.createdAt || now();
        this.db.prepare(`
      INSERT INTO route_bandit_feedback_v3 (
        id, agent_id, frame_id, chosen_action_id, reward, reward_components_json, cost, latency_ms, outcome_label, learning_bucket, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, feedback.agentId, feedback.frameId, feedback.chosenActionId, feedback.reward, JSON.stringify(feedback.rewardComponents ?? {}), feedback.cost, feedback.latencyMs, feedback.outcomeLabel, feedback.learningBucket ? 1 : 0, ts);
        return { ...feedback, id, createdAt: ts };
    }
    getRouteBanditStateV3(agentId) {
        const row = this.db.prepare('SELECT * FROM route_bandit_state_v3 WHERE agent_id = ?').get(agentId);
        return row ? rowToRouteBanditStateV3(row) : null;
    }
    upsertRouteBanditStateV3(state) {
        this.db.prepare(`
      INSERT INTO route_bandit_state_v3 (
        agent_id, learner_version, feature_schema_version, exploration_alpha, shared_weights_json, action_stats_json, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(agent_id) DO UPDATE SET
        learner_version = excluded.learner_version,
        feature_schema_version = excluded.feature_schema_version,
        exploration_alpha = excluded.exploration_alpha,
        shared_weights_json = excluded.shared_weights_json,
        action_stats_json = excluded.action_stats_json,
        updated_at = excluded.updated_at
    `).run(state.agentId, state.learnerVersion, state.featureSchemaVersion, state.explorationAlpha, JSON.stringify(state.sharedWeights ?? []), JSON.stringify(state.actionStats ?? {}), state.updatedAt);
        return this.getRouteBanditStateV3(state.agentId);
    }
    insertPolicySnapshotV3(snapshot) {
        const id = snapshot.id || uuid();
        const ts = snapshot.createdAt || now();
        if (snapshot.status === 'active') {
            this.db.prepare("UPDATE route_policy_snapshots_v3 SET status = 'shadow' WHERE agent_id = ? AND status = 'active'").run(snapshot.agentId);
        }
        this.db.prepare(`
      INSERT INTO route_policy_snapshots_v3 (
        id, agent_id, version, status, rules_json, action_priors_json, global_budgets_json, eval_summary_json, calibration_json, lineage_json,
        source_frame_ids_json, source_prototype_ids_json, model, prompt_version, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, snapshot.agentId, snapshot.version, snapshot.status, JSON.stringify(snapshot.rules ?? []), JSON.stringify(snapshot.actionPriors ?? {}), JSON.stringify(snapshot.globalBudgets ?? {}), snapshot.evalSummary ? JSON.stringify(snapshot.evalSummary) : null, snapshot.calibration ? JSON.stringify(snapshot.calibration) : null, snapshot.lineage ? JSON.stringify(snapshot.lineage) : null, JSON.stringify(snapshot.sourceFrameIds ?? []), JSON.stringify(snapshot.sourcePrototypeIds ?? []), snapshot.model ?? null, snapshot.promptVersion ?? null, ts);
        return { ...snapshot, id, createdAt: ts };
    }
    getActivePolicySnapshotV3(agentId) {
        const row = this.db.prepare("SELECT * FROM route_policy_snapshots_v3 WHERE agent_id = ? AND status = 'active' ORDER BY created_at DESC LIMIT 1").get(agentId);
        return row ? rowToRoutePolicySnapshotV3(row) : null;
    }
    listPolicySnapshotsV3(agentId, limit = 20) {
        return this.db.prepare('SELECT * FROM route_policy_snapshots_v3 WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, Math.min(200, Math.max(1, limit))).map(rowToRoutePolicySnapshotV3);
    }
    insertRouteDecision(decision) {
        const id = decision.id || uuid();
        const ts = now();
        const outcome = decision.outcome ?? 'pending';
        const reward = decision.reward ?? 0;
        this.db.prepare(`
      INSERT INTO route_decisions (
        id, agent_id, route_frame_id, session_id, turn_id, run_id, route, confidence, latency_tier,
        sync_llm_used, sync_latency_ms, fallback_used,
        turn_frame_json, retrieval_plan_json, injection_plan_json,
        selected_memory_ids_json, omitted_memory_ids_json,
        model, prompt_version, policy_snapshot_id, policy_rule_id,
        candidate_count, reason_code, injection_payload_hash,
        outcome, reward, created_at, resolved_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, decision.agentId, decision.routeFrameId ?? null, decision.sessionId ?? null, decision.turnId ?? null, decision.runId ?? null, decision.route, decision.confidence, decision.latencyTier, decision.syncLlmUsed ? 1 : 0, decision.syncLatencyMs ?? null, decision.fallbackUsed ? 1 : 0, JSON.stringify(decision.turnFrame), JSON.stringify(decision.retrievalPlan), JSON.stringify(decision.injectionPlan), JSON.stringify(decision.selectedMemoryIds), JSON.stringify(decision.omittedMemoryIds), decision.model ?? null, decision.promptVersion ?? null, decision.policySnapshotId ?? null, decision.policyRuleId ?? null, decision.candidateCount ?? 0, decision.reasonCode ?? null, decision.injectionPayloadHash ?? null, outcome, reward, ts, null);
        return { ...decision, id, outcome, reward, createdAt: ts };
    }
    getRouteDecision(id) {
        const row = this.db.prepare('SELECT * FROM route_decisions WHERE id = ?').get(id);
        return row ? rowToRouteDecision(row) : null;
    }
    resolveRouteDecision(id, outcome, reward) {
        this.db.prepare('UPDATE route_decisions SET outcome = ?, reward = ?, resolved_at = ? WHERE id = ?')
            .run(outcome, reward, now(), id);
    }
    getRecentRouteDecisions(agentId, limit = 20) {
        return this.db.prepare('SELECT * FROM route_decisions WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, limit).map(rowToRouteDecision);
    }
    getUnresolvedRouteDecisions(agentId) {
        return this.db.prepare("SELECT * FROM route_decisions WHERE agent_id = ? AND outcome = 'pending' ORDER BY created_at DESC LIMIT 50").all(agentId).map(rowToRouteDecision);
    }
    getResolvedRouteDecisions(agentId, limit = 100) {
        return this.db.prepare("SELECT * FROM route_decisions WHERE agent_id = ? AND outcome != 'pending' ORDER BY resolved_at DESC, created_at DESC LIMIT ?").all(agentId, limit).map(rowToRouteDecision);
    }
    countRouteDecisions(agentId) {
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM route_decisions WHERE agent_id = ?').get(agentId);
        return row?.cnt ?? 0;
    }
    countRouteDecisionsByLatencyTier(agentId, latencyTier) {
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM route_decisions WHERE agent_id = ? AND latency_tier = ?').get(agentId, latencyTier);
        return row?.cnt ?? 0;
    }
    countSyncPlannerCalls(agentId) {
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM route_decisions WHERE agent_id = ? AND sync_llm_used = 1').get(agentId);
        return row?.cnt ?? 0;
    }
    averageSyncPlannerLatency(agentId) {
        const row = this.db.prepare('SELECT AVG(sync_latency_ms) as avg_ms FROM route_decisions WHERE agent_id = ? AND sync_llm_used = 1 AND sync_latency_ms IS NOT NULL').get(agentId);
        return row?.avg_ms ? Number(row.avg_ms) : 0;
    }
    countSyncPlannerFallbacks(agentId) {
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM route_decisions WHERE agent_id = ? AND sync_llm_used = 1 AND fallback_used = 1').get(agentId);
        return row?.cnt ?? 0;
    }
    countRouteExamples(agentId, polarity = 'all') {
        const row = polarity === 'positive'
            ? this.db.prepare('SELECT COUNT(*) as cnt FROM route_examples WHERE agent_id = ? AND reward > 0').get(agentId)
            : polarity === 'negative'
                ? this.db.prepare('SELECT COUNT(*) as cnt FROM route_examples WHERE agent_id = ? AND reward < 0').get(agentId)
                : this.db.prepare('SELECT COUNT(*) as cnt FROM route_examples WHERE agent_id = ?').get(agentId);
        return row?.cnt ?? 0;
    }
    // ── Route examples and policy snapshots ──────────────────────────────────────
    insertRouteExample(example) {
        const id = example.id || uuid();
        const ts = now();
        this.db.prepare(`
      INSERT INTO route_examples (id, agent_id, turn_frame_json, route_decision_json, outcome, reward, lesson, tags_json, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, example.agentId, JSON.stringify(example.turnFrame), JSON.stringify(example.routeDecision), example.outcome, example.reward, example.lesson, JSON.stringify(example.tags), ts);
        return { ...example, id, createdAt: ts };
    }
    getRouteExamples(agentId, limit = 50) {
        return this.db.prepare('SELECT * FROM route_examples WHERE agent_id = ? ORDER BY created_at DESC, reward DESC LIMIT ?').all(agentId, limit).map((r) => ({
            ...r,
            turnFrame: JSON.parse(r.turn_frame_json),
            routeDecision: JSON.parse(r.route_decision_json),
            tags: JSON.parse(r.tags_json),
        }));
    }
    hasRouteExampleForDecision(agentId, routeDecisionId) {
        const row = this.db.prepare('SELECT id FROM route_examples WHERE agent_id = ? AND tags_json LIKE ? LIMIT 1').get(agentId, `%route_decision:${routeDecisionId}%`);
        return Boolean(row?.id);
    }
    getActivePolicySnapshot(agentId) {
        const row = this.db.prepare('SELECT * FROM route_policy_snapshots WHERE agent_id = ? AND active = 1 ORDER BY created_at DESC LIMIT 1').get(agentId);
        return row ? { ...row, examples: JSON.parse(row.examples_json) } : null;
    }
    insertPolicySnapshot(snapshot) {
        const id = snapshot.id || uuid();
        const ts = now();
        if (snapshot.active) {
            this.db.prepare('UPDATE route_policy_snapshots SET active = 0 WHERE agent_id = ?').run(snapshot.agentId);
        }
        this.db.prepare(`
      INSERT INTO route_policy_snapshots (id, agent_id, policy_text, examples_json, model, prompt_version, created_at, active)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, snapshot.agentId, snapshot.policyText, JSON.stringify(snapshot.examples), snapshot.model ?? null, snapshot.promptVersion ?? null, ts, snapshot.active ? 1 : 0);
        return { ...snapshot, id, createdAt: ts };
    }
    listPolicySnapshots(agentId, limit = 20) {
        return this.db.prepare('SELECT * FROM route_policy_snapshots WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, limit).map((row) => ({
            id: row.id,
            agentId: row.agent_id,
            policyText: row.policy_text,
            examples: JSON.parse(row.examples_json ?? '[]'),
            model: row.model,
            promptVersion: row.prompt_version,
            createdAt: row.created_at,
            active: row.active === 1,
        }));
    }
    // ── Distillation run audit ──────────────────────────────────────────────────
    insertDistillationRun(run) {
        const id = run.id || uuid();
        const ts = now();
        this.db.prepare(`
      INSERT INTO distillation_runs (
        id, agent_id, session_id, turn_id, run_id, phase, model, prompt_version,
        input_hash, redacted_input_summary, output_json, validation_status,
        validation_error, latency_ms, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, run.agentId, run.sessionId ?? null, run.turnId ?? null, run.runId ?? null, run.phase, run.model, run.promptVersion, run.inputHash, run.redactedInputSummary ?? null, run.outputJson, run.validationStatus, run.validationError ?? null, run.latencyMs ?? null, ts);
        return { ...run, id, createdAt: ts };
    }
    // ── Capture audit ──────────────────────────────────────────────────────────
    insertCaptureAudit(row) {
        const id = row.id || uuid();
        const ts = row.createdAt || now();
        this.db.prepare(`
      INSERT INTO capture_audit (
        id, agent_id, turn_id, session_id, run_id, created_at,
        retrieval_intent_json, capture_intent_json,
        capture_job_created, distiller_ran, distiller_model, distiller_latency_ms,
        fallback_ran, candidate_count, stored_count, rejected_count,
        rejection_reasons_json, safe_candidate_preview, evidence_hash
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, row.agentId, row.turnId ?? null, row.sessionId ?? null, row.runId ?? null, ts, JSON.stringify(row.retrievalIntent ?? null), JSON.stringify(row.captureIntent ?? null), row.captureJobCreated ? 1 : 0, row.distillerRan ? 1 : 0, row.distillerModel ?? null, row.distillerLatencyMs ?? null, row.fallbackRan ? 1 : 0, row.candidateCount, row.storedCount, row.rejectedCount, JSON.stringify(row.rejectionReasons ?? []), row.safeCandidatePreview ?? null, row.evidenceHash ?? null);
        return { ...row, id, createdAt: ts };
    }
    listCaptureAudit(agentId, limit = 20) {
        return this.db.prepare('SELECT * FROM capture_audit WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, Math.min(200, Math.max(1, limit))).map(rowToCaptureAudit);
    }
    countCaptureAudit(agentId) {
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM capture_audit WHERE agent_id = ?').get(agentId);
        return row?.cnt ?? 0;
    }
    // ── Job queue ───────────────────────────────────────────────────────────────
    enqueueJob(job) {
        if (job.agentId !== this.ownerAgentId)
            throw new Error(`job_agent_mismatch:${job.agentId}`);
        const id = job.id || uuid();
        const ts = now();
        const dedupeKey = jobDedupeKey(job.agentId, job.kind, job.payload);
        this.db.prepare(`
      INSERT OR IGNORE INTO background_jobs (id, agent_id, kind, status, priority, payload_json, attempts, max_attempts, available_at, started_at, finished_at, error, created_at, updated_at, dedupe_key)
      VALUES (?, ?, ?, 'pending', ?, ?, 0, ?, ?, NULL, NULL, NULL, ?, ?, ?)
    `).run(id, job.agentId, job.kind, job.priority, JSON.stringify(job.payload), job.maxAttempts, job.availableAt, ts, ts, dedupeKey);
        const existing = dedupeKey
            ? this.db.prepare("SELECT * FROM background_jobs WHERE agent_id = ? AND kind = ? AND dedupe_key = ? AND status IN ('pending', 'running') ORDER BY created_at ASC LIMIT 1").get(job.agentId, job.kind, dedupeKey)
            : this.db.prepare('SELECT * FROM background_jobs WHERE id = ?').get(id);
        return existing ? rowToJob(existing) : { ...job, id, status: 'pending', attempts: 0, createdAt: ts, updatedAt: ts };
    }
    claimNextJob(kind, agentId) {
        const ts = now();
        return this.transaction(() => {
            const filters = ["status = 'pending'", 'available_at <= ?'];
            const params = [ts];
            if (kind) {
                filters.push('kind = ?');
                params.push(kind);
            }
            if (agentId) {
                filters.push('agent_id = ?');
                params.push(agentId);
            }
            const job = this.db.prepare(`SELECT * FROM background_jobs WHERE ${filters.join(' AND ')} ORDER BY priority DESC, created_at ASC LIMIT 1`).get(...params);
            if (!job)
                return null;
            const info = this.db.prepare("UPDATE background_jobs SET status = 'running', started_at = ?, attempts = attempts + 1, updated_at = ? WHERE id = ? AND status = 'pending'")
                .run(ts, ts, job.id);
            if (Number(info?.changes || 0) !== 1)
                return null;
            return rowToJob(this.db.prepare('SELECT * FROM background_jobs WHERE id = ?').get(job.id));
        });
    }
    completeJob(id) {
        const ts = now();
        this.db.prepare("UPDATE background_jobs SET status = 'completed', finished_at = ?, updated_at = ? WHERE id = ?")
            .run(ts, ts, id);
    }
    failJob(id, error, retryAfterMs) {
        const job = this.db.prepare('SELECT * FROM background_jobs WHERE id = ?').get(id);
        if (!job)
            return;
        const ts = now();
        const attempts = (job.attempts ?? 0);
        const maxAttempts = (job.max_attempts ?? 3);
        if (attempts >= maxAttempts) {
            this.db.prepare("UPDATE background_jobs SET status = 'dead', error = ?, finished_at = ?, updated_at = ? WHERE id = ?")
                .run(error, ts, ts, id);
        }
        else {
            const availableAt = retryAfterMs ? new Date(Date.now() + retryAfterMs).toISOString() : ts;
            this.db.prepare("UPDATE background_jobs SET status = 'pending', error = ?, available_at = ?, updated_at = ? WHERE id = ?")
                .run(error, availableAt, ts, id);
        }
    }
    getJobQueueDepth(agentId) {
        if (agentId) {
            const row = this.db.prepare("SELECT COUNT(*) as cnt FROM background_jobs WHERE agent_id = ? AND status = 'pending'").get(agentId);
            return row?.cnt ?? 0;
        }
        const row = this.db.prepare("SELECT COUNT(*) as cnt FROM background_jobs WHERE status = 'pending'").get();
        return row?.cnt ?? 0;
    }
    adjustMemoryScore(memoryId, patch) {
        const existing = this.getMemory(memoryId);
        if (!existing)
            return null;
        const updated = this.updateMemory(memoryId, {
            importance: clamp01(existing.importance + (patch.importanceDelta ?? 0)),
            confidence: clamp01(existing.confidence + (patch.confidenceDelta ?? 0)),
            freshness: clamp01(existing.freshness + (patch.freshnessDelta ?? 0)),
            useCount: Math.max(0, existing.useCount + (patch.useCountDelta ?? 0)),
            usefulCount: Math.max(0, existing.usefulCount + (patch.usefulCountDelta ?? 0)),
            captureCount: Math.max(0, existing.captureCount + (patch.captureCountDelta ?? 0)),
            lastUsedAt: now(),
        });
        return updated;
    }
    pruneMemories(agentId, maxNodes) {
        const count = this.countMemories(agentId);
        if (count <= maxNodes)
            return 0;
        const overflow = count - maxNodes;
        const victims = this.db.prepare(`
      SELECT id FROM memory_nodes
      WHERE agent_id = ? AND deleted_at IS NULL
      ORDER BY importance ASC, confidence ASC, updated_at ASC
      LIMIT ?
    `).all(agentId, overflow);
        for (const victim of victims)
            this.softDeleteMemory(victim.id);
        return victims.length;
    }
    consolidateMemories(agentId, limit = 500) {
        const memories = this.listMemories(agentId, { limit });
        const groups = new Map();
        for (const memory of memories) {
            if (memory.deletedAt || memory.supersededBy)
                continue;
            const key = consolidationGroupKey(memory);
            const group = groups.get(key) ?? [];
            group.push(memory);
            groups.set(key, group);
        }
        let consolidated = 0;
        this.transaction(() => {
            for (const group of groups.values()) {
                if (group.length < 2)
                    continue;
                const [keeper, ...duplicates] = group.sort((a, b) => {
                    const scoreA = a.confidence + a.importance + a.captureCount * 0.05;
                    const scoreB = b.confidence + b.importance + b.captureCount * 0.05;
                    return scoreB - scoreA;
                });
                const mergedTags = [...new Set(group.flatMap((memory) => memory.tags || []))];
                const captureCount = group.reduce((sum, memory) => sum + (memory.captureCount || 0), 0);
                const usefulCount = group.reduce((sum, memory) => sum + (memory.usefulCount || 0), 0);
                const useCount = group.reduce((sum, memory) => sum + (memory.useCount || 0), 0);
                this.updateMemory(keeper.id, {
                    tags: mergedTags,
                    captureCount,
                    usefulCount,
                    useCount,
                    confidence: Math.min(1, Math.max(...group.map((memory) => memory.confidence)) + duplicates.length * 0.02),
                    importance: Math.min(1, Math.max(...group.map((memory) => memory.importance)) + duplicates.length * 0.01),
                    freshness: Math.max(...group.map((memory) => memory.freshness)),
                    lastSeenAt: group.map((memory) => memory.lastSeenAt).sort().at(-1) || keeper.lastSeenAt,
                });
                for (const duplicate of duplicates) {
                    this.supersedeMemory(duplicate.id, keeper.id);
                    this.insertEdge({
                        agentId,
                        fromId: duplicate.id,
                        toId: keeper.id,
                        relation: 'supersedes',
                        weight: 1,
                        evidenceCount: duplicate.captureCount || 1,
                    });
                    consolidated += 1;
                }
            }
        });
        return consolidated;
    }
    decayFreshness(agentId, decayPerDay = 0.01) {
        const cutoff = new Date(Date.now() - 7 * 24 * 3600 * 1000).toISOString();
        this.db.prepare('UPDATE memory_nodes SET freshness = MAX(0, freshness - ?), updated_at = ? WHERE agent_id = ? AND deleted_at IS NULL AND last_seen_at < ?').run(decayPerDay * 7, now(), agentId, cutoff);
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM memory_nodes WHERE agent_id = ? AND deleted_at IS NULL AND freshness < 0.1').get(agentId);
        return row?.cnt ?? 0;
    }
    getRouteExamplesByPolarity(agentId, polarity, limit = 50) {
        if (polarity === 'positive') {
            return this.db.prepare('SELECT * FROM route_examples WHERE agent_id = ? AND reward > 0 ORDER BY reward DESC, created_at DESC LIMIT ?').all(agentId, limit);
        }
        return this.db.prepare('SELECT * FROM route_examples WHERE agent_id = ? AND reward < 0 ORDER BY reward ASC, created_at DESC LIMIT ?').all(agentId, limit);
    }
    getConnectedMemories(memoryId, maxDepth = 1, agentId, scopeContext) {
        const seen = new Set([memoryId]);
        let frontier = [memoryId];
        for (let depth = 0; depth < Math.max(1, maxDepth); depth += 1) {
            const next = new Set();
            for (const current of frontier) {
                const edges = agentId
                    ? this.db.prepare('SELECT from_id, to_id FROM memory_edges WHERE agent_id = ? AND (from_id = ? OR to_id = ?)').all(agentId, current, current)
                    : this.db.prepare('SELECT from_id, to_id FROM memory_edges WHERE from_id = ? OR to_id = ?').all(current, current);
                for (const edge of edges) {
                    const neighbor = edge.from_id === current ? edge.to_id : edge.from_id;
                    if (!neighbor || seen.has(neighbor))
                        continue;
                    seen.add(neighbor);
                    next.add(neighbor);
                }
            }
            if (next.size === 0)
                break;
            frontier = [...next];
        }
        const nodeIds = [...seen].filter((id) => id !== memoryId);
        if (nodeIds.length === 0)
            return [];
        const placeholders = nodeIds.map(() => '?').join(',');
        const rows = agentId
            ? this.db.prepare(`SELECT * FROM memory_nodes WHERE id IN (${placeholders}) AND agent_id = ? AND deleted_at IS NULL AND superseded_by IS NULL ORDER BY importance DESC`).all(...nodeIds, agentId)
            : this.db.prepare(`SELECT * FROM memory_nodes WHERE id IN (${placeholders}) AND deleted_at IS NULL AND superseded_by IS NULL ORDER BY importance DESC`).all(...nodeIds);
        return filterMemoriesForScope(rows.map(rowToMemory), scopeContext);
    }
    countEdgesForAgent(agentId) {
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM memory_edges WHERE agent_id = ?').get(agentId);
        return row?.cnt ?? 0;
    }
    // ── Route teacher, counterfactuals, and policy v2 ─────────────────────────
    insertRouteGraphSnapshot(snapshot) {
        const id = snapshot.id || uuid();
        const ts = snapshot.createdAt || now();
        this.db.prepare(`
      INSERT OR REPLACE INTO route_graph_snapshots (
        id, agent_id, route_decision_id, query_set_json, candidate_memory_ids_json,
        candidate_summaries_json, graph_stats_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, snapshot.agentId, snapshot.routeDecisionId, JSON.stringify(snapshot.querySet ?? []), JSON.stringify(snapshot.candidateMemoryIds ?? []), JSON.stringify(snapshot.candidateSummaries ?? []), JSON.stringify(snapshot.graphStats ?? {}), ts);
        return { ...snapshot, id, createdAt: ts };
    }
    getRouteGraphSnapshot(routeDecisionId) {
        const row = this.db.prepare('SELECT * FROM route_graph_snapshots WHERE route_decision_id = ? LIMIT 1').get(routeDecisionId);
        return row ? rowToRouteGraphSnapshot(row) : null;
    }
    insertRouteTeacherRun(run) {
        const id = run.id || uuid();
        const ts = run.createdAt || now();
        this.db.prepare(`
      INSERT INTO route_teacher_runs (
        id, agent_id, route_decision_id, model, prompt_version, input_hash, output_hash,
        verdict, teacher_route, teacher_memory_ids_json, teacher_queries_json,
        teacher_graph_depth, sync_planner_worth_it, confidence, rationale, validated,
        rejection_reason, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, run.agentId, run.routeDecisionId, run.model, run.promptVersion, run.inputHash, run.outputHash, run.verdict, run.teacherRoute, JSON.stringify(run.teacherMemoryIds ?? []), JSON.stringify(run.teacherQueries ?? []), run.teacherGraphDepth, run.syncPlannerWorthIt ? 1 : 0, run.confidence, run.rationale, run.validated ? 1 : 0, run.rejectionReason ?? null, ts);
        return { ...run, id, createdAt: ts };
    }
    hasRouteTeacherRunForDecision(routeDecisionId) {
        const row = this.db.prepare('SELECT id FROM route_teacher_runs WHERE route_decision_id = ? LIMIT 1').get(routeDecisionId);
        return Boolean(row?.id);
    }
    listRouteTeacherRuns(agentId, limit = 20) {
        return this.db.prepare('SELECT * FROM route_teacher_runs WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, Math.min(200, Math.max(1, limit))).map(rowToRouteTeacherRun);
    }
    insertRouteCounterfactual(counterfactual) {
        const id = counterfactual.id || uuid();
        const ts = counterfactual.createdAt || now();
        this.db.prepare(`
      INSERT INTO route_counterfactuals (
        id, agent_id, route_teacher_run_id, route_decision_id, kind, memory_ids_json,
        memory_types_json, graph_depth, estimated_outcome, confidence, rationale, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, counterfactual.agentId, counterfactual.routeTeacherRunId, counterfactual.routeDecisionId, counterfactual.kind, JSON.stringify(counterfactual.memoryIds ?? []), JSON.stringify(counterfactual.memoryTypes ?? []), counterfactual.graphDepth, counterfactual.estimatedOutcome, counterfactual.confidence, counterfactual.rationale, ts);
        return { ...counterfactual, id, createdAt: ts };
    }
    listRouteCounterfactuals(agentId, routeDecisionId, limit = 50) {
        const safeLimit = Math.min(200, Math.max(1, limit));
        const rows = routeDecisionId
            ? this.db.prepare('SELECT * FROM route_counterfactuals WHERE agent_id = ? AND route_decision_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, routeDecisionId, safeLimit)
            : this.db.prepare('SELECT * FROM route_counterfactuals WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, safeLimit);
        return rows.map(rowToRouteCounterfactual);
    }
    insertRouteTrainingExampleV2(example) {
        const existing = this.db.prepare(`
      SELECT * FROM route_training_examples_v2
      WHERE agent_id = ? AND example_kind = ? AND task_type = ? AND route = ?
        AND memory_types_json = ? AND query_templates_json = ? AND graph_depth = ?
      LIMIT 1
    `).get(example.agentId, example.exampleKind, example.taskType, example.route, JSON.stringify(example.memoryTypes ?? []), JSON.stringify(example.queryTemplates ?? []), example.graphDepth);
        if (existing) {
            this.db.prepare(`
        UPDATE route_training_examples_v2
        SET support_count = support_count + ?, harm_count = harm_count + ?, confidence = MAX(confidence, ?), evidence_ids_json = ?, created_at = ?
        WHERE id = ?
      `).run(example.supportCount ?? 1, example.harmCount ?? 0, example.confidence, JSON.stringify([...new Set([...JSON.parse(existing.evidence_ids_json ?? '[]'), ...(example.evidenceIds ?? [])])]), now(), existing.id);
            return rowToRouteTrainingExampleV2(this.db.prepare('SELECT * FROM route_training_examples_v2 WHERE id = ?').get(existing.id));
        }
        const id = example.id || uuid();
        const ts = example.createdAt || now();
        this.db.prepare(`
      INSERT INTO route_training_examples_v2 (
        id, agent_id, route_decision_id, route_teacher_run_id, example_kind, task_type,
        turn_signals_json, route, memory_types_json, query_templates_json, graph_depth,
        confidence, support_count, harm_count, source, evidence_ids_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, example.agentId, example.routeDecisionId, example.routeTeacherRunId ?? null, example.exampleKind, example.taskType, JSON.stringify(example.turnSignals ?? []), example.route, JSON.stringify(example.memoryTypes ?? []), JSON.stringify(example.queryTemplates ?? []), example.graphDepth, example.confidence, example.supportCount ?? 1, example.harmCount ?? 0, example.source, JSON.stringify(example.evidenceIds ?? []), ts);
        return { ...example, id, createdAt: ts };
    }
    listRouteTrainingExamplesV2(agentId, limit = 100) {
        return this.db.prepare('SELECT * FROM route_training_examples_v2 WHERE agent_id = ? ORDER BY confidence DESC, support_count DESC, created_at DESC LIMIT ?').all(agentId, Math.min(500, Math.max(1, limit))).map(rowToRouteTrainingExampleV2);
    }
    countRouteTrainingExamplesV2(agentId) {
        const row = this.db.prepare('SELECT COUNT(*) as cnt FROM route_training_examples_v2 WHERE agent_id = ?').get(agentId);
        return row?.cnt ?? 0;
    }
    insertPolicySnapshotV2(snapshot) {
        const id = snapshot.id || uuid();
        const ts = snapshot.createdAt || now();
        if (snapshot.status === 'active') {
            this.db.prepare("UPDATE route_policy_snapshots_v2 SET status = 'shadow' WHERE agent_id = ? AND status = 'active'").run(snapshot.agentId);
        }
        this.db.prepare(`
      INSERT INTO route_policy_snapshots_v2 (
        id, agent_id, version, status, rules_json, global_budgets_json, eval_summary_json,
        example_ids_json, model, prompt_version, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, snapshot.agentId, snapshot.version, snapshot.status, JSON.stringify(snapshot.rules ?? []), JSON.stringify(snapshot.globalBudgets ?? {}), snapshot.evalSummary ? JSON.stringify(snapshot.evalSummary) : null, JSON.stringify(snapshot.exampleIds ?? []), snapshot.model ?? null, snapshot.promptVersion ?? null, ts);
        return { ...snapshot, id, createdAt: ts };
    }
    getActivePolicySnapshotV2(agentId) {
        const row = this.db.prepare("SELECT * FROM route_policy_snapshots_v2 WHERE agent_id = ? AND status = 'active' ORDER BY created_at DESC LIMIT 1").get(agentId);
        return row ? rowToRoutePolicySnapshotV2(row) : null;
    }
    listPolicySnapshotsV2(agentId, limit = 20) {
        return this.db.prepare('SELECT * FROM route_policy_snapshots_v2 WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, Math.min(200, Math.max(1, limit))).map(rowToRoutePolicySnapshotV2);
    }
    // ── Proof events ────────────────────────────────────────────────────────────
    insertProofEvent(event) {
        const id = event.id || uuid();
        const ts = now();
        this.db.prepare(`
      INSERT INTO proof_events (
        id, agent_id, kind, created_at, source_hook, turn_id, session_id, run_id,
        memory_id, injection_id, route_decision_id, distillation_run_id,
        raw_transcript_stored, payload_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, event.agentId, event.kind, ts, event.sourceHook ?? null, event.turnId ?? null, event.sessionId ?? null, event.runId ?? null, event.memoryId ?? null, event.injectionId ?? null, event.routeDecisionId ?? null, event.distillationRunId ?? null, event.rawTranscriptStored ? 1 : 0, JSON.stringify(event.payload));
        return { ...event, id, createdAt: ts };
    }
    getProofEvents(agentId, limit = 20) {
        return this.db.prepare('SELECT * FROM proof_events WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, limit).map((r) => ({
            ...r,
            rawTranscriptStored: r.raw_transcript_stored === 1,
            payload: JSON.parse(r.payload_json),
        }));
    }
    pruneProofEvents(agentId, retain) {
        this.db.prepare(`
      DELETE FROM proof_events
      WHERE agent_id = ?
        AND id NOT IN (
          SELECT id FROM proof_events
          WHERE agent_id = ?
          ORDER BY created_at DESC
          LIMIT ?
        )
    `).run(agentId, agentId, retain);
    }
    writeStatusSnapshot(agentId, status) {
        this.db.prepare(`
      INSERT INTO status_snapshots (agent_id, status_json, updated_at)
      VALUES (?, ?, ?)
      ON CONFLICT(agent_id) DO UPDATE SET
        status_json = excluded.status_json,
        updated_at = excluded.updated_at
    `).run(agentId, JSON.stringify(status), now());
        return status;
    }
    readStatusSnapshot(agentId) {
        const row = this.db.prepare('SELECT status_json FROM status_snapshots WHERE agent_id = ?').get(agentId);
        return row ? JSON.parse(row.status_json) : null;
    }
    // ── Transactions ────────────────────────────────────────────────────────────
    transaction(fn) {
        return this.db.transaction(fn)();
    }
}
// ── Row mappers ───────────────────────────────────────────────────────────────
function consolidationGroupKey(memory) {
    const parts = String(memory.normalizedKey || '').split(':').filter(Boolean);
    const broadTypes = new Set(['preference', 'workflow', 'routing_rule', 'agent_assignment', 'tool_convention', 'outcome']);
    const normalizedRoot = broadTypes.has(memory.type) && parts.length >= 3
        ? parts.slice(0, 3).join(':')
        : memory.normalizedKey;
    return `${memory.type}:${memory.scopeKind}:${memory.scopeKey ?? ''}:${normalizedRoot}`;
}
function rowToMemory(r) {
    return {
        id: r.id, agentId: r.agent_id, type: r.type, content: r.content,
        positive: r.positive, negative: r.negative,
        scopeKind: r.scope_kind, scopeKey: r.scope_key,
        normalizedKey: r.normalized_key, tags: JSON.parse(r.tags_json ?? '[]'),
        importance: r.importance, freshness: r.freshness, confidence: r.confidence,
        useCount: r.use_count, usefulCount: r.useful_count, captureCount: r.capture_count,
        distilledByModel: r.distilled_by_model, distillerPromptVersion: r.distiller_prompt_version,
        distillationConfidence: r.distillation_confidence,
        evidenceKind: r.evidence_kind, evidenceHash: r.evidence_hash,
        sourceHook: r.source_hook, sourceTurnId: r.source_turn_id, sourceSessionId: r.source_session_id,
        createdAt: r.created_at, updatedAt: r.updated_at, lastSeenAt: r.last_seen_at,
        lastUsedAt: r.last_used_at, supersededBy: r.superseded_by, deletedAt: r.deleted_at,
    };
}
function rowToInjection(r) {
    return {
        id: r.id, agentId: r.agent_id, memoryId: r.memory_id,
        routeDecisionId: r.route_decision_id, runId: r.run_id,
        turnId: r.turn_id, sessionId: r.session_id,
        query: r.query, rank: r.rank, score: r.score,
        injectedAt: r.injected_at, resolvedAt: r.resolved_at,
        outcome: r.outcome, correctionSignal: r.correction_signal,
    };
}
function rowToEdge(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        fromId: r.from_id,
        toId: r.to_id,
        relation: r.relation,
        weight: r.weight,
        evidenceCount: r.evidence_count,
        createdAt: r.created_at,
        updatedAt: r.updated_at,
    };
}
function rowToRouteFrame(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        sessionKeyHash: r.session_key_hash,
        turnHash: r.turn_hash,
        redactedTurnSummary: r.redacted_turn_summary,
        taskType: r.task_type,
        turnSignals: JSON.parse(r.turn_signals_json ?? '[]'),
        intentSignals: JSON.parse(r.intent_signals_json ?? '[]'),
        safetySignals: JSON.parse(r.safety_signals_json ?? '[]'),
        projectHint: r.project_hint,
        repoHint: r.repo_hint,
        latencyBudgetMs: r.latency_budget_ms,
        createdAt: r.created_at,
    };
}
function rowToRouteDecision(r) {
    return {
        id: r.id, agentId: r.agent_id, routeFrameId: r.route_frame_id, sessionId: r.session_id,
        turnId: r.turn_id, runId: r.run_id,
        route: r.route, confidence: r.confidence, latencyTier: r.latency_tier,
        syncLlmUsed: r.sync_llm_used === 1, syncLatencyMs: r.sync_latency_ms,
        fallbackUsed: r.fallback_used === 1,
        turnFrame: JSON.parse(r.turn_frame_json),
        retrievalPlan: JSON.parse(r.retrieval_plan_json),
        injectionPlan: JSON.parse(r.injection_plan_json),
        selectedMemoryIds: JSON.parse(r.selected_memory_ids_json ?? '[]'),
        omittedMemoryIds: JSON.parse(r.omitted_memory_ids_json ?? '[]'),
        model: r.model, promptVersion: r.prompt_version,
        policySnapshotId: r.policy_snapshot_id,
        policyRuleId: r.policy_rule_id,
        candidateCount: r.candidate_count,
        reasonCode: r.reason_code,
        injectionPayloadHash: r.injection_payload_hash,
        outcome: r.outcome, reward: r.reward,
        createdAt: r.created_at, resolvedAt: r.resolved_at,
    };
}
function rowToJob(r) {
    return {
        id: r.id, agentId: r.agent_id, kind: r.kind, status: r.status,
        priority: r.priority, payload: JSON.parse(r.payload_json ?? '{}'),
        attempts: r.attempts, maxAttempts: r.max_attempts,
        availableAt: r.available_at, startedAt: r.started_at,
        finishedAt: r.finished_at, error: r.error,
        createdAt: r.created_at, updatedAt: r.updated_at,
    };
}
function rowToCaptureAudit(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        turnId: r.turn_id,
        sessionId: r.session_id,
        runId: r.run_id,
        createdAt: r.created_at,
        retrievalIntent: JSON.parse(r.retrieval_intent_json ?? 'null'),
        captureIntent: JSON.parse(r.capture_intent_json ?? 'null'),
        captureJobCreated: r.capture_job_created === 1,
        distillerRan: r.distiller_ran === 1,
        distillerModel: r.distiller_model,
        distillerLatencyMs: r.distiller_latency_ms,
        fallbackRan: r.fallback_ran === 1,
        candidateCount: r.candidate_count,
        storedCount: r.stored_count,
        rejectedCount: r.rejected_count,
        rejectionReasons: JSON.parse(r.rejection_reasons_json ?? '[]'),
        safeCandidatePreview: r.safe_candidate_preview,
        evidenceHash: r.evidence_hash,
    };
}
function rowToRouteGraphSnapshot(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        routeDecisionId: r.route_decision_id,
        querySet: JSON.parse(r.query_set_json ?? '[]'),
        candidateMemoryIds: JSON.parse(r.candidate_memory_ids_json ?? '[]'),
        candidateSummaries: JSON.parse(r.candidate_summaries_json ?? '[]'),
        graphStats: JSON.parse(r.graph_stats_json ?? '{}'),
        createdAt: r.created_at,
    };
}
function rowToRouteTeacherRun(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        routeDecisionId: r.route_decision_id,
        model: r.model,
        promptVersion: r.prompt_version,
        inputHash: r.input_hash,
        outputHash: r.output_hash,
        verdict: r.verdict,
        teacherRoute: r.teacher_route,
        teacherMemoryIds: JSON.parse(r.teacher_memory_ids_json ?? '[]'),
        teacherQueries: JSON.parse(r.teacher_queries_json ?? '[]'),
        teacherGraphDepth: r.teacher_graph_depth,
        syncPlannerWorthIt: r.sync_planner_worth_it === 1,
        confidence: r.confidence,
        rationale: r.rationale,
        validated: r.validated === 1,
        rejectionReason: r.rejection_reason,
        createdAt: r.created_at,
    };
}
function rowToRouteCounterfactual(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        routeTeacherRunId: r.route_teacher_run_id,
        routeDecisionId: r.route_decision_id,
        kind: r.kind,
        memoryIds: JSON.parse(r.memory_ids_json ?? '[]'),
        memoryTypes: JSON.parse(r.memory_types_json ?? '[]'),
        graphDepth: r.graph_depth,
        estimatedOutcome: r.estimated_outcome,
        confidence: r.confidence,
        rationale: r.rationale,
        createdAt: r.created_at,
    };
}
function rowToRouteTrainingExampleV2(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        routeDecisionId: r.route_decision_id,
        routeTeacherRunId: r.route_teacher_run_id,
        exampleKind: r.example_kind,
        taskType: r.task_type,
        turnSignals: JSON.parse(r.turn_signals_json ?? '[]'),
        route: r.route,
        memoryTypes: JSON.parse(r.memory_types_json ?? '[]'),
        queryTemplates: JSON.parse(r.query_templates_json ?? '[]'),
        graphDepth: r.graph_depth,
        confidence: r.confidence,
        supportCount: r.support_count,
        harmCount: r.harm_count,
        source: r.source,
        evidenceIds: JSON.parse(r.evidence_ids_json ?? '[]'),
        createdAt: r.created_at,
    };
}
function rowToRouteFrameV3(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        routeDecisionId: r.route_decision_id,
        routeFrameId: r.route_frame_id,
        redactedTurnSummary: r.redacted_turn_summary,
        taskType: r.task_type,
        turnSignals: JSON.parse(r.turn_signals_json ?? '[]'),
        projectHint: r.project_hint,
        repoHint: r.repo_hint,
        toolHints: JSON.parse(r.tool_hints_json ?? '[]'),
        routeHintFlags: JSON.parse(r.route_hint_flags_json ?? '[]'),
        chosenActionId: r.chosen_action_id,
        chosenRoute: r.chosen_route,
        chosenMemoryTypes: JSON.parse(r.chosen_memory_types_json ?? '[]'),
        chosenGraphDepth: r.chosen_graph_depth,
        chosenSyncPlanner: r.chosen_sync_planner,
        policySnapshotId: r.policy_snapshot_id,
        policyRuleId: r.policy_rule_id,
        outcome: r.outcome,
        reward: r.reward,
        rewardComponents: r.reward_components_json ? JSON.parse(r.reward_components_json) : undefined,
        payloadHash: r.payload_hash,
        createdAt: r.created_at,
    };
}
function rowToRouteActionPrototypeV3(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        route: r.route,
        memoryTypes: JSON.parse(r.memory_types_json ?? '[]'),
        graphDepth: r.graph_depth,
        syncPlanner: r.sync_planner,
        queryTemplateFamily: JSON.parse(r.query_template_family_json ?? '[]'),
        sparseSignature: JSON.parse(r.sparse_signature_json ?? '[]'),
        denseEmbedding: JSON.parse(r.dense_embedding_json ?? '[]'),
        supportPrior: r.support_prior,
        harmPrior: r.harm_prior,
        status: r.status,
        provenance: r.provenance,
        sourceExampleIds: JSON.parse(r.source_example_ids_json ?? '[]'),
        createdAt: r.created_at,
        updatedAt: r.updated_at,
    };
}
function rowToRoutePairExampleV3(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        frameId: r.frame_id,
        positiveActionId: r.positive_action_id,
        negativeActionId: r.negative_action_id,
        labelSource: r.label_source,
        marginWeight: r.margin_weight,
        evidenceIds: JSON.parse(r.evidence_ids_json ?? '[]'),
        createdAt: r.created_at,
    };
}
function rowToRouteBanditFeedbackV3(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        frameId: r.frame_id,
        chosenActionId: r.chosen_action_id,
        reward: r.reward,
        rewardComponents: JSON.parse(r.reward_components_json ?? '{}'),
        cost: r.cost,
        latencyMs: r.latency_ms,
        outcomeLabel: r.outcome_label,
        learningBucket: r.learning_bucket === 1,
        createdAt: r.created_at,
    };
}
function rowToRouteBanditStateV3(r) {
    return {
        agentId: r.agent_id,
        learnerVersion: r.learner_version,
        featureSchemaVersion: r.feature_schema_version,
        explorationAlpha: r.exploration_alpha,
        sharedWeights: JSON.parse(r.shared_weights_json ?? '[]'),
        actionStats: JSON.parse(r.action_stats_json ?? '{}'),
        updatedAt: r.updated_at,
    };
}
function rowToRoutePolicySnapshotV2(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        version: 'route-policy-v2',
        status: r.status,
        rules: JSON.parse(r.rules_json ?? '[]'),
        globalBudgets: JSON.parse(r.global_budgets_json ?? '{}'),
        evalSummary: r.eval_summary_json ? JSON.parse(r.eval_summary_json) : undefined,
        exampleIds: JSON.parse(r.example_ids_json ?? '[]'),
        model: r.model,
        promptVersion: r.prompt_version,
        createdAt: r.created_at,
    };
}
function rowToRoutePolicySnapshotV3(r) {
    return {
        id: r.id,
        agentId: r.agent_id,
        version: 'route-policy-v3',
        status: r.status,
        rules: JSON.parse(r.rules_json ?? '[]'),
        actionPriors: JSON.parse(r.action_priors_json ?? '{}'),
        globalBudgets: JSON.parse(r.global_budgets_json ?? '{}'),
        evalSummary: r.eval_summary_json ? JSON.parse(r.eval_summary_json) : undefined,
        calibration: r.calibration_json ? JSON.parse(r.calibration_json) : undefined,
        lineage: r.lineage_json ? JSON.parse(r.lineage_json) : undefined,
        sourceFrameIds: JSON.parse(r.source_frame_ids_json ?? '[]'),
        sourcePrototypeIds: JSON.parse(r.source_prototype_ids_json ?? '[]'),
        model: r.model,
        promptVersion: r.prompt_version,
        createdAt: r.created_at,
    };
}
// ── DB helpers ────────────────────────────────────────────────────────────────
export function dbPathForAgent(activationRoot, agentId) {
    const substituted = activationRoot.replace('${agentId}', agentId);
    const dir = substituted === '~'
        ? os.homedir()
        : substituted.startsWith('~/')
            ? path.join(os.homedir(), substituted.slice(2))
            : path.resolve(substituted);
    return path.join(dir, 'openclawbrain.db');
}
export function openDb(dbPath) {
    const dir = path.dirname(dbPath);
    mkdirSync(dir, { recursive: true });
    const { db } = openDatabase(dbPath);
    db.pragma('journal_mode = WAL');
    db.pragma('synchronous = NORMAL');
    db.pragma('foreign_keys = ON');
    return db;
}
function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}
function tableHasColumn(db, table, column) {
    return db.prepare(`PRAGMA table_info(${table})`).all().some((row) => row.name === column);
}
function jobDedupeKey(agentId, kind, payload) {
    if (kind !== 'feedback_distillation' && kind !== 'outcome_classification')
        return null;
    const packet = payload?.packet || {};
    const runId = String(packet.runId || '').trim();
    const turnId = String(packet.turnId || '').trim();
    const promptHash = String(packet.metadata?.promptHash || '').trim();
    if (!runId && !turnId && !promptHash)
        return null;
    return [agentId, kind, runId, turnId, promptHash].join(':');
}
