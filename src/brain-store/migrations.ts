/**
 * SQLite schema for the brain's learned retrieval graph.
 *
 * These tables live alongside LCM's existing tables in the same database.
 * All brain tables are prefixed with "brain_" to avoid collisions.
 */

import type { DatabaseSync } from "node:sqlite";

function hasColumn(db: DatabaseSync, table: string, column: string): boolean {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name?: unknown }>;
  return rows.some((row) => row.name === column);
}

function ensureColumn(db: DatabaseSync, table: string, column: string, definition: string): void {
  if (hasColumn(db, table, column)) {
    return;
  }
  db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
}

export function runBrainMigrations(db: DatabaseSync): void {
  db.exec(`
    -- ═══════════════════════════════════════════
    -- Brain Knowledge Graph
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_nodes (
      id            TEXT PRIMARY KEY,
      kind          TEXT NOT NULL,
      content       TEXT NOT NULL,
      embedding     BLOB,
      source_uri    TEXT,
      trust         TEXT NOT NULL DEFAULT 'scanner',
      tags          TEXT NOT NULL DEFAULT '[]',
      token_count   INTEGER NOT NULL DEFAULT 0,
      metadata      TEXT NOT NULL DEFAULT '{}',
      created_at    INTEGER NOT NULL,
      updated_at    INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS brain_edges (
      source        TEXT NOT NULL,
      target        TEXT NOT NULL,
      kind          TEXT NOT NULL,
      weight        REAL NOT NULL DEFAULT 0.5,
      prior         REAL NOT NULL DEFAULT 0.5,
      metadata      TEXT NOT NULL DEFAULT '{}',
      decayed_at    INTEGER NOT NULL,
      created_at    INTEGER NOT NULL,
      PRIMARY KEY (source, target, kind)
    );

    CREATE INDEX IF NOT EXISTS brain_edges_source_idx ON brain_edges(source);
    CREATE INDEX IF NOT EXISTS brain_edges_target_idx ON brain_edges(target);
    CREATE INDEX IF NOT EXISTS brain_nodes_kind_idx ON brain_nodes(kind);

    CREATE TABLE IF NOT EXISTS brain_seed_weights (
      node_id       TEXT PRIMARY KEY,
      weight        REAL NOT NULL DEFAULT 0.0,
      updated_at    INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS brain_stop_local_weights (
      source_node_id TEXT PRIMARY KEY,
      weight         REAL NOT NULL DEFAULT 0.0,
      updated_at     INTEGER NOT NULL
    );

    -- ═══════════════════════════════════════════
    -- Episodes (full traversal records)
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_episodes (
      id                TEXT PRIMARY KEY,
      conversation_id   INTEGER,
      query_text        TEXT,
      query_embedding   BLOB,
      trajectory        TEXT NOT NULL,
      fired_nodes       TEXT NOT NULL,
      vetoed_nodes      TEXT NOT NULL DEFAULT '[]',
      context_chars     INTEGER NOT NULL DEFAULT 0,
      reward            REAL,
      reward_source     TEXT,
      pack_version      INTEGER,
      updated           INTEGER NOT NULL DEFAULT 0,
      created_at        INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_episodes_created_idx ON brain_episodes(created_at);

    -- ═══════════════════════════════════════════
    -- Labels (pending reward signals)
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_labels (
      id            TEXT PRIMARY KEY,
      episode_id    TEXT NOT NULL,
      source        TEXT NOT NULL,
      value         REAL NOT NULL,
      confidence    REAL NOT NULL DEFAULT 1.0,
      reason        TEXT,
      applied       INTEGER NOT NULL DEFAULT 0,
      created_at    INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_labels_episode_idx ON brain_labels(episode_id);
    CREATE INDEX IF NOT EXISTS brain_labels_applied_idx ON brain_labels(applied);

    -- ═══════════════════════════════════════════
    -- Raw Evidence + Resolved Label Decisions
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_evidence (
      id              TEXT PRIMARY KEY,
      episode_id      TEXT NOT NULL,
      conversation_id INTEGER,
      source          TEXT NOT NULL,
      kind            TEXT NOT NULL,
      value           REAL NOT NULL,
      confidence      REAL NOT NULL DEFAULT 1.0,
      reason          TEXT,
      content_snippet TEXT,
      metadata        TEXT NOT NULL DEFAULT '{}',
      resolved        INTEGER NOT NULL DEFAULT 0,
      created_at      INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_evidence_episode_idx ON brain_evidence(episode_id);
    CREATE INDEX IF NOT EXISTS brain_evidence_resolved_idx ON brain_evidence(resolved, created_at);

    CREATE TABLE IF NOT EXISTS brain_resolved_labels (
      id            TEXT PRIMARY KEY,
      evidence_id   TEXT NOT NULL,
      episode_id    TEXT NOT NULL,
      source        TEXT NOT NULL,
      value         REAL NOT NULL,
      confidence    REAL NOT NULL DEFAULT 1.0,
      resolution    TEXT NOT NULL,
      label_id      TEXT,
      note          TEXT,
      created_at    INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_resolved_labels_episode_idx ON brain_resolved_labels(episode_id, created_at);
    CREATE INDEX IF NOT EXISTS brain_resolved_labels_evidence_idx ON brain_resolved_labels(evidence_id);

    CREATE TABLE IF NOT EXISTS brain_trace_supervision (
      id              TEXT PRIMARY KEY,
      trace_id        TEXT NOT NULL,
      episode_id      TEXT NOT NULL,
      conversation_id INTEGER,
      source          TEXT NOT NULL,
      kind            TEXT NOT NULL,
      value           REAL NOT NULL,
      confidence      REAL NOT NULL DEFAULT 1.0,
      reason          TEXT,
      content_snippet TEXT,
      resolution      TEXT NOT NULL,
      label_id        TEXT,
      evidence_id     TEXT,
      metadata        TEXT NOT NULL DEFAULT '{}',
      created_at      INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_trace_supervision_trace_idx ON brain_trace_supervision(trace_id, created_at);
    CREATE INDEX IF NOT EXISTS brain_trace_supervision_episode_idx ON brain_trace_supervision(episode_id, created_at);

    -- ═══════════════════════════════════════════
    -- Durable Turn Observations
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_observations (
      id                    TEXT PRIMARY KEY,
      episode_id            TEXT NOT NULL UNIQUE,
      conversation_id       INTEGER,
      trace_id              TEXT,
      binding_mode          TEXT,
      serve_decision_record_id TEXT,
      selection_digest      TEXT,
      turn_compile_event_id TEXT,
      decision_recorded_at  TEXT,
      active_pack_id        TEXT,
      active_pack_event_export_digest TEXT,
      active_pack_graph_checksum TEXT,
      active_pack_router_checksum TEXT,
      active_pack_built_at  TEXT,
      query_text            TEXT NOT NULL,
      retrieved_context_json TEXT NOT NULL DEFAULT '[]',
      route_metadata_json   TEXT NOT NULL DEFAULT '{}',
      assistant_response    TEXT NOT NULL DEFAULT '',
      tool_results_json     TEXT NOT NULL DEFAULT '[]',
      follow_up_text        TEXT,
      phase1_score          REAL,
      phase2_score          REAL,
      final_score           REAL,
      confidence            REAL,
      reason                TEXT,
      status                TEXT NOT NULL DEFAULT 'pending_followup',
      teacher_evaluation_json TEXT,
      created_at            INTEGER NOT NULL,
      updated_at            INTEGER NOT NULL,
      evaluated_at          INTEGER
    );

    CREATE INDEX IF NOT EXISTS brain_observations_status_idx ON brain_observations(status, created_at);
    CREATE INDEX IF NOT EXISTS brain_observations_conversation_idx ON brain_observations(conversation_id, created_at);
    CREATE INDEX IF NOT EXISTS brain_observations_trace_idx ON brain_observations(trace_id, created_at);

    -- ═══════════════════════════════════════════
    -- Shadow Usefulness Evaluations
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_context_usefulness (
      id              TEXT PRIMARY KEY,
      observation_id  TEXT NOT NULL UNIQUE,
      episode_id      TEXT NOT NULL,
      trace_id        TEXT,
      conversation_id INTEGER,
      binding_mode    TEXT,
      follow_up_text  TEXT,
      tool_results_json TEXT NOT NULL DEFAULT '[]',
      signal_json     TEXT NOT NULL DEFAULT '{}',
      final_score     REAL NOT NULL,
      confidence      REAL NOT NULL,
      verdict         TEXT NOT NULL,
      reason          TEXT,
      created_at      INTEGER NOT NULL,
      updated_at      INTEGER NOT NULL,
      evaluated_at    INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_context_usefulness_observation_idx ON brain_context_usefulness(observation_id);
    CREATE INDEX IF NOT EXISTS brain_context_usefulness_trace_idx ON brain_context_usefulness(trace_id, evaluated_at);
    CREATE INDEX IF NOT EXISTS brain_context_usefulness_episode_idx ON brain_context_usefulness(episode_id, evaluated_at);
    CREATE INDEX IF NOT EXISTS brain_context_usefulness_created_idx ON brain_context_usefulness(created_at DESC);

    -- ═══════════════════════════════════════════
    -- Packs (immutable serving snapshots)
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_packs (
      version       INTEGER PRIMARY KEY AUTOINCREMENT,
      node_count    INTEGER NOT NULL,
      edge_count    INTEGER NOT NULL,
      health_json   TEXT NOT NULL,
      promoted_at   INTEGER,
      rolled_back   INTEGER NOT NULL DEFAULT 0,
      created_at    INTEGER NOT NULL
    );

    -- ═══════════════════════════════════════════
    -- Mutation Proposals
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_mutations (
      id            TEXT PRIMARY KEY,
      kind          TEXT NOT NULL,
      proposal      TEXT NOT NULL,
      evidence      TEXT,
      expected_gain REAL,
      status        TEXT NOT NULL DEFAULT 'pending',
      created_at    INTEGER NOT NULL,
      resolved_at   INTEGER
    );

    -- ═══════════════════════════════════════════
    -- Mutation Bundles
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_mutation_bundles (
      id              TEXT PRIMARY KEY,
      mutation_ids    TEXT NOT NULL,  -- JSON array of mutation proposal IDs
      bundle_size     INTEGER NOT NULL,
      status          TEXT NOT NULL DEFAULT 'pending',  -- pending, evaluating, promoted, rejected
      base_score      REAL,           -- graph score before mutations
      candidate_score REAL,           -- graph score after mutations
      expected_gain   REAL,
      rejection_reason TEXT,
      created_at      INTEGER NOT NULL,
      resolved_at     INTEGER
    );

    CREATE INDEX IF NOT EXISTS brain_mutation_bundles_status_idx ON brain_mutation_bundles(status, created_at);

    -- ═══════════════════════════════════════════
    -- Learning Journal
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_learning_journal (
      id              TEXT PRIMARY KEY,
      event_type      TEXT NOT NULL,
      mutation_id     TEXT,
      mutation_ids    TEXT NOT NULL DEFAULT '[]',
      bundle_id       TEXT,
      pack_version    INTEGER,
      payload         TEXT NOT NULL DEFAULT '{}',
      created_at      INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_learning_journal_created_idx ON brain_learning_journal(created_at);
    CREATE INDEX IF NOT EXISTS brain_learning_journal_event_idx ON brain_learning_journal(event_type, created_at);
    CREATE INDEX IF NOT EXISTS brain_learning_journal_bundle_idx ON brain_learning_journal(bundle_id, created_at);

    -- ═══════════════════════════════════════════
    -- Decision Traces
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_traces (
      id              TEXT PRIMARY KEY,
      episode_id      TEXT,
      pack_version    INTEGER,
      query_text      TEXT,
      seed_scores     TEXT NOT NULL,
      trajectory      TEXT NOT NULL,
      fired_nodes     TEXT NOT NULL,
      vetoed_nodes    TEXT NOT NULL DEFAULT '[]',
      context_chars   INTEGER NOT NULL,
      footer          TEXT NOT NULL,
      route_trace_json TEXT NOT NULL DEFAULT 'null',
      created_at      INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS brain_traces_created_idx ON brain_traces(created_at DESC);

    -- ═══════════════════════════════════════════
    -- Training State (key-value)
    -- ═══════════════════════════════════════════

    CREATE TABLE IF NOT EXISTS brain_training_state (
      key           TEXT PRIMARY KEY,
      value         TEXT NOT NULL
    );
  `);

  const mutationBundleColumns = db.prepare(`PRAGMA table_info(brain_mutation_bundles)`).all() as Array<{
    name: string;
  }>;
  if (!mutationBundleColumns.some((column) => column.name === "verdict_json")) {
    db.exec(`ALTER TABLE brain_mutation_bundles ADD COLUMN verdict_json TEXT`);
  }

  const traceColumns = db.prepare(`PRAGMA table_info(brain_traces)`).all() as Array<{
    name: string;
  }>;
  if (!traceColumns.some((column) => column.name === "route_trace_json")) {
    db.exec(`ALTER TABLE brain_traces ADD COLUMN route_trace_json TEXT NOT NULL DEFAULT 'null'`);
  }

  ensureColumn(db, "brain_observations", "binding_mode", "TEXT");
  ensureColumn(db, "brain_observations", "serve_decision_record_id", "TEXT");
  ensureColumn(db, "brain_observations", "selection_digest", "TEXT");
  ensureColumn(db, "brain_observations", "turn_compile_event_id", "TEXT");
  ensureColumn(db, "brain_observations", "decision_recorded_at", "TEXT");
  ensureColumn(db, "brain_observations", "active_pack_id", "TEXT");
  ensureColumn(db, "brain_observations", "active_pack_event_export_digest", "TEXT");
  ensureColumn(db, "brain_observations", "active_pack_graph_checksum", "TEXT");
  ensureColumn(db, "brain_observations", "active_pack_router_checksum", "TEXT");
  ensureColumn(db, "brain_observations", "active_pack_built_at", "TEXT");

  db.exec(`
    CREATE INDEX IF NOT EXISTS brain_observations_binding_mode_idx ON brain_observations(binding_mode);
    CREATE INDEX IF NOT EXISTS brain_observations_decision_record_idx ON brain_observations(serve_decision_record_id);
    CREATE INDEX IF NOT EXISTS brain_observations_selection_digest_idx ON brain_observations(selection_digest);
    CREATE INDEX IF NOT EXISTS brain_observations_turn_compile_event_idx ON brain_observations(turn_compile_event_id);
    CREATE INDEX IF NOT EXISTS brain_observations_pack_digest_idx ON brain_observations(active_pack_graph_checksum, selection_digest);
  `);
}
