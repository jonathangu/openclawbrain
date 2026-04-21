import Database from "better-sqlite3";

import type { AblationResult, Decision, MemoryBackend, Outcome, TurnSlice } from "../types.js";

const SCHEMA = `
CREATE TABLE IF NOT EXISTS decisions (
  decision_id      TEXT PRIMARY KEY,
  run_id           TEXT NOT NULL,
  case_id          TEXT NOT NULL,
  turn_index       INTEGER NOT NULL,
  backend          TEXT NOT NULL,
  slice            TEXT NOT NULL,
  gate_score       REAL,
  gate_threshold   REAL,
  fired            INTEGER NOT NULL,
  retrieved_json   TEXT NOT NULL,
  injected_tokens  INTEGER NOT NULL,
  query_text       TEXT NOT NULL,
  timestamp_ms     INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_decisions_run_case_backend
  ON decisions(run_id, case_id, backend);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_slice ON decisions(slice);

CREATE TABLE IF NOT EXISTS outcomes (
  decision_id              TEXT PRIMARY KEY REFERENCES decisions(decision_id),
  run_id                   TEXT NOT NULL,
  task_passed              INTEGER NOT NULL,
  used_retrieved_content   INTEGER NOT NULL,
  response_text            TEXT NOT NULL,
  response_tokens          INTEGER NOT NULL,
  latency_ms               INTEGER NOT NULL,
  counterfactual_backend   TEXT,
  counterfactual_passed    INTEGER,
  timestamp_ms             INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcomes_run ON outcomes(run_id);
`;

export class Ledger {
  private db: Database.Database;

  constructor(path: string) {
    this.db = new Database(path);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this.db.exec(SCHEMA);
  }

  getRecordedOutcome(args: {
    run_id: string;
    case_id: string;
    backend: MemoryBackend;
  }): { task_passed: boolean } | null {
    const row = this.db
      .prepare(
        `SELECT o.task_passed AS task_passed
         FROM decisions d
         JOIN outcomes o ON o.decision_id = d.decision_id
         WHERE d.run_id = ? AND d.case_id = ? AND d.backend = ?`,
      )
      .get(args.run_id, args.case_id, args.backend) as { task_passed: number } | undefined;

    return row ? { task_passed: row.task_passed === 1 } : null;
  }

  clearIncompleteDecision(args: {
    run_id: string;
    case_id: string;
    backend: MemoryBackend;
  }): number {
    const result = this.db
      .prepare(
        `DELETE FROM decisions
         WHERE run_id = ?
           AND case_id = ?
           AND backend = ?
           AND decision_id NOT IN (SELECT decision_id FROM outcomes)`,
      )
      .run(args.run_id, args.case_id, args.backend);

    return result.changes;
  }

  logDecision(decision: Decision): void {
    this.db
      .prepare(
        `INSERT INTO decisions (
           decision_id, run_id, case_id, turn_index, backend, slice,
           gate_score, gate_threshold, fired, retrieved_json,
           injected_tokens, query_text, timestamp_ms
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        decision.decision_id,
        decision.run_id,
        decision.case_id,
        decision.turn_index,
        decision.backend,
        decision.slice,
        decision.gate_score,
        decision.gate_threshold,
        decision.fired ? 1 : 0,
        JSON.stringify(decision.retrieved),
        decision.injected_tokens,
        decision.query_text,
        decision.timestamp_ms,
      );
  }

  logOutcome(outcome: Outcome): void {
    this.db
      .prepare(
        `INSERT INTO outcomes (
           decision_id, run_id, task_passed, used_retrieved_content,
           response_text, response_tokens, latency_ms,
           counterfactual_backend, counterfactual_passed, timestamp_ms
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        outcome.decision_id,
        outcome.run_id,
        outcome.task_passed ? 1 : 0,
        outcome.used_retrieved_content ? 1 : 0,
        outcome.response_text,
        outcome.response_tokens,
        outcome.latency_ms,
        outcome.counterfactual_backend,
        outcome.counterfactual_passed === null ? null : outcome.counterfactual_passed ? 1 : 0,
        outcome.timestamp_ms,
      );
  }

  aggregate(args: { run_id: string; backend: MemoryBackend; slice: TurnSlice | "all" }): AblationResult {
    const { run_id, backend, slice } = args;
    const sliceClause = slice === "all" ? "" : "AND d.slice = @slice";
    const params = { run_id, backend, slice };

    const totals = this.db
      .prepare(
        `SELECT
           COUNT(*)                                        AS total_cases,
           SUM(d.fired)                                    AS total_fires,
           AVG(o.task_passed)                              AS pass_rate,
           AVG(CASE WHEN d.fired = 1 THEN o.task_passed END) AS fire_pass,
           AVG(CASE WHEN d.fired = 0 THEN o.task_passed END) AS nofire_pass,
           AVG(d.injected_tokens)                          AS mean_in,
           AVG(o.response_tokens)                          AS mean_out,
           SUM(o.task_passed)                              AS total_pass
         FROM decisions d
         JOIN outcomes o ON o.decision_id = d.decision_id
         WHERE d.run_id = @run_id AND d.backend = @backend ${sliceClause}`,
      )
      .get(params) as {
      total_cases: number;
      total_fires: number | null;
      pass_rate: number | null;
      fire_pass: number | null;
      nofire_pass: number | null;
      mean_in: number | null;
      mean_out: number | null;
      total_pass: number | null;
    };

    const regret = this.db
      .prepare(
        `SELECT COUNT(*) AS n
         FROM decisions d
         JOIN outcomes o ON o.decision_id = d.decision_id
         JOIN decisions d0 ON d0.case_id = d.case_id
                           AND d0.run_id = d.run_id
                           AND d0.backend = 'none'
         JOIN outcomes o0 ON o0.decision_id = d0.decision_id
         WHERE d.run_id = @run_id
           AND d.backend = @backend
           AND d.fired = 0
           AND o.task_passed = 0
           AND o0.task_passed = 1
           ${sliceClause}`,
      )
      .get(params) as { n: number };

    const harm = this.db
      .prepare(
        `SELECT COUNT(*) AS n
         FROM decisions d
         JOIN outcomes o ON o.decision_id = d.decision_id
         JOIN decisions d0 ON d0.case_id = d.case_id
                           AND d0.run_id = d.run_id
                           AND d0.backend = 'none'
         JOIN outcomes o0 ON o0.decision_id = d0.decision_id
         WHERE d.run_id = @run_id
           AND d.backend = @backend
           AND d.fired = 1
           AND o.task_passed = 0
           AND o0.task_passed = 1
           ${sliceClause}`,
      )
      .get(params) as { n: number };

    const totalPass = totals.total_pass ?? 0;
    const meanIn = totals.mean_in ?? 0;
    const meanOut = totals.mean_out ?? 0;

    return {
      backend,
      slice,
      total_cases: totals.total_cases,
      total_fires: totals.total_fires ?? 0,
      pass_rate: totals.pass_rate ?? 0,
      fire_conditional_pass_rate: totals.fire_pass,
      nofire_conditional_pass_rate: totals.nofire_pass,
      abstention_regret_count: regret.n,
      false_fire_harm_count: harm.n,
      mean_injected_tokens: meanIn,
      mean_response_tokens: meanOut,
      tokens_per_pass: totalPass > 0 ? (totals.total_cases * (meanIn + meanOut)) / totalPass : Number.POSITIVE_INFINITY,
    };
  }

  close(): void {
    this.db.close();
  }
}
