# BUG SWEEP #4 — Deep integration, decay, autotune, feedback, and remaining untested paths

## Scope
- Covered modules and paths requested in Bug Sweep #4: `decay.py`, `autotune.py`, `feedback.py`, `router.py`, `migrate.py`, `mitosis.py`, CLI integration flows, and README examples.
- Executed full test suite: `python3 -m pytest --tb=short -q`.
- Result: `357 passed, 0 failed`.

## Findings and fixes

### 1) `crabpath/decay.py` — decay rate path ignored in edge recalculation
- Severity: **High**
- Issue: `decay_rate` override could be supplied, but `decay_weight()` ignored it and always used `half_life_turns` logic, causing incorrect weight updates and breaking `decay_rate=1.0` behavior.
- Fix: Passed `config.decay_rate` into the `decay_factor()` call inside `decay_weight()`.
- Test: `tests/test_decay.py::test_decay_rate_zero_and_one`
- File: `crabpath/decay.py`

### 2) `crabpath/migrate.py` — `extract_session` accepted non-user JSON-like message wrappers as queries
- Severity: **Medium**
- Issue: In `type="message"` records, when `message` was a JSON-like string for assistant/system content, fallback to raw string parsing treated it as a query.
- Fix: In message extraction, skip raw-string fallback for structured payload-like text (starts with `{`/`[` and ends with `}`/`]`), preventing non-query payloads from being misclassified.
- Tests: `tests/test_cli.py::test_extract_sessions_command`, `tests/test_migrate.py::test_parse_session_logs_multiple_openclaw_formats`
- File: `crabpath/migrate.py`

### 3) `crabpath/migrate.py` — empty input and malformed transcripts/unsafe files
- Severity: **Low**
- Issue addressed by tests and defensive parsing:
  - empty file handling and malformed JSON are tolerated without raising,
  - only user-like records and valid message payloads are extracted,
  - symlinks/hidden/binary files are ignored during workspace gather.
- Tests:
  - `tests/test_migrate.py::test_parse_session_logs_empty_and_malformed`
  - `tests/test_migrate.py::test_gather_files_ignores_symlinks_hidden_and_binary`
- File: `crabpath/migrate.py`

### 4) `crabpath/router.py` — empty candidates / all-zero weights / temperature behavior
- Severity: **Medium**
- Issue: routing path had unstable behavior for empty candidate sets and non-positive weighted candidates.
- Fixes implemented:
  - safe empty `RouterDecision` for empty inputs,
  - fallback suppresses routing decision when all candidate weights are non-positive,
  - added explicit temperature-path test coverage.
- Tests:
  - `tests/test_router.py::test_decide_next_empty_candidates_returns_empty_decision`
  - `tests/test_router.py::test_select_fallback_returns_empty_for_all_zero_weights`
  - `tests/test_router.py::test_decide_next_uses_temperature_when_llm_client_configured`
- Files: `crabpath/router.py`

### 5) `crabpath/autotune.py` — health metrics for degenerate graphs
- Severity: **Medium**
- Issue: health metrics needed validation for empty/no-edge/all-inhibitory graphs.
- Fixes: Added tests to verify outputs remain valid, bounded percentages, and expected target ranges.
- Tests:
  - `tests/test_autotune.py::test_measure_health_empty_graph_returns_zeroed_metrics`
  - `tests/test_autotune.py::test_measure_health_no_edges_returns_zero_percentages`
  - `tests/test_autotune.py::test_measure_health_all_inhibitory_edges`
  - `tests/test_autotune.py::test_health_targets_have_typical_metric_ranges`

### 6) `crabpath/feedback.py` — outcome, snapshot mapping, and path handling
- Severity: **Low**
- Issue: edge-case behavior for zero-turns/no corrections, malformed lines, and cross-platform snapshot path handling.
- Fixes and checks:
  - `auto_outcome` returns neutral `0.0` when there is no elapsed signal and no corrections,
  - malformed JSON lines skipped during raw snapshot load,
  - already-attributed records filtered from correction mapping,
  - snapshot path no longer assumes pre-existing parent directories and supports direct env-provided path.
- Tests:
  - `tests/test_feedback.py::test_auto_outcome_classification_and_edge_cases`
  - `tests/test_feedback.py::test_map_correction_to_snapshot_ignores_invalid_records`
  - `tests/test_feedback.py::test_snapshot_path_supports_cross_platform_path`
- File: `crabpath/feedback.py`

### 7) `crabpath/mitosis.py` — workspace bootstrap, timeout fallback, and division history tracking
- Severity: **Medium**
- Issue: empty directories and LLM failures in split needed safe fallback behavior and state invariants.
- Fixes and checks:
  - bootstrap handles empty directories,
  - `split_with_llm` falls back cleanly on timeout and parse/transport errors,
  - division history tracking verified via `MitosisState`.
- Tests:
  - `tests/test_mitosis.py::test_bootstrap_workspace_empty_dir`
  - `tests/test_mitosis.py::test_split_with_llm_timeout_uses_fallback`
  - `tests/test_mitosis.py::test_bootstrap_tracks_split_history`
- File: `crabpath/mitosis.py`

### 8) Integration and examples
- Severity: **Low**
- CLI integration cycles added/verified:
  - fresh-graph query/learn/query flow,
  - init → query → explain → learn → health flow,
  - snapshot save/load/feedback + extraction/CLI smoke checks.
- README API example fixes:
  - corrected `MemoryController` result access (`result.context`), removed stale `guardrails` usage, and corrected inhibitory API examples.
- Tests:
  - `tests/test_cli.py::test_cli_query_learn_query_cycle`
  - `tests/test_cli.py::test_init_query_explain_learn_health_cycle`
  - existing CLI coverage for extract/snapshot/learn/explain flows.
- Files: `tests/test_cli.py`, `README.md`

## Command run
- `python3 -m pytest --tb=short -q`
- Result: all tests pass.
