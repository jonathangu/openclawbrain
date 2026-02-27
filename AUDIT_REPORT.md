# OpenClawBrain Audit Report

## Scope
Reviewed all `openclawbrain/*.py` and `tests/*.py` files in the repository.

## Findings

### 1) CRITICAL — State persistence is not crash-safe (non-atomic writes)
- **File/Line:** `openclawbrain/store.py:146`
- **Issue:** `save_state` writes JSON directly to final path via `write_text`.
- **Impact:** A daemon crash or process termination during write can leave `state.json` truncated/corrupted, causing full state loss on next startup.
- **Suggested fix:** Write to a temp file in the same directory, `fsync`, then `os.replace` atomically.

### 2) HIGH — No protocol-level authentication/authorization on daemon socket
- **File/Line:** `openclawbrain/socket_server.py:271`, `openclawbrain/socket_server.py:249-253`
- **Issue:** Local socket server exposes write paths (`inject`, `learn`, `maintain`, `shutdown`) without any auth/ticket and trusts any process that can connect.
- **Impact:** Any local user with filesystem access to the socket path can mutate state or trigger daemon shutdown.
- **Suggested fix:** Require per-process auth token or Unix socket peer credential checks and reject unauthenticated requests.

### 3) HIGH — Unbounded request/response frame sizes enable DoS via oversized payloads
- **File/Line:** `openclawbrain/socket_server.py:203-205`, `openclawbrain/socket_server.py:153-161`
- **Issue:** No request/response length cap; `readline()` and `json.loads()` on arbitrary payloads.
- **Impact:** Very large malicious messages can consume memory/CPU and block the server (or cause JSON decode failures repeatedly).
- **Suggested fix:** Enforce hard frame-size limits and reject requests above threshold before `json.loads`.

### 4) HIGH — Replay auto-scoring uses substring match, causing false positives
- **File/Line:** `openclawbrain/replay.py:379`
- **Issue:** `_auto_score_query_outcome` rewards a query if any fired node content is a raw substring of response text.
- **Impact:** Common tokens/phrases can incorrectly increase/reverse outcomes and distort edge weights at scale.
- **Suggested fix:** Match on token/phrase boundaries (e.g., token overlap/Jaccard or semantic similarity) instead of raw substring membership.

### 5) MEDIUM — Maintenance `dry_run` still performs full deep copies
- **File/Line:** `openclawbrain/daemon.py:529-530`, `openclawbrain/maintain.py:269-270`
- **Issue:** Dry-run paths copy entire graph and index (`copy.deepcopy`) before each operation.
- **Impact:** 1000+ node graphs can incur high CPU/memory spikes and frequent allocations, hurting daemon latency.
- **Suggested fix:** Use copy-on-write/read-only snapshots or lazy clone-on-write for maintenance tasks.

### 6) MEDIUM — `replay_queries` re-snapshots full edge set per query
- **File/Line:** `openclawbrain/replay.py:509-523`
- **Issue:** For each replayed query it snapshots and diffs the full edge map.
- **Impact:** O(Q * E) memory/time during replay; expensive with large graphs.
- **Suggested fix:** Use `apply_outcome` return value (`updates`) to accumulate changed edges incrementally.

### 7) MEDIUM — `apply_connections` rewrites existing edges without reporting/update intent
- **File/Line:** `openclawbrain/connect.py:147-159`
- **Issue:** Existing `(source,target)` edge is silently overwritten every call, but `added` count only increments for new edges.
- **Impact:** Repeated maintenance/connect passes can churn edge weights and produce misleading metrics while silently mutating graph semantics.
- **Suggested fix:** Gate updates to changed weights only (or expose explicit `overwrite` semantics), and surface update vs add counts separately.

### 8) MEDIUM — `split_workspace` fails hard on non-UTF8 file contents
- **File/Line:** `openclawbrain/split.py:411`
- **Issue:** `read_text(encoding="utf-8")` has no error handling.
- **Impact:** A single non-UTF8 workspace file causes the entire split pass to fail and blocks downstream sync/maintenance.
- **Suggested fix:** Use explicit fallback decoding (`errors="replace"`) or detect/skip unsupported encodings.

### 9) MEDIUM — Journal readers are not resilient to malformed records
- **File/Line:** `openclawbrain/journal.py:85-87`
- **Issue:** `read_journal()` assumes every non-empty line is valid JSON and will raise on one malformed line.
- **Impact:** One corrupted journal entry can break health/journal/replay workflows.
- **Suggested fix:** Skip bad lines (with a warning) and continue processing remaining entries.

### 10) LOW — Dependency pins are broad (supply-chain and behavior drift risk)
- **File/Line:** `pyproject.toml:20`, `pyproject.toml:21`
- **Issue:** Optional dependencies are version-range unbounded (`openai>=1.0`, `sentence-transformers>=2.0`).
- **Impact:** Upstream API/behavior changes can break production behavior unexpectedly after environment rebuilds.
- **Suggested fix:** Pin to known-good versions and include lockfile/pinning strategy.

## Test coverage gaps

1. **Input validation / malformed protocol payloads**
   - `tests/test_socket.py` focuses on happy-path socket calls and does not validate oversized payloads, malformed JSON, invalid `id` types, or `shutdown` authorization paths.

2. **State corruption recovery paths**
   - No tests assert daemon recovery after interrupted/corrupt `state.json` writes.

3. **Replay outcome scoring edge cases**
   - Existing replay tests cover positive auto-score, but not false-positive/substring-only matches in `_auto_score_query_outcome`.

4. **Encoding/robustness in workspace split and sync**
   - `tests/test_split.py` and `tests/test_sync.py` do not include non-UTF8/binary files or large files to validate split pipeline degradation.

5. **Concurrency/restart behavior**
   - Tests do not cover request races during daemon restart (simultaneous socket requests + `SIGTERM`/watchdog restart path).
