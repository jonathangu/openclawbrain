# CrabPath Architectural Review

Date: 2026-02-26

## Executive assessment

The architecture is strong in concept and has the right ingredients, but it currently has one foundational coupling problem: routing, learning, and inhibition are not governed by a single **authoritative memory policy layer**. The result is inconsistent behavior across call paths (especially simulator/legacy paths), duplicated node/structural code, and hidden negative-signal semantics.

## 1) Most important architectural change (single structural change)

### Create a first-class `MemoryMemoryEngine` (or `MemoryController`) as the single weighted-graph brain

**Change**: Introduce one orchestration layer that owns
- edge retrieval,
- action scoring/routing,
- phase-dependent learning rules,
- inhibition enforcement,
- and baseline/episode grouping policy.

This should become the only code path for traversal and learning decisions.

#### Why this one change fixes the largest cluster of issues
- It makes routing impossible without weights and policy: router can no longer run a mock/fallback path that ignores graph state.
- It gives a single source of truth for inhibitory signal handling (currently split across synaptogenesis + traversal/learning assumptions).
- It encodes phase behavior structurally rather than implicitly.
- It prevents simulator/CLI/MCP divergence by forcing all execution through the same policy engine.

#### Concrete file touch points
- `router.py` and `traversal.py` should delegate to the controller's policy interface, removing direct heuristic fallback logic.
- `learning.py` should expose only learning updates and never select edges directly.
- `synaptogenesis.py` and `feedback.py` should write inhibition/correction events into the controller store, not into arbitrary graph-adjacent side channels.
- `adapter.py`, `mcp_server.py`, `cli.py`, and `lifecycle_sim.py` should share one runtime orchestration path.

Evidence of current mismatch:
- In `traversal.py`, when no habitual/reflex candidates exist traversal halts (`if not candidate_edges: return current_node`) without policy fallback that can still reason over low-salience edges. 
- In `router.py`, `decide_next()` still contains fallback selection that can run even when edge-derived signal is unavailable.
- `learning.py` supports family-level baseline grouping but only receives that state if callers provide appropriate keys.

## 2) What to merge or split

### Merge (high priority)

1. `lifecycle_sim.py` + `simulator.py` → one simulation module
- `lifecycle_sim.py` (`473` lines) and `simulator.py` (`172` lines) currently overlap in run-loop responsibilities.
- Keep richer lifecycle simulation and deprecate/reduce the lightweight simulator to compatibility wrappers.
- This removes divergent semantics and prevents future regressions between "toy" and full-loop behavior.

2. `mitosis.py` + `neurogenesis.py`
- `mitosis.py` already implements node creation (`should_create_node`, `create_node`) plus structural splitting/merging; `neurogenesis.py` is a lighter duplicate for novelty/connect path.
- Merge into a single `neurogenesis.py` or `cell_growth.py` module with strategy pattern:
  - deterministic split rules (current `neurogenesis` heuristics)
  - LLM-guided restructuring (current `mitosis` responsibilities)
- This avoids conflicting node lifecycles.

3. `consolidation.py` + `graph.py`
- `graph.py` already has `consolidate()` and structural mutation helpers.
- `consolidation.py` duplicates consolidation and maintenance helpers with thin wrappers; move unused/dubious helpers into `graph.py` and keep consolidation as legacy or remove.

### Split (high priority)

1. `synaptogenesis.py` (promotion/creation) + `inhibition.py` (new)
- Inhibition is a core differentiator from BM25 but is currently a side effect in correction logic.
- Split suppression mechanics (edge-sign updates, negative evidence policy, correction decay, inhibition-aware tier/ranking) into dedicated module(s).

2. `learning.py` into `policy_learning.py` + `credit_assignment.py`
- The file currently mixes RL config, reward bookkeeping, baseline/state extraction, REINFORCE math, and skip/weight guardrails.
- Keep the core estimator in a policy module and move reward normalization/signaling into a separate subsystem.

## 3) Inhibition pathway: promote to first-class module

Current status (issue): inhibition exists but implicit
- `synaptogenesis.py` `record_correction()` creates/updates inhibitory edges.
- `traversal.py` and `_structural_utils.classify_edge_tier` do not consistently enforce a separate inhibitory tier in route selection.
- Some callers rely on side-effect edge-weight sign, others ignore it.

Architectural recommendation:
1. Add `inhibition.py` with explicit `apply_inhibition(event, context)` contract.
2. Track per-edge attributes:
   - `weight`, `sign` (excitatory/inhibitory/neutral), `source`, `target`, `confidence`, `decay_half_life`.
3. Introduce explicit combine rule used by routing:
   - `effective_score = excitatory_score - inhibitory_score * lambda` (or equivalently signed weights with separate clipping).
4. Ensure traversal ignores only true dormant/excluded edges; inhibitory edges remain visible for suppression effects rather than silently removed.
5. Route selection and learning should use the same inhibition-aware scorer.

Relevant code location evidence:
- `synaptogenesis.py` correction path around `record_correction` (negative edge creation/decay logic).
- `feedback.py` calls this path for corrected signals.
- `_structural_utils.py` tiering currently classifies by magnitude only; no robust inhibitory mode exposed to traversal policy.

## 4) Two-phase learning: encode explicitly

Current status (issue): developmental sequence is emergent only
- Hebbian bootstrapping and policy-gradient refinement are observed operationally, but not governed structurally.
- This causes phase bleed-through and inconsistent outcomes when query distribution shifts.

Proposed architecture:
1. Add explicit `LearningPhaseManager` with per-graph and/or per-domain phase state.
2. Define phase transition signals (hard gates with hysteresis), e.g.:
   - Phase 0 (bootstrap): PG off, Hebbian+neurogenesis on, high exploration/structure growth.
   - Phase 1 (competition): mixed update, controlled proto-edge promotion.
   - Phase 2 (policy): PG dominant, Hebbian damped/targeted, stronger anti-overfit regularization.
3. Drive transitions from stable metrics, not manual toggles:
   - edge activation entropy, novelty rate, reward variance, retrieval latency/coverage.
4. Enforce module behavior by phase:
   - `learning.py` skip/update gates become explicit phase guards.
   - `synaptogenesis.py` competition/skip penalties and proto-edge thresholds become phase-scaled.

This makes the behavior reliable and reproducible across environments and simulation/tooling.

## 5) Dead code / removable technical debt

### Strong candidates for removal or strict deprecation
1. `simulator.py`
- Narrow surface and overlap with `lifecycle_sim.py`.
- Keep only thin compatibility wrappers if needed by external scripts.

2. `activation.py`
- `adapter.query` and most runtime paths are traversal-led; current activation implementation appears legacy/parallel.
- If you want an alternate spiking mode, isolate behind explicit `--engine=activation` flag and move to `legacy/`.

3. `consolidation.py`
- Overlaps with graph-level structural methods and mitosis/maintenance paths.
- Either absorb used functions into canonical modules or archive.

4. `_structural_utils.py` helpers that are not on current critical path
- `parse_markdown_json`/some diagnostics helpers should be moved into CLI/util tools if they are only used for command parsing/logging.
- Keep only policy-critical helpers (`classify_edge_tier`-like) centralized if still used.

### Ambiguous but likely dead
- Dormant/reflex thresholds are defined inconsistently across modules (`_structural_utils.py` vs `synaptogenesis.py` config defaults), so several helper thresholds likely intended for one phase/config were never fully wired.
- Check and remove old constants/settings after unifying this into controller-driven policy.

## 6) Graph / Node / Edge primitives

Current primitives are probably under-specified for a production graph-memory system.

`Node` in `graph.py` appears to track only `id/content/summary/metadata` and edge representation is basic (`source/target/weight`).

Recommended additions:
- `Edge`
  - `kind` (`excitatory|inhibitory|neutral`)
  - `evidence_count`, `last_touched`, `learn_count`
  - `source`/`target` confidence fields
  - `provenance`: rule or event that created/modified edge
  - `temperature` / `policy_logit` cache for traversal scoring
  - `decay_group` or `ttl` for maintenance
- `Node`
  - `embedding` fingerprints / vector cache metadata (for rapid similarity family mapping)
  - `cluster_id` / `archetype` for two-phase family baselines
  - `activation_budget`, `access_count`, `failure_count`, `relevance_decay`
- `Graph`
  - explicit `edge_index` + `edge_stats` map keyed for O(1) lookup
  - graph-level `learning_phase`, `health_snapshot`, and `policy_version`

Without these, `learning.py` and `feedback.py` keep re-deriving context from weak metadata, and inhibition/phase policy remains hardcoded/implicit.

## 7) Architecture rating

### Current score: **6.5 / 10**

What works
- Clear experimental architecture with working Hebbian + policy-gradient + structural adaptation loops.
- Strong module separation by algorithm family.
- Autotune already measures useful health metrics and exposes control knobs.

What prevents 9/10
- No single source of truth for route/learn policy (routing fallback paths + partial integration of modules).
- Inhibition is effective but not canonical.
- Two-phase learning not encoded, leading to phase drift.
- Duplicate simulation and structural code paths.
- Inconsistent tier thresholds across modules.
- Legacy pathways (activation/mocks/consolidation split) still compete with canonical traversal flow.

### What gets it to 9
1. Implement the controller/orchestrator change above.
2. Make phase explicit and phase-aware.
3. Promote inhibition to dedicated module + explicit scorer.
4. Collapse duplicate modules and prune dead code.
5. Unify thresholds, policy contracts, and telemetry schema across CLI/MCP/adapter/sim.

## Concrete references used

- `graph.py` (~446 lines): primitives and traversal/storage operations.
- `traversal.py` (~264 lines): tier-aware candidate gating and stop conditions.
- `router.py` (~512 lines): routing fallback behavior and selection flow.
- `learning.py` (~341 lines): REINFORCE, baselines, skip/decay updates, episode key handling.
- `synaptogenesis.py` (~398 lines): proto-edge lifecycle + correction/skip updates (inhibition side effect).
- `mitosis.py` (~744 lines): LLM split/merge/node-creation and lifecycle growth.
- `neurogenesis.py` (~137 lines): heuristic novelty-based node creation/connecting.
- `activation.py` (~226 lines): currently parallel activation/stdp path, weakly integrated in production flow.
- `simulator.py` (~172 lines) and `lifecycle_sim.py` (~473 lines): overlapping simulation concerns.
- `feedback.py` (~219 lines): correction detection and edge correction injection path.
- `_structural_utils.py` (~127 lines): tiering utility and JSON helpers.
- `autotune.py` (~1187 lines): health metrics and knobs (target thresholds should be reconciled with phase routing policy).
- `adapter.py` (~452 lines), `cli.py` (~870 lines), `mcp_server.py` (~778 lines): multiple entry points that currently can reach different logic branches.

## Migration sequence (minimal-risk)

1. Add policy controller interface first and move all score/selection into it while keeping existing modules as callees.
2. Gate learning updates by phase in controller.
3. Move `record_correction` and inhibitory effects into explicit suppression module.
4. Merge simulation and neurogenesis pathways.
5. Remove or archive dead modules and align all entry points to canonical controller path.

