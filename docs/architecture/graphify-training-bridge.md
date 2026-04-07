# Graphify training-review bridge

Status: repo-native bridge note.

This note defines the smallest useful bridge from Graphify-derived outputs into reviewable training-row inputs.

The bridge is intentionally **review-first**:

- route rows stay canonical
- explicit corrections stay above Graphify
- Graphify-derived context stays off the hot path
- only EXTRACTED neighborhood context can contribute to candidate training input
- provenance-gap findings remain review-only and never become live truth

## Goal

Graphify can help explain a route row, but it must not replace the route row.

The useful output is a bounded training-review bundle that can attach:

- a canonical route row
- EXTRACTED neighborhood context from a compiled artifact pack / import slice
- review-only hard-negative support notes
- review-only provenance-gap hints

That gives downstream learning jobs something inspectable without letting Graphify outrank raw authority or explicit correction.

## Review bundle shape

A training-review bundle should keep the original row intact and add a sidecar review view:

- `route_row_core` — stable, bounded subset of the route row
- `graphify_neighborhood_context` — EXTRACTED neighborhood priors and evidence pointers
- `hard_negative_supports` — Graphify-supported notes explaining why a canonical hard negative remains reviewable
- `provenance_gap_hints` — diagnostic hints pulled from Graphify provenance-gap reporting
- `review_boundary` — explicit trust boundary proving Graphify remains subordinate

## Truth boundary

The boundary is the same one used throughout the Graphify bridge:

1. runtime truth
2. proof truth
3. docs truth
4. Graphify proposal truth

Graphify review surfaces must also honor correction precedence:

- explicit correction memory
- recent raw user/source turns
- raw proof/runtime evidence
- frozen docs truth

Anything in the review bundle that is INFERRED or AMBIGUOUS stays review-only.
Only EXTRACTED neighborhood context may feed a later live-eligible import slice.

## What counts as value

This bridge is useful if it can answer questions like:

- what neighborhood did this route row come from?
- which evidence pointers make the row reviewable?
- which hard negatives are still supportable under the Graphify-derived neighborhood view?
- which provenance gaps still block promotion or import?

If it cannot answer those questions in a bounded way, it is too vague and too strong.

## What it is not

This bridge does **not**:

- mutate route-row truth
- store correction-like memory
- add a serve-path dependency
- turn provenance gaps into live labels
- widen import beyond EXTRACTED context

The bridge exists to produce reviewable training value, not to create a second source of truth.
