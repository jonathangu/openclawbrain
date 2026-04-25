# Cold-start router sample curation

This directory holds a tiny checked docs/QA sample used to prove the cold-start data compilation path end to end.

## What is here

- `docs-qa-sample.raw.json` — approved raw source bundle for the sample compiler

## Source family

- docs/QA, derived from:
  - `docs/architecture/routing-prior.md`
  - `docs/architecture/teacher-v3-lints.md`

## What the sample exercises

- registry entry construction
- evidence-span resolution from real docs text
- query / rationale trimming
- cursor-path cleanup
- candidate deduplication and score-based canonicalization
- hard-negative normalization
- teacher-label and verifier emission
- graph-compiler contract synthesis

## Notes

The checked fixture is intentionally small. It is not a dataset dump.
