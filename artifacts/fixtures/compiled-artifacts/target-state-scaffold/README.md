# Compiled artifact example pack

Status: target-state scaffolding.

This fixture pack makes the compiled-artifact substrate concrete without claiming that the runtime already writes these artifacts.

## What is included

- `ca_example_concept_substrate_01` — a concept page for the compiled-artifact pair shape
- `ca_example_map_of_territory_01` — a map-of-territory page that keeps live truth and derived surfaces separate
- `ca_example_provenance_gap_report_01` — a provenance-gap report that names the follow-on implementation gaps

Each artifact is stored as:

- `artifact.md` for the human-readable body
- `artifact.meta.json` for the canonical sidecar metadata

## How later implementation should use this pack

- Treat the markdown body as immutable once written.
- Treat the sidecar as the machine-readable source of truth for the compiled artifact record.
- Recompute `contentHash` when the body changes.
- Keep evidence references explicit so compiler and lint code can replay the example shape.
- Preserve the target-state/scaffolding boundary in any UI or proof surface that consumes these files.

## Guardrails

This pack is derived from docs only. It is not a runtime proof bundle, not a promoted pack, and not a replacement for live authority.
