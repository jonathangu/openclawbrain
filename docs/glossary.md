# Glossary

**Pack**: An immutable OpenClawBrain artifact directory containing a manifest plus graph, vector, provenance, and optional router payloads.

**Candidate pack**: A newly assembled pack staged for validation before activation.

**Active pack**: The promoted pack currently selected for compilation.

**Activation pointers**: Small metadata records that identify the active, candidate, and previous packs.

**Runtime compile**: Deterministic pack-backed context compilation for a single OpenClaw request.

**Learned `route_fn`**: The router artifact stored in a pack and used when the manifest requires learned routing.

**Native structural compaction**: Deterministic merging and truncation of pack-backed context blocks under an explicit compile budget.

**Normalized event export**: The learner input shape that packages interaction and feedback events with stable provenance metadata.

**Route policy**: The manifest-level rule that either allows heuristic compilation or requires the promoted pack's learned `route_fn`.
