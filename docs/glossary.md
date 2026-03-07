# Glossary

**Pack**: An immutable OpenClawBrain artifact directory containing a manifest plus graph, vector, and optional router payloads.

**Candidate pack**: A newly assembled pack staged for validation before activation.

**Active pack**: The pack currently selected for runtime compilation.

**Activation pointers**: Small metadata records that identify the active, candidate, and previous packs.

**Runtime compile**: Deterministic context selection over an active pack for a single OpenClaw request.

**Normalized event export**: The learner input shape that packages interaction and feedback events with stable provenance metadata.

**Route policy**: The manifest-level rule that allows heuristic compilation or requires learned routing artifacts.
