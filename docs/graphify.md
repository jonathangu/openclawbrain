# Graphify in OpenClawBrain

## Short version

OpenClawBrain now ships with a **Graphify bridge**.

That bridge is **real**, **useful**, and **intentionally off the hot path**.

Graphify is not the live answer engine. It is an **offline compiler / diagnostics surface** that helps OpenClawBrain in two places:

1. **Cold start** — build stronger initial structure before a home has much learned personal history
2. **Maintenance / observability** — show drift, provenance gaps, and graph-vs-OCB differences in a bounded operator surface

The live runtime still serves **promoted OpenClawBrain packs**, not live Graphify output.

## What shipped in 0.4.36

The shipped Graphify lane includes:

- canonical source-bundle export
- Graphify projection export
- managed Graphify run lane
- compiled-artifact pack bridge
- deterministic Graphify/OCB lints
- conservative **EXTRACTED-only** import slice
- candidate-pack input bridge
- maintenance diff lane
- final replay / eval proof lane

## What we proved

The final proof packet showed:

- **`graphify_artifacts_only` won cold start** in the evaluated packet
- **`graphify_import + learned_route` beat the learned-route baseline** without Graphify import
- deterministic lints and maintenance diff are **useful diagnostic tools**, but **not live truth authority**

That is the product truth.

## What Graphify is for

Graphify belongs in the **artifact-first compiler / diagnostic lane**.

Use it to:

- build stronger cold-start priors
- compile inspectable off-path artifacts
- surface maintenance differences between Graphify and OCB
- highlight provenance gaps and review targets
- improve operator understanding of the graph structure

## What Graphify is not for

Graphify is **not**:

- the hot-path runtime answer engine
- current-truth authority
- a replacement for explicit corrections
- a live dependency on every serve
- a license to mutate the graph live without replay / promotion

OpenClawBrain still keeps the same core runtime contract:

- bounded live path
- learned routing in the live path
- off-path learning and compilation
- replay / promotion before live serving
- fail-open behavior when the memory layer cannot safely help

## The right mental model

Think of the shipped Graphify feature as:

> **an off-path graph compiler and maintenance lens for OpenClawBrain**

not:

> **a new runtime that replaced OpenClawBrain’s serve path**

That distinction is the important one.

## Operator impact

The public install story does **not** change.

You still use the normal OpenClawBrain lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Graphify improves the shipped product behind that install lane. It does not add a second operator workflow.
