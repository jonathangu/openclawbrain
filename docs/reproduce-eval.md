# Reproduce Evaluation + Proofs

This document is the reproducibility entrypoint for the current public OpenClawBrain surface.

## 1) Bootstrap the workspace

```bash
cd /path/to/openclawbrain
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```

## 2) Reproduce the mechanism proofs in this repo

These are the proofs implemented directly in the public package workspace today.

### Lifecycle proof

```bash
pnpm lifecycle:smoke
```

This proves the learning lifecycle across:

- normalized events
- event export
- learner pack materialization
- activation staging and promotion
- promoted-pack compilation

### Observability proof

```bash
pnpm observability:smoke
```

This proves the operator-facing diagnostics surface for:

- activation health
- promotion readiness
- freshness inspection
- learned `route_fn` evidence
- deterministic fallback diagnostics

## 3) Reproduce consumer proof from the published packages

Use the checked-in outside-consumer smoke:

```bash
tmpdir="$(mktemp -d)"
cp examples/npm-consumer/package.json "$tmpdir/package.json"
cp examples/npm-consumer/smoke.mjs "$tmpdir/smoke.mjs"
cd "$tmpdir"
npm install
npm run smoke
```

## Claim boundary

### Proven directly in this repo today

- package build and typecheck
- mechanism proof via lifecycle smoke
- operator observability proof via observability smoke
- publishable package surface and registry install path

### Maintained separately

Broader comparative benchmark families live in the separate public proof repo `brain-ground-zero`.

Use that repo's own instructions for benchmark reproduction.

### Not claimed here

- full comparative benchmark coverage inside this repo
- live production answer-quality proof on served OpenClaw traffic
- shadow-mode or online rollout proof

## Related docs

- [openclaw-integration.md](openclaw-integration.md)
- [openclaw-attach-quickstart.md](openclaw-attach-quickstart.md)
- [operator-observability.md](operator-observability.md)
