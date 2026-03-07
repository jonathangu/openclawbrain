# Reproduce Evaluation + Proofs

This document is the reproducibility entrypoint for the current public OpenClawBrain surface.

## 1) Bootstrap the workspace

```bash
cd /path/to/openclawbrain
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:status
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
pnpm observability:report
```

This proves the operator-facing diagnostics surface for:

- activation health
- promotion freshness
- rollback readiness and rollback lineage
- supervision freshness by source
- teacher freshness
- async teacher-loop no-op detection
- learned `route_fn` freshness/version
- graph-dynamics freshness
- learned `route_fn` evidence
- explicit fallback usage

`pnpm observability:report` prints the local JSON report for those proofs. It only claims what is materialized inside the repo fixture lane; it does not claim live production telemetry coverage.

## 3) Reproduce outside-consumer proof from local release tarballs

Use the checked-in outside-consumer smoke after `pnpm release:pack` creates `.release/` tarballs:

```bash
repo_root="$(pwd)"
tmpdir="$(mktemp -d)"
cp examples/npm-consumer/package.json "$tmpdir/package.json"
cp examples/npm-consumer/smoke.mjs "$tmpdir/smoke.mjs"
cd "$tmpdir"
npm install \
  "$repo_root/.release/openclawbrain-contracts-0.1.1.tgz" \
  "$repo_root/.release/openclawbrain-event-export-0.1.1.tgz" \
  "$repo_root/.release/openclawbrain-events-0.1.1.tgz"
npm run smoke
```

This is the truthful outside-consumer proof for the current repo-only wave. It proves the current tarballs install cleanly with plain `npm` without claiming that the registry has already been updated.

## 4) Optional post-publish registry proof

After a matching `v0.1.1` tag has shipped and post-publish checks succeed, you can rerun the same smoke using registry versions instead of local tarball paths.

## Claim boundary

### Proven directly in this repo today

- package build and typecheck
- mechanism proof via lifecycle smoke
- operator observability proof via observability smoke/report
- versioned package surface and local tarball install path

### Maintained separately

Broader comparative benchmark families live in the separate public proof repo `brain-ground-zero`.

Use that repo's own instructions for benchmark reproduction.

### Not claimed here

- full comparative benchmark coverage inside this repo
- live production answer-quality proof on served OpenClaw traffic
- shadow-mode or online rollout proof
- npm publication for the current wave unless post-publish verification has been completed

## Related docs

- [openclaw-integration.md](openclaw-integration.md)
- [openclaw-attach-quickstart.md](openclaw-attach-quickstart.md)
- [operator-observability.md](operator-observability.md)
