# Reproduce Evaluation + Proofs

This document is the reproducibility entrypoint for the current public OpenClawBrain surface.

## 1) Bootstrap the TypeScript workspace

```bash
cd /path/to/openclawbrain
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```

## 2) Reproduce the mechanism proof in this repo

These are the proofs that are implemented directly in the TypeScript workspace today.

### Lifecycle proof

```bash
pnpm lifecycle:smoke
```

This proves the package-first lifecycle across:
- normalized events
- event export
- learner pack materialization
- activation staging/promotion
- compiler runtime compilation

### Observability proof

```bash
pnpm observability:smoke
```

This proves the operator-facing diagnostics surface for:
- activation health
- promotion readiness
- freshness inspection
- deterministic priority fallback

## 3) Reproduce comparative benchmark proof families

The larger benchmark harness currently lives in the sibling public proof repo:
- `https://github.com/jonathangu/brain-ground-zero`

That repo is the source of truth for the currently published benchmark families such as recorded-session head-to-head and sparse-feedback proof bundles. It is separate from this repo's supported TypeScript package surface.

### Quickstart for the proof harness repo

```bash
git clone https://github.com/jonathangu/brain-ground-zero.git
cd brain-ground-zero
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m brain_ground_zero.cli smoke
python scripts/validate_configs.py
```

### Example benchmark runs in the proof harness repo

```bash
python -m brain_ground_zero.cli run \
  --family configs/families/relational_drift.yaml \
  --baselines configs/baselines/all.yaml

python -m brain_ground_zero.cli multiseed \
  --family configs/families/relational_drift.yaml \
  --baselines configs/baselines/all.yaml \
  --seeds 10,20,30,40,50,60,70,80,90,100
```

## Claim boundary

### Proven directly in this repo today
- package build/typecheck
- mechanism proof via lifecycle smoke
- operator observability proof via observability smoke
- publishable package surface and registry install path

### Proven in the separate public proof repo today
- comparative benchmark bundles
- recorded-session head-to-head proof family
- sparse-feedback proof family

### Not claimed here
- live production answer-quality proof on served OpenClaw traffic
- shadow-mode or online rollout proof

## Related docs
- [openclaw-integration.md](openclaw-integration.md)
- [openclaw-attach-quickstart.md](openclaw-attach-quickstart.md)
- [operator-observability.md](operator-observability.md)
