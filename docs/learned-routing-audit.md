# Learned Routing Audit

Learned routing is the top invariant. This doc traces the lifecycle, surfaces configuration knobs, and provides a concrete operator checklist to verify that learned routing is active end-to-end.

**Scope:** route_fn + route_model + labels/traces + training + loop + serve + hook + report/health.

## Lifecycle Diagram

```
init/bootstrap
  -> labels/traces (harvest + async-route-pg)
    -> train-route-model (route_model.npz)
      -> serve loads model
        -> runtime uses learned route_fn
          -> loop/dream retrains
```

## Configuration Knobs

**Routing mode**
- `--route-mode learned|edge|edge+sim|off` (daemon/serve/clients)
- `OPENCLAWBRAIN_ROUTE_MODE` (profile env override)
- `policy.route_mode` in profile JSON

**Learned enforcement (high-trust operators)**
- `--assert-learned` (daemon/serve/socket client)
- `OPENCLAWBRAIN_ROUTE_ASSERT_LEARNED` (profile env override)
- `policy.assert_learned` in profile JSON
- Query param: `assert_learned=true` (socket client/query calls)

**Stop action**
- `--route-enable-stop true|false`
- `--route-stop-margin <float>`
- `OPENCLAWBRAIN_ROUTE_ENABLE_STOP`, `OPENCLAWBRAIN_ROUTE_STOP_MARGIN`
- `policy.route_enable_stop`, `policy.route_stop_margin` in profile JSON

**Route model path**
- Default: `route_model.npz` next to `state.json`
- Override: `--route-model /path/to/route_model.npz`

## Failure Modes and Detection

**Missing route model**
- Symptom: learned configured, effective mode falls back to `edge+sim`.
- Detection: daemon health shows `route_model_present=false`, `route_mode_effective=edge+sim`, `route_model_error=missing`.
- Operator action: run `async-route-pg` and `train-route-model`, or rebuild via loop/build-all.

**Route model load failure**
- Symptom: route_model exists but cannot be loaded, fallback to `edge+sim`.
- Detection: daemon health `route_model_error=load_failed: ...`, report warning, `route-audit` shows `route_model_loaded=false`.
- Operator action: regenerate route_model from traces/labels, verify file permissions/format.

**Embedding dimension mismatch**
- Symptom: daemon query errors with embedding dimension mismatch.
- Detection: query error: `expected <dim>, got <dim>`.
- Operator action: rebuild embeddings or reembed to match state metadata.

**Stale route model**
- Symptom: labels/traces updated recently but route_model timestamp lags far behind.
- Detection: `route-audit` shows `last_train_iso` older than recent labels file mtime.
- Operator action: retrain route model (loop or `train-route-model`).

**No labels**
- Symptom: `labels.jsonl` empty or missing; training has no new supervision.
- Detection: `route-audit` shows `labels_count=0` or file missing.
- Operator action: ensure `harvest` or `async-route-pg` runs with labels output.

**State lock contention**
- Symptom: training/maintenance steps skip due to lock held.
- Detection: build-all/loop logs show state lock timeout or skipped steps.
- Operator action: stop conflicting writer or run with `--force` (expert use).

**Teacher disabled/unavailable**
- Symptom: `async-route-pg` reports `teacher_available=false`, no labels created.
- Detection: `async-route-pg --json` output errors like `OPENAI_API_KEY not set` or `teacher disabled`.
- Operator action: enable teacher, configure model/key, or rely on self/human labels.

## Operator Checklist

**1) Verify effective routing mode**
```
openclawbrain route-audit --state ~/.openclawbrain/main/state.json
```
Expected (healthy learned mode):
- `route_mode_configured: learned`
- `route_mode_effective: learned`
- `route_model_present: True`
- `route_model_loaded: True`

**2) Check daemon health (includes routing fields)**
```
openclawbrain health --state ~/.openclawbrain/main/state.json
```
Expected: payload includes `route_model_present`, `route_mode_configured`, `route_mode_effective`.

**3) Check report warnings**
```
openclawbrain report --state ~/.openclawbrain/main/state.json
```
Expected: no warnings about degraded learned routing.

**4) Ensure labels are being written**
```
wc -l ~/.openclawbrain/main/labels.jsonl
```
Expected: non-zero once supervision has been collected.

**5) Run teacher loop (optional but recommended)**
```
openclawbrain async-route-pg --state ~/.openclawbrain/main/state.json --apply --json
```
Expected: `decision_points_labeled > 0` when teacher is available and there is query history.

**6) Train route model**
```
openclawbrain train-route-model \
  --state ~/.openclawbrain/main/state.json \
  --traces-in ~/.openclawbrain/main/route_traces.jsonl \
  --labels-in ~/.openclawbrain/main/labels.jsonl \
  --out ~/.openclawbrain/main/route_model.npz
```
Expected: non-zero `points_used` and a new `route_model.npz` mtime.

**7) Enforce learned routing (high-trust)**
```
openclawbrain serve start --state ~/.openclawbrain/main/state.json --assert-learned
```
Expected: daemon refuses to start if effective mode is not `learned`.
