# OpenClawBrain 0.4.34

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.34`
- `@openclawbrain/cli@0.4.34`

## Why this release exists

`0.4.34` fixes a real install/self-heal bug from `0.4.33`.

The bug was narrow but bad: after activation-root self-heal, the installed hook could be rewritten into invalid JavaScript and fail to load on the real gateway surface.

This release hardens that install lane so the generated hook stays syntactically safe and repeated self-heal/repin paths stop corrupting the loader.

## What changed

- fixes the activation-root repin/self-heal path so it rewrites only the real `ACTIVATION_ROOT` constant line instead of accidentally mutating the helper regex line
- hardens shadow-extension generation to fail closed if the patchable activation-root template line is missing
- keeps repeated register/self-heal calls from corrupting the installed hook source in one process
- pins plugin-manager converge operations to the selected `OPENCLAW_HOME` target instead of relying on ambient default-home state
- closes the exact install bug tracked in issue `#19`

## Operator truth

This is an install-hardening patch release.

It does **not** change the product story from `0.4.33`.
It makes the existing public install story safer and more reliable.

The known cosmetic plugin-id mismatch warning is not the point of this release and may still appear on some hosts. The critical fix here is that install/self-heal should no longer write a broken hook.

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If you operate multiple homes on one machine, keep install, status, proof, and rollback on the same chosen `--openclaw-home` path instead of mixing the shared default home with a staging home.
