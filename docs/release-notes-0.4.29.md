# OpenClawBrain 0.4.29

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.29`
- `@openclawbrain/cli@0.4.29`

## Why this release exists

`0.4.29` is the follow-on compatibility release for the public Teacher v3 line. The main `0.4.28` release landed the Teacher v3 runtime/proof adoption stack. This release closes the next real host seam: **Gemma 4 31B is now accepted as a first-class compatible local teacher on the canonical install lane**, instead of requiring a manual override.

## What changed

- the local-teacher compatibility list now includes `gemma4:31b` (plus nearby Gemma/Qwen large-model lanes)
- install/runtime autodetect can now select Gemma 4 31B as the local Brain teacher when it is present on Ollama
- the single-version / single-install-path public story remains intact

## Host verification target

After install and gateway restart, the detailed status surface should show the Brain teacher resolved through the canonical provider config, not a stale Qwen-only compatibility path.

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```
