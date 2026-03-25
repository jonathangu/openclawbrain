# Fail-open design

OpenClawBrain is designed to avoid blocking OpenClaw when the memory layer cannot safely add context.

## What fail-open means

If the runtime cannot compile brain context, the extension returns no extra context and lets OpenClaw continue. The agent still answers. The memory layer steps out of the way.

## Common fail-open cases

| Situation | Runtime behavior | Operator signal |
| --- | --- | --- |
| Activation root is still a placeholder | No brain context is injected | gateway logs `BRAIN NOT YET LOADED` |
| `before_prompt_build` payload is malformed or empty | The extension returns without extra context | warning is logged for the unsupported event shape |
| Active pack is missing or compile fails | The extension does not inject brain context | status can report `fail_open_static_context` or `hard_fail` |

## What the agent sees

The agent keeps running with its normal OpenClaw context. OpenClawBrain does not partially inject broken state.

## What operators should inspect

Start with the standard verify command:

```bash
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

When the extension itself emits warnings, it also appends them to `~/.openclawbrain/extension-errors.log`.

## Recovery

Most recoveries are straightforward:

1. install or repair the OpenClawBrain hook
2. restart the gateway
3. verify the selected OpenClaw home again

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If status still shows a fallback serve path, continue with [Troubleshooting](../operating/troubleshooting.md).
