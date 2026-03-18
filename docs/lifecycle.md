# Canonical lifecycle and compatibility migration

This file is the repo's convergence decision for the current public package surfaces.

## Decision

- Canonical operator lane for the `0.3.5` wave: `@openclawbrain/openclaw@0.3.5` plus the `openclawbrain` CLI.
- Canonical package installs now have two truthful hook layouts for the same front-door package:
  - CLI-managed generated shadow extension: `npm install -g @openclawbrain/openclaw@0.3.5` then `openclawbrain install --openclaw-home <path>`
  - native package plugin: keep the `openclawbrain` CLI available, run `openclaw plugins install @openclawbrain/openclaw@0.3.5`, then run `openclawbrain install --openclaw-home <path>` to pin the activation root
- Compatibility lane for older installs: `@jonathangu/openclawbrain@0.3.5` through `openclaw plugins install`.
- New docs, reinstall guides, upgrade guides, and support replies should lead with the canonical lane.
- The compatibility lane stays published so older plugin/wrapper installs do not break, but it is not the main operator story.

## Copy-paste lifecycle

```bash
# Install
npm install -g @openclawbrain/openclaw@0.3.5
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed

# Or keep the CLI available, let OpenClaw own the package install, then pin the activation root
openclaw plugins install @openclawbrain/openclaw@0.3.5
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed

# Upgrade
npm install -g @openclawbrain/openclaw@0.3.5
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed

# Verify
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain status --openclaw-home ~/.openclaw --json

# Detach and keep data
openclawbrain detach --openclaw-home ~/.openclaw
openclaw gateway restart

# Uninstall and keep data
openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart

# Uninstall and purge data
openclawbrain uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
npm uninstall -g @openclawbrain/openclaw
```

Semantics:

- `detach` removes only the OpenClaw profile hook and always keeps OpenClawBrain data.
- `uninstall --keep-data` removes the hook and leaves activation data behind.
- `uninstall --purge-data` removes the hook and deletes activation data for that install.
- `status --openclaw-home <path>` now reports whether that hook is a generated shadow extension or a native package plugin install.
- `openclaw gateway restart` is the truthful post-change step after install, detach, or uninstall when you want the running profile to pick up the new hook state immediately.

## Compatibility path

Older plugin/wrapper installs can stay on the compatibility package:

```bash
openclaw plugins install @jonathangu/openclawbrain@0.3.5
```

Treat that as a holdover lane, not the primary install story.

## Migration path

For operators:

1. Keep the existing compatibility install until you have a maintenance window.
2. Choose the canonical package lane you want OpenClaw to load:
   - CLI-managed shadow hook: `npm install -g @openclawbrain/openclaw@0.3.5`
   - native package plugin: keep the CLI available, then run `openclaw plugins install @openclawbrain/openclaw@0.3.5`
3. Pin or repair the target OpenClaw home with `openclawbrain install --openclaw-home ~/.openclaw`.
4. Restart the gateway: `openclaw gateway restart`.
5. Verify with `openclawbrain status --openclaw-home ~/.openclaw --detailed`.
6. Only after the canonical lane is verified should you remove any older compatibility-package wiring through your existing OpenClaw plugin management flow.

For maintainers:

1. Lead README/package/release surfaces with `@openclawbrain/openclaw@0.3.5`.
2. Label `@jonathangu/openclawbrain@0.3.5` as compatibility-only in package metadata and docs.
3. Keep `openclawbrain install` as the activation-root pinning step for both canonical layouts until the host can infer that boundary without the CLI.
