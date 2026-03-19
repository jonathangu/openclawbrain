# Canonical lifecycle and compatibility migration

This file is the repo's convergence decision for the current public package surfaces.

## Decision

- Canonical operator lane for the `0.4.0` wave: install the plugin/runtime payload with `openclaw plugins install @openclawbrain/openclaw@0.4.0`, then run the operator CLI through `npx @openclawbrain/cli@0.4.0 openclawbrain ...`.
- This exact public-registry flow already passed on the real host `redogfood`.
- Compatibility lane for older installs: `@jonathangu/openclawbrain@0.3.5` through `openclaw plugins install`.
- New docs, reinstall guides, upgrade guides, and support replies should lead with the split `0.4.0` lane.
- Current caveat: some hosts still warn about a plugin id mismatch because the manifest uses `openclawbrain` while the package/entry hint uses `openclaw`. Document that warning as expected/currently cosmetic rather than hiding it.
- The compatibility lane stays published so older plugin/wrapper installs do not break, but it is not the main operator story.

## Copy-paste lifecycle

```bash
# Install
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed

# Upgrade or repair
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed

# Verify
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --json

# Detach and keep data
npx @openclawbrain/cli@0.4.0 openclawbrain detach --openclaw-home ~/.openclaw
openclaw gateway restart

# Uninstall and keep data
npx @openclawbrain/cli@0.4.0 openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data
openclaw gateway restart

# Uninstall and purge data
npx @openclawbrain/cli@0.4.0 openclawbrain uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
```

Semantics:

- `npx @openclawbrain/cli@0.4.0 openclawbrain install` pins or repairs the activation root for one OpenClaw home after the plugin payload is installed.
- `detach` removes only the OpenClaw profile hook and always keeps OpenClawBrain data.
- `uninstall --keep-data` removes the hook and leaves activation data behind.
- `uninstall --purge-data` removes the hook and deletes activation data for that install.
- `openclaw gateway restart` is the truthful post-change step after install, detach, or uninstall when you want the running profile to pick up the new hook state immediately.
- The plugin payload itself is managed through OpenClaw's plugin manager. Remove it there separately only if you want the installed package files gone too.

## Compatibility path

Older plugin/wrapper installs can stay on the compatibility package:

```bash
openclaw plugins install @jonathangu/openclawbrain@0.3.5
```

Treat that as a holdover lane, not the primary install story.

## Migration path

For operators:

1. Keep the existing compatibility install until you have a maintenance window.
2. Install the canonical plugin/runtime payload: `openclaw plugins install @openclawbrain/openclaw@0.4.0`.
3. Pin or repair the target OpenClaw home with `npx @openclawbrain/cli@0.4.0 openclawbrain install --openclaw-home ~/.openclaw`.
4. Restart the gateway: `openclaw gateway restart`.
5. Verify with `npx @openclawbrain/cli@0.4.0 openclawbrain status --openclaw-home ~/.openclaw --detailed`.
6. Only after the split lane is verified should you remove any older compatibility-package wiring and any obsolete globally installed combined-package leftovers.

For maintainers:

1. Lead README/package/release surfaces with the split `0.4.0` flow.
2. Label `@jonathangu/openclawbrain@0.3.5` as compatibility-only in package metadata and docs.
3. Keep the current plugin id mismatch warning documented until the manifest/package ids are aligned.
4. Keep the `install` step as the activation-root pinning step until the host can infer that boundary without the CLI.
