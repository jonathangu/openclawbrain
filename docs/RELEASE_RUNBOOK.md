# OpenClawBrain v0.1 ClawHub-first Release Runbook

Status: **ready for final publish execution; replace the existing ClawHub Skill with this Code Plugin after gates pass**
Date: 2026-04-30

## Product rule

OpenClawBrain has one public install path and one public version story:

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
```

GitHub is the source and release record. ClawHub is the install/discovery channel. npm is optional later for registry fallback, JS package reuse, or npm provenance workflows.

Do not ship a separate installer, root `openclawbrain` config key, or context-engine replacement. The existing ClawHub `openclawbrain` Skill is legacy and must be replaced/superseded by this native Code Plugin for the canonical public install path.

## Release blockers

1. **Package metadata must be ClawHub-compatible.** `packages/openclaw-plugin/package.json` must ship built JS and declare both source and runtime entries:

   ```json
   {
     "openclaw": {
       "extensions": ["./src/index.ts"],
       "runtimeExtensions": ["./dist/index.js"],
       "compat": {
         "pluginApi": ">=2026.4.29",
         "minGatewayVersion": "2026.4.29"
       },
       "build": {
         "openclawVersion": "2026.4.29",
         "pluginSdkVersion": "2026.4.29"
       }
     }
   }
   ```

2. **Replace the public ClawHub identity.** A public ClawHub `openclawbrain` Skill already exists (`jonathangu`, latest observed `12.2.1`). Jonathan approved replacing it. The final state must make `clawhub:openclawbrain` resolve to this native Code Plugin package, not the legacy Skill.

   Replacement rule:
   - preserve the old listing state in logs/output before mutation;
   - hide/delete/supersede the legacy Skill as needed;
   - publish this package as the canonical `openclawbrain` Code Plugin;
   - verify package inspect/explore and fresh install resolve to Code Plugin `openclawbrain@0.1.1`.

3. **Publish with GitHub-tag provenance.** Prefer publishing from a GitHub tag. If local-folder publish is required, pass explicit source repo/commit/ref overrides and verify ClawHub records them.

4. **Use strict config commands only.** String config values must be strict JSON strings, or set the full config object atomically.

## Phase 0 — Preflight

```bash
git status --porcelain=v1
git diff --exit-code
git diff --cached --exit-code
git rev-parse HEAD
git log -1 --oneline

clawhub whoami
openclaw version || openclaw --version
node --version
pnpm --version

node -e 'const p=require("./packages/openclaw-plugin/package.json"); console.log(p.name, p.version, p.openclaw)'
test -f packages/openclaw-plugin/openclaw.plugin.json
```

Required ClawHub capability: the CLI must support plugin package workflows (`clawhub package ...`). The older skill-only CLI is not enough for this release path.

Slug/name collision checks with current ClawHub CLI:

```bash
clawhub -V
clawhub package inspect openclawbrain
clawhub package explore openclawbrain
```

`clawhub package publish --dry-run --json` is the authoritative publish preview. Older runbook variants referenced `package info --json` and `search --json`; those commands are not present in ClawHub `0.12.0`.

## Phase 1 — Release gates

```bash
pnpm --dir packages/openclaw-plugin check
pnpm --dir packages/openclaw-plugin build
pnpm test:product
pnpm ocb:traces:production-status
pnpm ocb:e2e:smoke
pnpm ocb:e2e:production
```

Metadata gate:

```bash
node -e '
const p=require("./packages/openclaw-plugin/package.json");
if (p.name !== "openclawbrain") throw new Error("wrong package name");
if (p.version !== "0.1.1") throw new Error("wrong package version");
if (!p.openclaw?.extensions?.includes("./src/index.ts")) throw new Error("missing source extension");
if (!p.openclaw?.runtimeExtensions?.includes("./dist/index.js")) throw new Error("missing runtime extension");
if (!p.openclaw?.compat?.pluginApi) throw new Error("missing plugin API compat");
'
```

Pack and unpack check:

```bash
TARBALL=$(npm pack --workspace packages/openclaw-plugin --pack-destination /tmp | tail -1)
TARBALL="/tmp/$TARBALL"
rm -rf /tmp/ocb-pack-check
mkdir -p /tmp/ocb-pack-check
tar -xf "$TARBALL" -C /tmp/ocb-pack-check
find /tmp/ocb-pack-check -maxdepth 3 -type f | sort
test -f /tmp/ocb-pack-check/package/openclaw.plugin.json
test -f /tmp/ocb-pack-check/package/dist/index.js
```

## Phase 2 — Fresh install dogfood

Normal user path first:

```bash
TMP_HOME=$(mktemp -d)
HOME="$TMP_HOME" openclaw plugins install "$TARBALL"
HOME="$TMP_HOME" openclaw plugins enable openclawbrain
HOME="$TMP_HOME" openclaw plugins inspect openclawbrain --json
HOME="$TMP_HOME" openclaw config validate --json
```

Optional stricter isolation lane:

```bash
TMP_HOME=$(mktemp -d)
HOME="$TMP_HOME" OPENCLAW_DISABLE_BUNDLED_PLUGINS=1 openclaw plugins install "$TARBALL"
```

Do not make the bundled-disabled lane the only fresh-install gate; it is useful for undeclared dependency detection but is not the normal user environment.

## Phase 3 — GitHub source record

```bash
git push origin main
git tag -a openclawbrain-v0.1.1 -m "OpenClawBrain native plugin v0.1.1"
git push origin openclawbrain-v0.1.1
git rev-parse openclawbrain-v0.1.1^{commit}
```

Only tag after the package version gate passes. The package version, Git tag, GitHub release title, and ClawHub version must all agree.

## Phase 4 — ClawHub dry run

Preferred:

```bash
clawhub package publish jonathangu/openclawbrain@openclawbrain-v0.1.1 --dry-run --json
```

If package-root resolution requires a local folder, dry-run with explicit source metadata:

```bash
EXPECTED_SHA=$(git rev-parse openclawbrain-v0.1.1^{commit})
clawhub package publish packages/openclaw-plugin \
  --source-repo jonathangu/openclawbrain \
  --source-commit "$EXPECTED_SHA" \
  --source-ref refs/tags/openclawbrain-v0.1.1 \
  --dry-run \
  --json
```

Dry-run JSON must prove:

- package root = `packages/openclaw-plugin`
- package name/slug = `openclawbrain` (or the chosen collision-safe slug)
- version = `0.1.1`
- format = native OpenClaw plugin
- manifest id = `openclawbrain`
- runtime extension = `./dist/index.js`
- `configSchema` present
- source repo = `jonathangu/openclawbrain`
- source ref = `openclawbrain-v0.1.1`
- source commit = expected SHA
- package files exclude test/cache/local proof data

## Phase 5 — Replace legacy Skill and publish Code Plugin

Archive current listing state first:

```bash
clawhub package inspect openclawbrain
clawhub package explore openclawbrain
```

Then remove the legacy Skill from the canonical slug path if it still resolves there:

```bash
clawhub hide openclawbrain
```

Use `clawhub delete openclawbrain` only if hide does not free/supersede the slug for the Code Plugin publish path. Jonathan approved replacing the old Skill with this native plugin.

Preferred publish:

```bash
clawhub package publish jonathangu/openclawbrain@openclawbrain-v0.1.1 --json
```

Fallback only if dry-run proved exact metadata:

```bash
clawhub package publish packages/openclaw-plugin \
  --source-repo jonathangu/openclawbrain \
  --source-commit "$EXPECTED_SHA" \
  --source-ref refs/tags/openclawbrain-v0.1.1 \
  --json
```

## Phase 6 — Public install verification

```bash
TMP_HOME=$(mktemp -d)
HOME="$TMP_HOME" openclaw plugins install clawhub:openclawbrain
HOME="$TMP_HOME" openclaw plugins enable openclawbrain
HOME="$TMP_HOME" openclaw config validate --json
HOME="$TMP_HOME" openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
HOME="$TMP_HOME" openclaw config validate --json
HOME="$TMP_HOME" openclaw config set plugins.entries.openclawbrain.hooks.allowPromptInjection true --strict-json
HOME="$TMP_HOME" openclaw config validate --json
HOME="$TMP_HOME" openclaw gateway restart
HOME="$TMP_HOME" openclaw gateway status --deep --require-rpc
HOME="$TMP_HOME" openclaw plugins inspect openclawbrain --json
```

Then verify live local routes with the Gateway auth/token path for the temp home:

```text
/plugins/openclawbrain/status
/plugins/openclawbrain/proof?limit=20
```

## Phase 7 — Docs/site copy

Public docs must use strict JSON config syntax:

```bash
openclaw plugins install clawhub:openclawbrain
openclaw plugins enable openclawbrain
openclaw config set plugins.entries.openclawbrain.config.mode '"conservative"' --strict-json
openclaw config validate
openclaw gateway restart
```

This release supersedes the old PyPI/Skill-era OpenClawBrain story. Public docs should call those paths legacy and point normal OpenClaw users to `openclaw plugins install clawhub:openclawbrain`.

## v0.1 npm decision

Do not publish npm for v0.1 unless there is a separate explicit decision. ClawHub-first is enough for the canonical OpenClaw install story once `clawhub:openclawbrain` resolves cleanly to a plugin package.
