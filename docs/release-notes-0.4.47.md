# OpenClawBrain 0.4.47

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.47`
- `@openclawbrain/cli@0.4.47`

## Why this release exists

`0.4.47` publishes the post-`0.4.46` explicit-preference precedence fix.

The release is intentionally narrow: when the user updates a durable preference, the newer explicit preference should become the current truth and the older versioned preference should stop firing for normal current-truth queries.

## What changed

- treats newer explicit preferences for the same durable subject as replacements rather than co-current siblings
- recognizes explicit rejection of an older preference value as deterministic supersession
- handles versioned tool/model choice updates such as `Use Codex GPT-5.4 first` followed by `Use latest Codex GPT-5.5 first`
- marks the older correction node `superseded` and links it to the newer current node
- excludes the superseded older node from normal current-truth retrieval after the handoff
- documents the correction/routing rule that newer explicit preferences supersede older values in the same durable slot
- adds regression coverage for the Codex model preference handoff

## Operator truth

The public lane is unchanged:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

This release does not introduce a new install front door. The post-publish harvest should recapture the standard operator proof bundle from the installed `0.4.47` packages.

## Honest boundary

This release proves a specific current-preference precedence fix.

It does **not** claim:

- broad memory is solved
- broad online answer quality is proven
- every preference phrasing can be canonicalized perfectly
- live tool execution behavior changed
- the operator proof has already been recaptured for the unpublished package version

## Focused verification

Release-prep verification should include:

- `npx vitest run test/brain-runtime/user-correction-demo.test.ts --reporter=dot`
- `npm run release:verify:docs-drift`
- `npm run release:plan`
- package tarball checks from `npm run release:verify:openclaw` and `npm run release:verify:cli` when publish is being harvested

Repo-wide typecheck may still report pre-existing repo-wide type errors unrelated to this lane; keep the publish decision tied to the release verification contract.

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Keep the claim boundary narrow: this release fixes explicit-preference precedence for current durable choices; it is not a broad-memory release.
