# npm consumer smoke example

This is the smallest copy-paste proof that the current wave's installable packages work from a fresh consumer directory without using the monorepo workspace.

For the current repo-only wave, use locally packed `.tgz` artifacts from `.release/` instead of assuming registry publication has already happened.

## What it proves

- install works with plain `npm` from the locally packed public artifacts
- Node can import the packaged ESM entrypoints directly
- the narrow package split is usable without guessing which package exports what

## Run it in a fresh directory

```bash
pnpm release:pack
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

Expected output is a small JSON object with `"ok": true`, a `runtime_compile.v1` contract id, an event range of `10..11`, and a deterministic export digest.

## After a real npm publish

Once `pnpm release:status` shows a tagged release candidate, the publish lane completes, and post-publish checks pass, you can swap the tarball paths above for registry versions in `npm install`.

## Why these packages

This example intentionally stays small while still touching the real public surface:

- `@openclawbrain/contracts` validates a runtime-compile request
- `@openclawbrain/events` builds normalized interaction and feedback events
- `@openclawbrain/event-export` derives the deterministic export range and provenance from those events
