# npm consumer smoke example

This is the smallest copy-paste proof that the published `@openclawbrain/*` packages work from a fresh consumer directory without using the monorepo workspace.

## What it proves

- registry install works with plain `npm`
- Node can import the published ESM entrypoints directly
- the narrow package split is usable without guessing which package exports what

## Run it in a fresh directory

```bash
tmpdir="$(mktemp -d)"
cp examples/npm-consumer/package.json "$tmpdir/package.json"
cp examples/npm-consumer/smoke.mjs "$tmpdir/smoke.mjs"
cd "$tmpdir"
npm install
npm run smoke
```

Expected output is a small JSON object with `"ok": true`, a `runtime_compile.v1` contract id, an event range of `10..11`, and a deterministic export digest.

## Why these packages

This example intentionally stays small while still touching the real public surface:

- `@openclawbrain/contracts` validates a runtime-compile request
- `@openclawbrain/events` builds normalized interaction and feedback events
- `@openclawbrain/event-export` derives the deterministic export range and provenance from those events
