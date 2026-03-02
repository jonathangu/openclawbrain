# OpenClawBrain Operator Quickstart

This is the canonical operator path for running OpenClawBrain in production.

## Canonical entrypoint

Use `openclawbrain serve` for service lifecycle:

```bash
openclawbrain serve start --state ~/.openclawbrain/main/state.json
```

`openclawbrain daemon` is a low-level NDJSON stdio worker used internally by `serve` and custom integrations.

## Socket path convention

Default socket path is:

```text
~/.openclawbrain/<agent>/daemon.sock
```

`<agent>` is derived from the directory containing `state.json`.

Example:
- state: `~/.openclawbrain/main/state.json`
- socket: `~/.openclawbrain/main/daemon.sock`

`serve start` prints the resolved socket path on startup.

## Start

```bash
openclawbrain serve start --state ~/.openclawbrain/main/state.json
```

Optional service templates:

```bash
openclawbrain serve start --state ~/.openclawbrain/main/state.json --launchd
openclawbrain serve start --state ~/.openclawbrain/main/state.json --systemd
```

## Status

`serve status` checks socket existence and performs a health ping over the socket.

```bash
openclawbrain serve status --state ~/.openclawbrain/main/state.json
```

## Query

```bash
python3 -m openclawbrain.socket_client \
  --socket ~/.openclawbrain/main/daemon.sock \
  --method query \
  --params '{"query":"summarize deploy risks","top_k":4}'
```

## Defaults that matter

- `serve start` defaults to `--route-mode learned`.
- `init` writes `route_model.npz` beside `state.json`; if missing/unloadable, query routing falls back to `edge+sim`.
- Override route mode at service start with `--route-mode off|edge|edge+sim|learned`.
- Query embeddings default to `--embed-model auto`:
  - `local:*` state metadata -> local embeddings (fastembed)
  - `hash-v1` state metadata -> hash embeddings (legacy only)
  - OpenAI state metadata -> use `--embed-model openai:<model>` explicitly if you want OpenAI query embeddings

## Stop

```bash
openclawbrain serve stop --state ~/.openclawbrain/main/state.json
```

This sends daemon `shutdown` over the socket. If the socket is unavailable, the command prints exact launchd/systemd stop commands.

## Troubleshooting

- Stale socket (`daemon.sock` exists but `serve status` fails):
  - Restart with `openclawbrain serve start --state ...`.
  - The socket server removes stale sockets on startup.
- State lock errors (`state.json.lock`):
  - Another writer is active (daemon/replay/maintenance).
  - Stop the writer before direct state mutations or use rebuild-then-cutover.
- Logs:
  - launchd/systemd log destinations are configured in the unit/plist.
  - Common defaults are `~/.openclawbrain/<agent>/daemon.stdout.log` and `~/.openclawbrain/<agent>/daemon.stderr.log`.
