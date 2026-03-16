# Evidence bundles

Store proof artifacts under dated commit directories:

```text
YYYY-MM-DD/<git-sha>/
```

Each bundle should include the files described in `../EVIDENCE.md`.

For host-install validation runs, also keep the pre-run diagnostic ladder beside the report bundle:

- `status-all.txt`
- `gateway-probe.txt`
- `gateway-status.txt`
- `channels-status.txt`
