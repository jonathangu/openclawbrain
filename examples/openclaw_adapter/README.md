# OpenClaw adapter

This adapter is for agent frameworks that manage API keys internally.

The adapter scripts expect `OPENAI_API_KEY` to be present in `os.environ`.
The agent framework injects this key into the process environment before invoking
these scripts, so no key discovery, keychain lookup, or dotfile parsing is
performed.

There is no manual key configuration for these scripts. Provide workspace,
sessions, and output paths and the scripts run end-to-end.

The adapter is the integration layer between the pure CrabPath library and the
framework. It handles:

- Building a workspace graph with `openai-text-embedding-3-small` metadata.
- Persisting `state.json`, `graph.json`, and `index.json`.
- Querying the graph via `query_brain.py`.
- Replaying history and printing health diagnostics in `init_agent_brain.py`.
