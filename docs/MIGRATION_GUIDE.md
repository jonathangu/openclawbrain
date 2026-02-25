# Migration Guide: workspace files → CrabPath graph

## The problem

Current agent prompts load workspace instruction files (`AGENTS.md`, `MEMORY.md`, `TOOLS.md`, `USER.md`, etc.) into every turn.

These files often contain 30K+ chars.

You pay the full parse cost even when only one section is relevant.

You also keep irrelevant rules, obsolete context, and duplicate statements in the active context every turn.

CrabPath moves this into a retrieval graph:

- split workspace text into typed nodes
- keep only nodes needed for the current task
- learn edge weights from outcomes instead of rereading everything

## The conversion

Treat every workspace heading section as a node.

Each `##` and `###` block becomes one node.

Node type rules:

- headings that contain a rule, constraint, or prohibition become `guardrail`
- headings that encode procedural steps become `procedure`
- headings with explicit tool usage become `tool_call`
- everything else becomes `fact`

For each file, add edges between neighboring sections in the same order.

Create cross-file edges when one section text mentions another section heading.

This is strict string-match at bootstrap time.

## Warm start weights

Every new node starts with `0.5`.

Increase weight for:

- sections that are frequently referenced by other sections
- gates/procedures with high fire counts from historical correction logs

Edges are initialized as:

- `0.6` for same-file neighbor edges
- `0.4` for cross-file heading-reference edges

## Bootstrap script

Use:

```bash
python3 scripts/bootstrap_from_workspace.py /path/to/workspace --output graph.json
```

The script:

- reads all `.md` files under the workspace
- splits by `##`/`###` headings
- classifies each node type by text heuristics
- creates same-file edges at `0.6`
- creates cross-file reference edges at `0.4`
- writes a CrabPath graph JSON
- prints conversion stats

## What stays static

Do not put these into the graph:

- core identity block (the first paragraph of `SOUL.md`)  
- hard safety policy text that never changes by turn
- current user context and session state

Keep those outside graph context and inject separately as hardcoded context.

## Expected results

On large workspaces:

- 32K chars static context shrinks toward 300-500 chars per turn
- token spend reduction in the 95-99% range versus raw workspace replay
- corrected behavior propagates by edge weight updates without manual file rewrites

The practical effect is lower context cost and faster convergence on active workspace policy.
