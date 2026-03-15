# Agent tools

OpenClawBrain exposes two tool families:

1. **LCM recall tools** — search and expand compacted conversation history
2. **Brain runtime tools** — teach the live routing layer, inspect status, inspect traces

## LCM recall tools

Use these when you need recall from compacted history.

### Escalation pattern: `lcm_grep` → `lcm_describe` → `lcm_expand_query`

1. **`lcm_grep`** — find relevant summaries or messages by keyword/regex
2. **`lcm_describe`** — inspect a specific summary or file cheaply
3. **`lcm_expand_query`** — deep recall through bounded sub-agent expansion

Start with grep. Expand only when the summary is too compressed for the task.

### `lcm_grep`
Search across messages and/or summaries using regex or full-text search.

### `lcm_describe`
Read the full content and metadata for a summary or stored large file.

### `lcm_expand_query`
Answer a focused question by expanding summaries through the DAG.

### `lcm_expand`
Low-level DAG expansion tool used internally by the delegated expansion sub-agent.

## Brain runtime tools

Use these when working with the learned routing layer directly.

### `brain_teach`
Teach the brain a correction or reusable guidance.

Current truth:
- immediate retrieval is wired into the runtime
- taught nodes are embedded immediately when embeddings are configured
- taught corrections bind most strongly when invoked from a live tool session on the active conversation

Primary code:
- `src/brain-runtime/tools.ts`
- `src/brain-runtime/service.ts`
- `test/brain-runtime/service.test.ts`

### `brain_status`
Inspect operator/runtime truth for the brain layer.

Typical surfaces include:
- enabled / disabled state
- embedding configuration truth
- worker mode / PID / heartbeat / health
- current promoted pack metadata
- last assembly decision
- graph/health counters

Primary code:
- `src/brain-runtime/tools.ts`
- `src/brain-runtime/service.ts`
- `src/brain-cli.ts`

### `brain_trace`
Inspect the most recent trace or a specific trace id.

Typical surfaces include:
- chosen seed
- seed scores
- fired nodes
- pack version
- route footer / summary
- episode linkage

Primary code:
- `src/brain-runtime/tools.ts`
- `src/brain-runtime/service.ts`
- `src/brain-core/trace.ts`

## When to use what

- Need compacted history recall? use **LCM tools**
- Need to teach/update the current routing layer? use **`brain_teach`**
- Need current runtime/operator truth? use **`brain_status`**
- Need to inspect a specific routing decision? use **`brain_trace`**

## Prompting guidance for agents

A good agent prompt should make both tool families explicit:

```markdown
## Memory and routing tools

For recall from compacted history:
- `lcm_grep`
- `lcm_describe`
- `lcm_expand_query`

For the live learned routing layer:
- `brain_teach`
- `brain_status`
- `brain_trace`

Use LCM tools for recall. Use brain tools only when you are teaching or auditing the routing layer itself.
```
