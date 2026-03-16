# Agent tools

OpenClawBrain exposes two different tool families:

1. **LCM recall tools** — use these to recover compacted conversation history
2. **Brain runtime tools** — use these to teach or inspect the live learned-routing layer

That distinction matters. A lot of confusion comes from treating them like one system.

## 1. LCM recall tools

Use these when you are trying to remember what happened in a conversation.

### Default escalation pattern
1. `lcm_grep` — find relevant summaries or messages
2. `lcm_describe` — inspect a specific summary or large file cheaply
3. `lcm_expand_query` — recover compressed detail through bounded expansion

Use the cheapest tool that answers the question.

### `lcm_grep`
Search messages and/or summaries by regex or full-text.

Use it when:
- you need to find where a topic came up
- you need candidate summary IDs before expanding
- you do not yet know whether the detail lives in raw messages or summaries

### `lcm_describe`
Read the content and metadata for a summary or stored file.

Use it when:
- you already have a summary ID
- you want the exact text of a compressed summary
- you want to inspect a stored large file without opening everything else

### `lcm_expand_query`
Answer a focused question by expanding summary DAG context through a bounded delegated sub-agent.

Use it when:
- the summary is too compressed to trust for specifics
- you need a precise answer with citations to summary IDs
- you are answering questions about prior decisions, dates, commands, or exact claims

### `lcm_expand`
Low-level DAG expansion tool used by the delegated expansion path.

Use it directly only when you know you need manual DAG traversal rather than the usual query flow.

## 2. Brain runtime tools

Use these when you are working with the learned routing layer itself, not when you are doing ordinary memory recall.

### `brain_teach`
Teach the brain a correction or reusable guidance.

Current truth:
- immediate retrieval is wired into the runtime
- taught nodes are embedded immediately when embeddings are configured
- taught corrections bind most strongly when invoked from a live tool session on the active conversation
- deterministic session-bound proof exists even though raw prompt-driven host proof is not the release boundary

Primary code:
- `src/brain-runtime/tools.ts`
- `src/brain-runtime/service.ts`
- `test/brain-runtime/service.test.ts`

### `brain_status`
Inspect current operator/runtime truth for the learned layer.

Typical surfaces include:
- enabled/disabled state
- embedding configuration truth
- worker mode / PID / heartbeat / health
- current promoted pack metadata
- last assembly decision
- graph and health counters

Primary code:
- `src/brain-runtime/tools.ts`
- `src/brain-runtime/service.ts`
- `src/brain-cli.ts`

### `brain_trace`
Inspect the most recent trace or a specific trace ID.

Typical surfaces include:
- chosen seed
- seed scores
- fired nodes
- pack version
- route summary/footer
- episode linkage

Primary code:
- `src/brain-runtime/tools.ts`
- `src/brain-runtime/service.ts`
- `src/brain-core/trace.ts`

## When not to use brain tools

Do **not** reach for brain tools just because you need memory.

If the question is really:
- "what happened earlier?"
- "what decision did we already make?"
- "what was the command / date / file path?"

…then start with **LCM recall tools**, not `brain_teach`, `brain_status`, or `brain_trace`.

Brain tools are for:
- teaching the learned layer
- auditing runtime/operator truth
- inspecting a live routing decision

They are **not** the general-purpose recall interface.

## Prompting guidance for agents

A good agent prompt should make the split explicit:

```markdown
## Memory and routing tools

For recall from compacted history:
- lcm_grep
- lcm_describe
- lcm_expand_query

For the live learned routing layer:
- brain_teach
- brain_status
- brain_trace

Use LCM tools for recall.
Use brain tools only when teaching or auditing the routing layer itself.
```
