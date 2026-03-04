# Glossary

**Turn (LLM API call)**: One request/response to a model provider, including the prompt payload and the model completion.

**User-visible exchange**: A single user message plus all assistant turns and tool calls that occur before the next user message.

**Cheap turn**: A low-cost LLM call that uses a smaller context budget or cheaper model tier (often for routing, triage, or policy checks).

**Brain-first**: Integration mode where OpenClawBrain injects `[BRAIN_CONTEXT]` automatically at the hook layer before the agent sees the prompt.

**Hot path**: The latency-sensitive, per-message path (query traversal, context assembly, daemon socket) that must stay fast.

**Cold path**: Background or offline work (learning updates, maintenance, pruning, replay, rebuilds) that can run asynchronously.

**QTsim**: Query-time simulation of candidate traversal routes to estimate which edges will fire before committing to a path.

**graph_prior**: The initial edge topology and weights derived from static sources (workspace structure, heuristics, or bootstraps) before learning updates.

**λ confidence mixing**: A weighted blend (lambda coefficient) that combines graph prior strength with runtime confidence signals when scoring routes.

**Inhibitory suppression**: Negative-weight edges that actively reduce activation of a target node to prevent recurring wrong-path retrieval.
