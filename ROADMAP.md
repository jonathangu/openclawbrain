# ROADMAP

## Open Work

### Completed
- [DONE] Real-time correction flow with fired-node logging and dedup.
- [DONE] OpenAI embedding and LLM integration (`--embedder openai`, `--llm openai`).
- [DONE] CLI DX improvements (`query` text output includes node IDs, `learn --json` summary, `health` readable output).
- [DONE] Hash-embedder inject auto-detection with `connect_min_sim` defaulting to 0.0.
- [DONE] TEACHING/DIRECTIVE injection types documented and fully supported.
- [DONE] Live injection primitives (`inject_node`, `inject_correction`, `inject_batch`) and `crabpath inject` CLI command.
- [DONE] Correction propagation through inhibitory edges for teaching/correction workflows.

### Near-term
- Review if the following are now complete:
  - Chunked/binary storage for graphs with >10K nodes (currently persisted as a single JSON payload).
  - Streaming/incremental traversal via `traverse_stream` (generator-based API for large graphs).
  - CI synchronization checks to ensure `SKILL.md` stays consistent between GitHub and ClawHub.

### Platform extensions
- Add a custom `VectorIndex` backend callback path (HNSW/FAISS-compatible providers).
- Add multi-workspace federation support for querying across multiple brains.
