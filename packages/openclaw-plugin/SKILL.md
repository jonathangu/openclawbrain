---
name: openclawbrain
description: Native OpenClaw code plugin that stores redacted local memory graph data, exposes proof/search/status routes, and injects bounded context only when explicitly enabled.
---

# OpenClawBrain Plugin Package

This package is a native OpenClaw code plugin, not an instruction-only skill.

It provides a local SQLite memory graph for OpenClaw agents:

- redacted memory nodes and graph edges
- local FTS5 search
- bounded prompt-context augmentation when the OpenClaw plugin hook is enabled by the operator
- proof, graph, learning, search, status, and doctor HTTP routes
- native SQLite/FTS5 self-checks for installed-extension reliability

Safety defaults:

- disabled by default
- raw transcript storage disabled
- prompt-context augmentation requires explicit OpenClaw hook permission
- optional LLM distillation is disabled unless configured by the operator
- plugin state is local to the configured activation root

Operator note: configure, enable, inspect, or remove this plugin through OpenClaw plugin/config commands. This file is package documentation only.
