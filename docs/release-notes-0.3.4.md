# OpenClawBrain 0.3.4 — Stop teacher pollution from operational scaffolding

Published package: `@jonathangu/openclawbrain@0.3.4`

Git tag: `v0.3.4`

## What shipped

This release closes a real passive-learning integrity bug:
operational scaffolding was being admitted as if it were genuine human teaching.

In practice, that meant OpenClawBrain could misread things like:

- heartbeat prompts
- session-start/reset scaffolding
- message envelope metadata wrappers
- replied-message wrapper blocks

as correction or teaching evidence.

That is bad for two reasons:

1. it pollutes the teacher substrate with non-human operational text
2. it makes operator health/status less trustworthy because the learning surface looks active for the wrong reason

## Fix

0.3.4 adds a system-message exclusion gate at evidence detection time.

Specifically:

- adds a dedicated system-filter module with explicit operational markers
- blocks heartbeat/runtime-envelope/startup scaffolding from entering feedback extraction
- preserves genuine human correction and teaching signals
- adds focused unit + integration coverage for both exclusion and inclusion behavior

## Why this matters

OpenClawBrain’s value depends on learned truth being about real human supervision, not runtime debris.

0.3.4 makes the passive-learning boundary more honest:
- real corrections still flow through
- operational scaffolding stops pretending to be learning signal
- future route-function updates are less likely to drift from fake evidence

## Verification bar

Release verification for this package should include:

- test suite passes
- `npm pack --dry-run`
- heartbeat/startup/metadata examples classify as excluded
- genuine correction/teaching examples still classify as included
