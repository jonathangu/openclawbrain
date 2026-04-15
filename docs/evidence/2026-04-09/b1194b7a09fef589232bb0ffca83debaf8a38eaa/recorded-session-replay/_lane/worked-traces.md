# Worked Traces

- traces included: 8/20
- selection rule: highest bundle score spread first, then trace id; turns ordered by per-turn score spread
- source manifest: `canonical-frozen-20` (canonical_recorded_session_trace_set_manifest.v1, 952aff638de8)
- omitted traces: 12 (see _lane/summary-tables.json for the complete table)

## tern-recorded-session-proof

- bundle dir: `tern-recorded-session-proof`
- ranked winner: `graph_prior_only`
- top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 2/2 | 4/4 | 0 | 0 |
| learned_route | 100 | 2/2 | 4/4 | 1 | 0 |
| vector_only | 100 | 2/2 | 4/4 | 0 | 0 |
| no_brain | 0 | 0/2 | 0/4 | 0 | 0 |

### turn-alpha

- user: Where is the restart checklist archived and how are incidents tagged?
- expected phrases: `docs/evidence`, `postmortem IDs`
- feedback kinds: `correction`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | 0dcd96bb635d | Teaching feedback on cli session session-tern-proof-seed: The operator lane restart checklist is archived in docs/evidence and incidents ... |
| learned_route | train | 100 | yes | 2/2 | no | yes | e5301ea165cb | Teaching feedback on cli session session-tern-proof-seed: The operator lane restart checklist is archived in docs/evidence and incidents ... |
| vector_only | eval | 100 | yes | 2/2 | no | no | e5301ea165cb | Teaching feedback on cli session session-tern-proof-seed: The operator lane restart checklist is archived in docs/evidence and incidents ... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

### turn-beta

- user: Summarize the operator lane restart order.
- expected phrases: `operator lane`, `restart order`
- feedback kinds: `teaching`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | 82cec35e2735 | Teaching feedback on cli session session-tern-proof-seed: Keep the operator lane restart order explicit when proving a recorded session r... |
| learned_route | eval | 100 | yes | 2/2 | yes | no | f362ee639b4f | Teaching feedback on cli session session-tern-proof-seed: Keep the operator lane restart order explicit when proving a recorded session r... |
| vector_only | eval | 100 | yes | 2/2 | no | no | 6813a3c9128c | Teaching feedback on cli session session-tern-proof-seed: Keep the operator lane restart order explicit when proving a recorded session r... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

## trace-comparative-replay

- bundle dir: `trace-comparative-replay`
- ranked winner: `graph_prior_only`
- top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 2/2 | 2/2 | 0 | 0 |
| learned_route | 100 | 2/2 | 2/2 | 1 | 0 |
| vector_only | 100 | 2/2 | 2/2 | 0 | 0 |
| no_brain | 0 | 0/2 | 0/2 | 0 | 0 |

### turn-1

- user: show the routing guide
- expected phrases: `routing guide`
- feedback kinds: `teaching`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | no | no | f1b76b33bafb | Teaching feedback on cli session session-comparative-replay-seed: The routing guide lives here.. Related interaction: session-comparative... |
| learned_route | train | 100 | yes | 1/1 | no | yes | f1b76b33bafb | Teaching feedback on cli session session-comparative-replay-seed: The routing guide lives here.. Related interaction: session-comparative... |
| vector_only | eval | 100 | yes | 1/1 | no | no | f1b76b33bafb | Teaching feedback on cli session session-comparative-replay-seed: The routing guide lives here.. Related interaction: session-comparative... |
| no_brain | eval | 0 | no | 0/1 | no | no | none | none |

### turn-2

- user: show the routing guide again
- expected phrases: `routing guide`
- feedback kinds: `approval`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | no | no | f1b76b33bafb | Teaching feedback on cli session session-comparative-replay-seed: The routing guide lives here.. Related interaction: session-comparative... |
| learned_route | eval | 100 | yes | 1/1 | yes | no | 144f886f5446 | Teaching feedback on cli session session-comparative-replay-seed: The routing guide lives here.. Related interaction: session-comparative... |
| vector_only | eval | 100 | yes | 1/1 | no | no | f1b76b33bafb | Teaching feedback on cli session session-comparative-replay-seed: The routing guide lives here.. Related interaction: session-comparative... |
| no_brain | eval | 0 | no | 0/1 | no | no | none | none |

## trace-correction-answer-paths-explicit

- bundle dir: `trace-correction-answer-paths-explicit`
- ranked winner: `graph_prior_only`
- top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 2/2 | 3/3 | 0 | 0 |
| learned_route | 100 | 2/2 | 3/3 | 1 | 0 |
| vector_only | 100 | 2/2 | 3/3 | 0 | 0 |
| no_brain | 0 | 0/2 | 0/3 | 0 | 0 |

### explicit-paths-turn-1

- user: Where is the archive again?
- expected phrases: `docs/evidence`
- feedback kinds: `correction`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | no | no | fbc68c6466f5 | Teaching feedback on cli session session-correction-answer-paths-explicit-seed: When answering archive and incident-tag questions, keep t... |
| learned_route | train | 100 | yes | 1/1 | no | yes | fbc68c6466f5 | Teaching feedback on cli session session-correction-answer-paths-explicit-seed: When answering archive and incident-tag questions, keep t... |
| vector_only | eval | 100 | yes | 1/1 | no | no | fbc68c6466f5 | Teaching feedback on cli session session-correction-answer-paths-explicit-seed: When answering archive and incident-tag questions, keep t... |
| no_brain | eval | 0 | no | 0/1 | no | no | none | none |

### explicit-paths-turn-2

- user: Say it again without dropping the concrete path or incident tag.
- expected phrases: `docs/evidence`, `postmortem IDs`
- feedback kinds: none
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | fbc68c6466f5 | Teaching feedback on cli session session-correction-answer-paths-explicit-seed: When answering archive and incident-tag questions, keep t... |
| learned_route | eval | 100 | yes | 2/2 | yes | no | ec78ef07f14e | Teaching feedback on cli session session-correction-answer-paths-explicit-seed: When answering archive and incident-tag questions, keep t... |
| vector_only | eval | 100 | yes | 2/2 | no | no | fbc68c6466f5 | Teaching feedback on cli session session-correction-answer-paths-explicit-seed: When answering archive and incident-tag questions, keep t... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

## trace-correction-deeper-proof-story

- bundle dir: `trace-correction-deeper-proof-story`
- ranked winner: `learned_route`
- top score modes: `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_route | 100 | 3/3 | 4/4 | 2 | 0 |
| graph_prior_only | 70 | 3/3 | 2/4 | 0 | 0 |
| vector_only | 70 | 3/3 | 2/4 | 0 | 0 |
| no_brain | 0 | 0/3 | 0/4 | 0 | 0 |

### deeper-story-turn-3

- user: Rewrite it correctly now.
- expected phrases: `BRAIN LOADED`, `routeFn available=yes`
- feedback kinds: none
- top modes: `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| learned_route | eval | 100 | yes | 2/2 | yes | no | f969ef8dcb45 | Teaching feedback on telegram session session-correction-deeper-proof-story-seed: OpenClawBrain starts as a correctly attached but mostly... |
| graph_prior_only | eval | 40 | yes | 0/2 | no | no | d9a32542f181 | Teaching feedback on telegram session session-correction-deeper-proof-story-seed: OpenClawBrain starts as a correctly attached but mostly... |
| vector_only | eval | 40 | yes | 0/2 | no | no | d9a32542f181 | Teaching feedback on telegram session session-correction-deeper-proof-story-seed: OpenClawBrain starts as a correctly attached but mostly... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

### deeper-story-turn-1

- user: Use our current brain install to make it concrete.
- expected phrases: `memory scaffold`
- feedback kinds: `teaching`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | no | no | d9a32542f181 | Teaching feedback on telegram session session-correction-deeper-proof-story-seed: OpenClawBrain starts as a correctly attached but mostly... |
| learned_route | train | 100 | yes | 1/1 | no | yes | dfb9dc2a53f4 | Teaching feedback on telegram session session-correction-deeper-proof-story-seed: OpenClawBrain starts as a correctly attached but mostly... |
| vector_only | eval | 100 | yes | 1/1 | no | no | dfb9dc2a53f4 | Teaching feedback on telegram session session-correction-deeper-proof-story-seed: OpenClawBrain starts as a correctly attached but mostly... |
| no_brain | eval | 0 | no | 0/1 | no | no | none | none |

## trace-correction-mode-paths-explicit

- bundle dir: `trace-correction-mode-paths-explicit`
- ranked winner: `graph_prior_only`
- top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 2/2 | 3/3 | 0 | 0 |
| learned_route | 100 | 2/2 | 3/3 | 1 | 0 |
| vector_only | 100 | 2/2 | 3/3 | 0 | 0 |
| no_brain | 0 | 0/2 | 0/3 | 0 | 0 |

### mode-paths-turn-1

- user: Summarize the per-mode outputs.
- expected phrases: `modes/no_brain.json`
- feedback kinds: `correction`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | no | no | e2d63a8d58d8 | Teaching feedback on cli session session-correction-mode-paths-explicit-seed: The proof bundle writes per-mode outputs at modes/no_brain.... |
| learned_route | train | 100 | yes | 1/1 | no | yes | e2d63a8d58d8 | Teaching feedback on cli session session-correction-mode-paths-explicit-seed: The proof bundle writes per-mode outputs at modes/no_brain.... |
| vector_only | eval | 100 | yes | 1/1 | no | no | e2d63a8d58d8 | Teaching feedback on cli session session-correction-mode-paths-explicit-seed: The proof bundle writes per-mode outputs at modes/no_brain.... |
| no_brain | eval | 0 | no | 0/1 | no | no | none | none |

### mode-paths-turn-2

- user: Say it again without dropping the concrete file names.
- expected phrases: `modes/no_brain.json`, `modes/learned_route.json`
- feedback kinds: none
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | e2d63a8d58d8 | Teaching feedback on cli session session-correction-mode-paths-explicit-seed: The proof bundle writes per-mode outputs at modes/no_brain.... |
| learned_route | eval | 100 | yes | 2/2 | yes | no | 298c8349fea8 | Correction feedback on cli session session-correction-mode-paths-explicit: Keep the file paths explicit: modes/no_brain.json and modes/le... |
| vector_only | eval | 100 | yes | 2/2 | no | no | e2d63a8d58d8 | Teaching feedback on cli session session-correction-mode-paths-explicit-seed: The proof bundle writes per-mode outputs at modes/no_brain.... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

## trace-direct-answer-release-verify

- bundle dir: `trace-direct-answer-release-verify`
- ranked winner: `graph_prior_only`
- top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 2/2 | 4/4 | 0 | 0 |
| learned_route | 100 | 2/2 | 4/4 | 1 | 0 |
| vector_only | 100 | 2/2 | 4/4 | 0 | 0 |
| no_brain | 0 | 0/2 | 0/4 | 0 | 0 |

### release-verify-turn-1

- user: What root command runs release verification?
- expected phrases: `npm run release:verify`, `npm test`
- feedback kinds: `teaching`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | ded840d5e33b | Teaching feedback on cli session session-direct-answer-release-verify-seed: The root release verification entrypoint is npm run release:v... |
| learned_route | train | 100 | yes | 2/2 | no | yes | ded840d5e33b | Teaching feedback on cli session session-direct-answer-release-verify-seed: The root release verification entrypoint is npm run release:v... |
| vector_only | eval | 100 | yes | 2/2 | no | no | ded840d5e33b | Teaching feedback on cli session session-direct-answer-release-verify-seed: The root release verification entrypoint is npm run release:v... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

### release-verify-turn-2

- user: Which command is it again?
- expected phrases: `npm run release:verify`, `npm test`
- feedback kinds: none
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | 086b69db9fac | Teaching feedback on cli session session-direct-answer-release-verify-seed: The root release verification entrypoint is npm run release:v... |
| learned_route | eval | 100 | yes | 2/2 | yes | no | 4a61f50d06dc | Teaching feedback on cli session session-direct-answer-release-verify-seed: The root release verification entrypoint is npm run release:v... |
| vector_only | eval | 100 | yes | 2/2 | no | no | 086b69db9fac | Teaching feedback on cli session session-direct-answer-release-verify-seed: The root release verification entrypoint is npm run release:v... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

## trace-direct-answer-reproduce-eval-command

- bundle dir: `trace-direct-answer-reproduce-eval-command`
- ranked winner: `graph_prior_only`
- top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 2/2 | 4/4 | 0 | 0 |
| learned_route | 100 | 2/2 | 4/4 | 1 | 0 |
| vector_only | 100 | 2/2 | 4/4 | 0 | 0 |
| no_brain | 0 | 0/2 | 0/4 | 0 | 0 |

### reproduce-command-turn-1

- user: What command reruns a sanitized trace proof?
- expected phrases: `tsx scripts/validate-recorded-session-replay.ts`, `--trace`
- feedback kinds: `teaching`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | 211a91055b64 | Teaching feedback on cli session session-direct-answer-reproduce-eval-command-seed: Run tsx scripts/validate-recorded-session-replay.ts -... |
| learned_route | train | 100 | yes | 2/2 | no | yes | 211a91055b64 | Teaching feedback on cli session session-direct-answer-reproduce-eval-command-seed: Run tsx scripts/validate-recorded-session-replay.ts -... |
| vector_only | eval | 100 | yes | 2/2 | no | no | 211a91055b64 | Teaching feedback on cli session session-direct-answer-reproduce-eval-command-seed: Run tsx scripts/validate-recorded-session-replay.ts -... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

### reproduce-command-turn-2

- user: What is the proof rerun command again?
- expected phrases: `tsx scripts/validate-recorded-session-replay.ts`, `--trace`
- feedback kinds: none
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 2/2 | no | no | 345ebb8bf8aa | Teaching feedback on cli session session-direct-answer-reproduce-eval-command-seed: Run tsx scripts/validate-recorded-session-replay.ts -... |
| learned_route | eval | 100 | yes | 2/2 | yes | no | 99d7cb3e21f3 | Teaching feedback on cli session session-direct-answer-reproduce-eval-command-seed: Run tsx scripts/validate-recorded-session-replay.ts -... |
| vector_only | eval | 100 | yes | 2/2 | no | no | 345ebb8bf8aa | Teaching feedback on cli session session-direct-answer-reproduce-eval-command-seed: Run tsx scripts/validate-recorded-session-replay.ts -... |
| no_brain | eval | 0 | no | 0/2 | no | no | none | none |

## trace-openclaw-replay-freeze-identity

- bundle dir: `trace-openclaw-replay-freeze-identity`
- ranked winner: `graph_prior_only`
- top score modes: `vector_only`, `graph_prior_only`, `learned_route`
- score spread: 100

| mode | quality | compile ok | phrase hits | promotions | warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph_prior_only | 100 | 3/3 | 3/3 | 0 | 0 |
| learned_route | 100 | 3/3 | 3/3 | 1 | 0 |
| vector_only | 100 | 3/3 | 3/3 | 0 | 0 |
| no_brain | 0 | 0/3 | 0/3 | 0 | 0 |

### turn-1

- user: What should I read before editing?
- expected phrases: `readme before editing`
- feedback kinds: `approval`
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | no | no | 5c28678a8abc | Teaching feedback on chat session session-openclaw-replay-freeze-identity-seed: Always read README before editing code.. Related interact... |
| learned_route | train | 100 | yes | 1/1 | no | yes | 5c28678a8abc | Teaching feedback on chat session session-openclaw-replay-freeze-identity-seed: Always read README before editing code.. Related interact... |
| vector_only | eval | 100 | yes | 1/1 | no | no | 5c28678a8abc | Teaching feedback on chat session session-openclaw-replay-freeze-identity-seed: Always read README before editing code.. Related interact... |
| no_brain | eval | 0 | no | 0/1 | no | no | none | none |

### turn-2

- user: Before changing files, what is the rule?
- expected phrases: `readme before editing`
- feedback kinds: none
- top modes: `vector_only`, `graph_prior_only`, `learned_route` (spread 100)

| mode | phase | quality | compile | phrase hits | learned route | promoted | selection | context preview |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| graph_prior_only | eval | 100 | yes | 1/1 | no | no | d2a7fe080941 | Teaching feedback on chat session session-openclaw-replay-freeze-identity-seed: Always read README before editing code.. Related interact... |
| learned_route | eval | 100 | yes | 1/1 | yes | no | 994d27715193 | Teaching feedback on chat session session-openclaw-replay-freeze-identity-seed: Always read README before editing code.. Related interact... |
| vector_only | eval | 100 | yes | 1/1 | no | no | d2a7fe080941 | Teaching feedback on chat session session-openclaw-replay-freeze-identity-seed: Always read README before editing code.. Related interact... |
| no_brain | eval | 0 | no | 0/1 | no | no | none | none |
