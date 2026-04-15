# Recorded Session Replay Proof Bundle

- trace id: `trace-seed-carry-forward-eval-dedup`
- winner mode: `learned_route`
- trace hash: `sha256-afe79e684511872cd52b734532e20cfb688f02de0364132f171ad94e390921db`
- fixture hash: `sha256-884b77f64eaa55d9f2a409b7b1218f0bf4d168526afc66b534b49f49d5f4b795`
- score hash: `sha256-c60ee6bb15eadb317e390ee75799cba46ed25e397cf9e96c1d64ca0c1031b2ae`
- bundle hash: `sha256-bc7e3dd58582e7b79082a1a993b4e4db10e22e4e670fec926e58d0af12a26db2`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | graph_prior_only | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 8/16
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-d3ebda202ab0b33e75a0a4712c3568a0bbd9e431cce03b3d55dc6620c9a7a753 |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-60f95bdfeb93c063099d58979390130a723d0afe70fc6bfb52546fba0e35b19f |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-61ec9e63eaea5506ca240acb4e2e257c0d3a47e5e5a59f00e4bfa9b2af94fdba |
| learned_route | 3 | 3 | 4/4 | 2 | 2 | 3 | 2 | 0 | sha256-be6334c0187617bd6f2cb920d7d8f8ed699a71c3f2da365f8bda44a0065ceba5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | story-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | story-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | story-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | story-turn-1 | 100 | yes | 1/1 | no | no | pack-2ad23a2d | sha256-c34233338775e7a99b0e60f651ea7540ffaa03b0e79f5644c12f865bd8f6bdab |
| vector_only | story-turn-2 | 100 | yes | 1/1 | no | no | pack-2ad23a2d | sha256-648f76fde455ab5036a20e9c0a224bbf36cdd96ae4fdd6f004b12e1154614540 |
| vector_only | story-turn-3 | 40 | yes | 0/2 | no | no | pack-2ad23a2d | sha256-c34233338775e7a99b0e60f651ea7540ffaa03b0e79f5644c12f865bd8f6bdab |
| graph_prior_only | story-turn-1 | 100 | yes | 1/1 | no | no | pack-2ad23a2d | sha256-c34233338775e7a99b0e60f651ea7540ffaa03b0e79f5644c12f865bd8f6bdab |
| graph_prior_only | story-turn-2 | 100 | yes | 1/1 | no | no | pack-2ad23a2d | sha256-648f76fde455ab5036a20e9c0a224bbf36cdd96ae4fdd6f004b12e1154614540 |
| graph_prior_only | story-turn-3 | 40 | yes | 0/2 | no | no | pack-2ad23a2d | sha256-c34233338775e7a99b0e60f651ea7540ffaa03b0e79f5644c12f865bd8f6bdab |
| learned_route | story-turn-1 | 100 | yes | 1/1 | no | yes | pack-2ad23a2d | sha256-c34233338775e7a99b0e60f651ea7540ffaa03b0e79f5644c12f865bd8f6bdab |
| learned_route | story-turn-2 | 100 | yes | 1/1 | yes | yes | pack-7f4116ee | sha256-0639ea20f40a8895b0a97e865ef9e014b82319c9697d2b83dbbd2b9b35457441 |
| learned_route | story-turn-3 | 100 | yes | 2/2 | yes | no | pack-f2f52cd6 | sha256-bd16754261983de4acc2769a7ed22216d1a359ef86d92e99c80b5d8d12f0209a |
