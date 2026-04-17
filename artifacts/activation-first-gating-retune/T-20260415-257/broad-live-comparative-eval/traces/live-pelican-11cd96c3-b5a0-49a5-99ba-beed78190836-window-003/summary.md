# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ec605c7758b471d35c95979aeb2cdfe7a4674e948b05ffbdd6046eabf723431`
- fixture hash: `sha256-076f85a33a3de7d14b01739ce6654a252ec79b49aa247d0b8cb77da6c5a8a9ec`
- score hash: `sha256-b6f4df99c0561483c56be4b88fb48d000b2917ef64f8a3c6da4cd06646b30a45`
- bundle hash: `sha256-94464249fd3fe9d6aa5e749cd56bbf545d477de0f846b21e1a396ad91d262186`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b4e2e11e992d3b83a5df7a249ce0dd37bdac79f45db7926d41f83ea82d964f78 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1c8aaae2b5080b733599a3856431d8cd0a955a4096588e6a1d8b52d9fe59641a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-370c913e45006d1a0c39fadc59aa18b00578cc9c859c8f0a032add926ab6d290 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-d070721491404c765c95b7b19f09cce962678ba60102b8de4835b23e88e3a1ba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-764e6b27 | sha256-20bf307e4ea815e131b7d32d517b7897760772e1bc2f48513c6c13253920af32 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-764e6b27 | sha256-fa5c4b7b0bda68f35b54bc81b0c1f60b46b29848cbe7042209905290ae0d7025 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-c723a4d6 | sha256-18c7b3b5db5fe5e48bf23435e07ad8628dfe520e1ea94cac69975adab3ae5e2c |
