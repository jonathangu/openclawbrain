# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-82b94292f904129190996d09645352442519cd34f4a6fe4ddc3d8ccfdc15ed4f`
- fixture hash: `sha256-2b7971a9291be722d620678727dc2afe570e5b9dc9a97d0983cbb8375a8b4f0f`
- score hash: `sha256-e9bc507291ce25532768ad499121e45ca4820e733f93f617c6e414f94cacdf44`
- bundle hash: `sha256-4463e4e05da93bb0f77cd4120c2ea9f47f4abd8615f0085b24577ada5795de21`

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
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2126389181abd46124f339c97d016b2e80dbdd1c3f4a30cb14b5104924e09f3e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9ed9ff09a6aeb2c030cf816d5113898bf7860e747d86b9ab29ba490ed8707cbd |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f6f147d533032ebee9c1597f6ea538a402d9cc616c6dc490aa38f2b0a918d4d8 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-74b12f3eb9ca7a6ef6a6da0c79a2a355a6df482c8451ba8856ff67caeef85390 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3c1fb0da | sha256-65560f20cbda5cbcb6cbca7084e30b8968196aa1a1eb4823c6c15d0e71bc89c8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3c1fb0da | sha256-ebf31bf50caac8755b589549f253b14891424461d761798fcea97ee87c63fe7e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d0f2bab7 | sha256-e6dbad2060522cc4f1d92eb026431f1164deac3b9e496fcbf6089630ac422007 |
