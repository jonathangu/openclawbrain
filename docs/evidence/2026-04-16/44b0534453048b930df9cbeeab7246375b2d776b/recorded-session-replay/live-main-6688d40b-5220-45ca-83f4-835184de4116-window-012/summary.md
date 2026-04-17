# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3f29e706c8408e5460da5ae181547f400604bd45efe4b812bde36a617f82f5`
- fixture hash: `sha256-3bf32dcbf845b428f375103144110fdafde5982202bc1871fff67136d9720e81`
- score hash: `sha256-5773e48e65b2f935bfd4976c2c933d798d15377c512b648c0c0b98dadc392eb9`
- bundle hash: `sha256-6abdfb1af77335adbe11df62e2db973e5fee0ec62ca81b956bbca3452bcdba76`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-865ec1deeaa418471d6bb216a38e6bca377292e05c38cf14fb63e270894197b5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-41ea8055ef39f81c25b837a030147d3b0b39eb01df53d79358621de260f4ed2d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5ad5f0a1d986c7cd9516c77cdfca5a6e13be042d3b6b644426d9a71b073b1e0e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-755ce4045f3898f1addd027d820849c5d86e9713082b1a4df06076cfbe62c6d0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6209582a | sha256-5d9268f938188c682f5caa3156edb60999c9533e8dff6535e34403e11570500b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6209582a | sha256-41740f92f2a9ad1d5ed97b1d2599fa9f4a51490e677ff2e014e1b4bfdec10146 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0810ab81 | sha256-91c2137b7ab9fe6cae5ce5c9b8341b0f465c6e7c4e9f9f0c06c3291517bcbbf1 |
