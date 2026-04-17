# Recorded Session Replay Proof Bundle

- trace id: `live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1238bb817085e52d5386a747baa6ea8bf61e3a37516af898c3b116b0246d9843`
- fixture hash: `sha256-edd92cf0e628f6e0582722d507204fe8af0abb5e8a70f6ed2001e47aa93a6a45`
- score hash: `sha256-e2873f3457b1d3d91ecf33421cfde5a5344f4fbc4fbad2189f0f3384609b401b`
- bundle hash: `sha256-b7da4e74a8058830c9998aeaba6305e20eed31eb6b7c7215cbff76c0123556d3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2f9bcbabb6e41c0be690a68df09ebb71d4f854521659c85e60ae6817b1b9042 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-977488266b20e876b259e2e334891dbe709a19f647656bbf8a90cc87e06a3125 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3bc11ae048318e8695eceb25c1e224f336b8a47963603236e12993239d32ed1a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3a3932a80139d9e67efdba5b6a88c8018072a9aa8639b8aac73c2b1b03f3b0ce |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f9b373ef | sha256-24841815eebed8fcc40999a3dac6b82f1ac4f88a4396fb780640cdee9400da3a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f9b373ef | sha256-e2456156bc016b023622b0329ee19972af1b8454901bd26231c2c255634da84f |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-18ce650e | sha256-735b7ca7a88251ad4fdd51940e33cb16076c0ada0b00043193566d55e3f92b93 |
