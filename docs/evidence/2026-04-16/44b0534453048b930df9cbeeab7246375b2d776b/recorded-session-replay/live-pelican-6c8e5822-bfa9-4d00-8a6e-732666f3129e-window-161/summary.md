# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b83c4e3036cd20627db9b3b691867f0c4cf67798691db96c05537d9efc454f0`
- fixture hash: `sha256-d883ca17da8d181a1200f08513acd619f27d5b75e1c49c4953044231381c83cd`
- score hash: `sha256-ab86415037e5017bb7e805293a355a28d89ec95611b8595de0be2fe61ef372bc`
- bundle hash: `sha256-4e18699d8bb7dbf4db386a24ebc679b588de8b2e58954e7d64aaaad1c74a5824`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5551839106976537fe1c9ce0dbda883b66824b3f67f3049bc2763f475be1647 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f6209316cd6defe87857c04f932b708cc30228a1c51c4d8cd3fb36334174bf62 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a9157390a9fdc505ae16fbf81c974d011dcae887b6405e5bc892d433258f739c |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5a9c5825e02c80b5a2e8e11e18286293278bc281606c047f5cf0e62b68c84ae9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-50ce2f28 | sha256-b6a713c0ac902778761c2d9344bd163deaa15e9ae12aba5e44cda2edc47f1eb3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-50ce2f28 | sha256-54f6b117f088da17337b83d7804c781fad9817094d79d9b142da6928f47ad65c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6281d281 | sha256-4431a1f4790e8a3298f1a89627e5b29202511e10a208ffb2d6651a63a62ed233 |
