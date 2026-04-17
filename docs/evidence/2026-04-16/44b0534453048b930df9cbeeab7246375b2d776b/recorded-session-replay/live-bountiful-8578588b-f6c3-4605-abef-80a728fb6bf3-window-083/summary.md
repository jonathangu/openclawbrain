# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-083`
- winner mode: `graph_prior_only`
- trace hash: `sha256-57ccc92788c2790c0c3abd5be28501a9f0f77e46149a354e0d4dac5d8146bf48`
- fixture hash: `sha256-49b325de5ef5c4cbab453f7084ce8035e6fbd63068087fd5530b66fcb0390183`
- score hash: `sha256-7b22f88bb275fee488146dc2d65bb0a81f1319638c4775e46356f7f5327db0e0`
- bundle hash: `sha256-323ee64808d93211878d0806b808e981fbc49d9708968fe1f7d7c3f3ddb54b00`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b5cfa06ec873c57458d2a0e78a0e3bbb2620ec1455300c12f88247e370c3b4a6 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4af9c11cb684dce0acf8786cd899c902c9c6d1b6f1c788b35fd5082b3b32700b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f958907b27f7222df3abb3dba8da7612863db35a3ca7d4def85cc5901a11d09 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4b7c22c8fa31e011279faee174607111b0a179ab89afb9f86010bc725af1b524 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-be367f16 | sha256-dd44362ba835e905bd84e40cccbf770a136681b2ba043fba65b45e246db2e14c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-be367f16 | sha256-f0023e601424901920c69372fef93be2d2022a7975f365075100466aa8def2d2 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ae15fdb3 | sha256-23ee0c549427be291fef9bd2894bd4b041c94d0df9f1e5b661da60a14e3bab8a |
