# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c591f5c65a17f8b728581b0a64e58f54d550808c4d6d87e9681919456c4e7956`
- fixture hash: `sha256-7d10d338cf842d955d253c17c711f61a919941b05a2192e292201851e3214a2a`
- score hash: `sha256-1d13581d20f1f7b5e6c3291a7640d4e72223d74f37787b5c890de203290f2d7e`
- bundle hash: `sha256-4a97c9db0a3c1accf02604800615ef34c1574573e6ec302f01ec14ed38ca8c44`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dc716d2fe029608c9fe52ecb8defed0c2de7ebf60cb8d8503f70a55a165b4d33 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-879a0553d641997df661e0a5462935a9ceb2e22b9397e7884deb496a8e81f79d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-785cab51fb08447257099d375557a1ad702fb14916f3d0419ab92f02768df581 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3c10bff59bb81b0804512b634b28eddc1153047f95fc293bc05e750af3a22656 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32c0cf04 | sha256-595875f6682d9f9143e93d4c0e85f29fb77643e36e665ecfb200553951f7f102 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-32c0cf04 | sha256-3c7bbcc99ae30cadb54e89198cb20f2377986a7f52688cd7ec83ec7164f78e94 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9c1661c3 | sha256-b57c539794d2a916dc254b618d3ce2eed023b82189f740f243f7a7af9683db43 |
