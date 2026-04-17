# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-070`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d996931dbdf0fc6eb1a041ef94d9b0afac8583cd9d36374bfb1b580b3ae115d4`
- fixture hash: `sha256-88843dbca5a068f5c1ecc181f00d1fb7032df4d94e84a695ddbe0eb2f4ef844a`
- score hash: `sha256-9debe3ac15b15e511b2878dba994be189c20878ef867eb9f52815cada39b3a33`
- bundle hash: `sha256-b62dfb69c21eded3069cc3ea6a7d89bd5b62ab6446bc01d0e376addbd6adf74f`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b985231f563afd8790d84f55159160a675cf549c23dad2a3570253340699dd26 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3a2962650dbc77d1c3a4d9989e60cd0eacc4a3fe045c9c964754e53fecdbc8c8 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-53273c5d5550887b892c7ff3645ecb608c222ff42aa01aceef4778827536657b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-93a8dd1d0000bdf89ff6c1b59156818fc329bbc7bb939d851918b2130b6f37e1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3865fa20 | sha256-a1fd16b8ad3989d90e63a827cfb54535467cc252130cfd3a5cb8b6abdbfc84c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3865fa20 | sha256-3f882f78d259a660ee488d21a53814ca8bd52e2f772d88807b43381acd492d67 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-ccf30cfd | sha256-b72cb3d6e16d2085332582e617ece17d815e5da12b4036950e005a2897f251f1 |
