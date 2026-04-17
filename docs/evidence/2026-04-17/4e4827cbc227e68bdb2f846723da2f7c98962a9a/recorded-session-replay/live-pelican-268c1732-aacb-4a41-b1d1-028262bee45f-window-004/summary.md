# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-380e12e9dd757771937f4748557c11b50a1f9a231591dd724ca65839af3ce6a8`
- fixture hash: `sha256-86ffcadee00971f5c46315d2afa19ae2e85e45bae4dad0e458c42f57f711f9d0`
- score hash: `sha256-174f781c5638af0285824d8444b5dc48733bc5063fea684148782eab096cc473`
- bundle hash: `sha256-655dfd58b3469891dd6f8bf07f1df4e8384d5d1a450cfaac02b1ef8793ee6fa4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0928afcbb7b85c41b2a1d624e920cbdedd75575cc8baa6c3ef5218e9d291b99a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-433c9ff116ff46346ba574a352bcd2ce67ef5c7591bdb858d058d758874851b7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-45f0b05df0d385d573c52f1a37b2843ba6180b67b2537ed07762152ae1b3ff5d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-380b8483ac29d05a323f79f8075af14314a2c9f6d7a04f488a007d573ecfc706 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b66b1dc1 | sha256-f3a7c7246caaddff2fcedfb1ee3adb5f8f0a939b3d0f2682edfd6b3ea0b9adf4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b66b1dc1 | sha256-baa37b4cfddd0e7c4d0cd3ab50c70894e1b87cfc7157bb28708edb5235aed5df |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-efdfe35c | sha256-64a08ef7878ee0666143710bc0fde971533137e26f068e5fac571018e82d152e |
