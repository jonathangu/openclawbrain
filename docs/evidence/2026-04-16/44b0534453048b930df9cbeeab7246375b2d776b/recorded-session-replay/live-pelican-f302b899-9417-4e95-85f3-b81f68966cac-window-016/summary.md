# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60359ed78d5b78e9d115bf8cb9e9ba270e0f90bac409bf6884d4a443b2440f94`
- fixture hash: `sha256-0da91a494c8a34b6c27eb293958b781dbe6bc334337372f9fbd368fd3d0ee08d`
- score hash: `sha256-1d7470d65fdcc5734fe8cd4130be24242ede1bc30469efc8647d27044d0ddb8f`
- bundle hash: `sha256-fdcce6864a96766a901a34704c9d48df5a30fe2c7a82ca8754da5e1b3ad0bf99`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c77331726a9326f500ec3f7c3dbbaeae387d368e17255232ecaec7597f897fed |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e38e5841701b5512064debac50d257f275cf4f7e5dcefbc78eb0a38bd97bbbb8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca6a937f3c2c1198497e2ca050a33450ff10c497fea17d7d5af7d7ad7885b29b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a0ebf420189621ea84d52d9dafc66b39f224c463bc69bf4fa61d70ffe7ff0be8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec56c6b8 | sha256-0d4b48596daeda6b6d42f439733a9272c583d4941eb034f7407a4835546f2e37 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ec56c6b8 | sha256-400a80c22241f3c021f338ba879554416005ad5e1ce82308f7fdce673687a455 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-012ce6d1 | sha256-af9beee0ab8f852aa1cf424916561e8fa33595dab3f64b2118c717c5d2be462c |
