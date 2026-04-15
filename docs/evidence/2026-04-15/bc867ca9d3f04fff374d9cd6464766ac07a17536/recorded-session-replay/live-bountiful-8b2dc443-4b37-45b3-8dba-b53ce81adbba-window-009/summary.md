# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8eced4262f5a642239299c7c899085a7bd53ad7880d03357a10326803fe33aa8`
- fixture hash: `sha256-5aa5748a68c006cb4152d6b9766d43523c43872689382d99e9608f0fedb263a8`
- score hash: `sha256-cb600391abf8b98448ba563fade06a905fcb06ee210c8f1d7b90c123116dfae2`
- bundle hash: `sha256-ba2fbadf76370671874479ecb68d86b91b2359d9b62c82cfb8062d4e18cb272e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86ead80920d9422dc3144931f0210740c8474d5a0351518c55316e7dfbfbffe7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a3f11c1cf8f651f00be304df7522dc8b5d2aeeb2af38c7fb9b52e70ca1a34973 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d34bf54977838edb0765cca1a77a2fb90d5b722c57d13b85f827da04bb68c068 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-2726b089530626d5d82e6830a67fd207d519de8fbf5661b0fdff615d954278fd |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dff9f3ae | sha256-ee24af7f6bd5aea969b3520e7768bb0f8743e3d70a8efe21486e9a9a45497d96 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dff9f3ae | sha256-ab72298e0cc508b02773258d49a561bda5d19eef72cc6d0c5499bce09607a8ec |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-dff9f3ae | sha256-ee24af7f6bd5aea969b3520e7768bb0f8743e3d70a8efe21486e9a9a45497d96 |
