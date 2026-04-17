# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d5c4c3a45a443f8773663b6b718260034f858fe43001100c3e44deaa92dae64`
- fixture hash: `sha256-8ee17d6b70fb97105471476aa616629c3b433fcacd6e10fa09857f62252427e6`
- score hash: `sha256-734485d13d6339b9b52b5d093ea8c4f2ac36e985205ff882be587ff21aa37e61`
- bundle hash: `sha256-edc3009156577a3185c33400529bd22189273be9445f0ae479b0d1245a8f9d62`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-09e55df402125c6b04d503b2df670ff995850f1c31d072adf7d8fb44788c9b43 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cca1dda81c5894a25be2496e7f308a3d89907eb9d27c6e9c41b57c6c9925eab1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-39c8a3a48a71fb277a15eccc2ce9b0b8ee8a08d97b1a0910973532ec75882e29 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-4a48f5ae88a06948e872417a110b99054c5c5e0bb86dab8f733cbd858986367e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3ff80c24 | sha256-9e51b3b450d21a716d468e5d533e844d2136b4a7abc76eaddc6819f141934fb4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3ff80c24 | sha256-b93dc448bfe3c0f530e0528782e2256f761f3bc0813dbd7e178ee6998c227737 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-bb5df279 | sha256-da2bdaa755ad39da610b271ce918a6b43cb5e0c5c6329ebe349efc0e16362f7c |
