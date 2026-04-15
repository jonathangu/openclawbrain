# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1fa535bc6d12bb71f464c3299ac8be0aea3ff43ab51a534634c743c27f7f228f`
- fixture hash: `sha256-3d8b02b35686d528b31efaf88c8dfba1c6ab218b6a7b654b6357f35e5f82ffce`
- score hash: `sha256-7528d4f2eb215990643ae7b8013b6da0378b5ee94d0a8028437ffb045964e30b`
- bundle hash: `sha256-60ae8b8da37bbd96083d5333b40c8eee2b72fc155e2778c68be0b2b64796bd78`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94d47cd92404766dbef910ada1cf5f255eef0ca70682ef6fa7b016001adffcb3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-68ed27b8dc8c7b6a66e11f28c47e1ddcfd13f92e8070c76cd46f0c52c3dbbd5d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-625fd76ea004936a4f313425155ced530436b74dad316ad3f4a4cd3303adbc90 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2677881f6ed21ad9c727fca5a9831549d30a65f28bacfd2309e488737524ac93 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8c57a1dd | sha256-3058fc82fe63ff9263d0886cdd17a124e383b2b850d11f3b95e9121f079fd3da |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8c57a1dd | sha256-d2ba5d4dd050df20281c54a4f7f9677b29da22c23abfde512395a0a3f3c4b184 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8c57a1dd | sha256-3058fc82fe63ff9263d0886cdd17a124e383b2b850d11f3b95e9121f079fd3da |
