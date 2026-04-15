# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-cb6ab1b9-3527-43e5-a3ba-879a338b6120-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f07ed34ccc6a5c54819d12a1e93195c70560e32cc80e0d0e09592b4765b8105`
- fixture hash: `sha256-00d9f388b90351cc79a6666fb1faf09e6f2109bf7c85e8cdc18048263ccb39a6`
- score hash: `sha256-fd90769e55a9ef7159fd8ecf49dc990f7e83a14cacba76f71b93828935978c91`
- bundle hash: `sha256-4d2d1eab7eca46dff180e6d38ee7d9e7668e48dfb0b4475f50438d094bf7b677`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f9e400fcbe43d9ab55b6048a20689714c3c7aae22f85e1babf49f3474335a32 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f08bbac36a23db8a9f6b7688173d50651fd3925c457a8ef7835569c247aea3c2 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cfe3f6cbb39784db7bb3ce25a07bf3bbc3364736f162c11bc3fabb611231bf9b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-5d2054934e110ee418acea805327cc03af8e00463c0f913160ce6619f53556db |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3a12f415 | sha256-87f60ad15ea437bfdc068ff96d7136fb9580eed8299d1fff9c82621af756eb87 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3a12f415 | sha256-b91b36b9a048b26f786e41a7e580db41df64d617384d74476b30675e31a18559 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3a12f415 | sha256-87f60ad15ea437bfdc068ff96d7136fb9580eed8299d1fff9c82621af756eb87 |
