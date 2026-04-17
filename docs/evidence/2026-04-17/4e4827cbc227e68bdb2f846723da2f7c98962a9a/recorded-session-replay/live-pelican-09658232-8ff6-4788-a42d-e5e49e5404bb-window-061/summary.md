# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70ab1ae3977bf5b8f105672a2af7f511f5d5e8eab54227af0f4c11c32810b91e`
- fixture hash: `sha256-f43d8483c3b4eb473890c9d4aad38b8eb4a81081d719d9c58fd2752db7997c33`
- score hash: `sha256-cbf12339a5bf26d261735786527a65a434b18b404604d3b8824d016896cb13ba`
- bundle hash: `sha256-ff270cb5740588fdf617fbff152967778db0dc20241feb3d4e359ab4e7137701`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0af8593686f4dcd1625a4737259415fed87f48af0fee073ee2e87cde2bfd51e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-25f307a78b9e637629e7596760ffc377c9d6c720a723e06269d2b5063f07f219 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd5a7f776a884dfb4411257b04952fd020fd3307d6e875e3c47eda477dd98a36 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d431f5ca8b3a6b7fc0f458cfad1da7e3ccb26ba39b79d1cdf74dd7176503df81 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f3c9c7ee | sha256-6a2d92fb1446d3027516b207f36dc1a61d5521fd5f9e6219a1ee7e29dd931661 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f3c9c7ee | sha256-2b66fcd56d1369c4c0021c9fdf0684f0460fa40b9845f7e2871b49d8e48248f1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-1fb4005d | sha256-b9ac7b8180f7d77915e93fb1fc56378b4ebf24eb00585b68fba81eeea709741f |
