# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-42569e83640d83d518ba29ca1a5d1e9b6e4d4199b30f617c4791a1dd88228113`
- fixture hash: `sha256-f48809b81bb4e1254f7d39666aaa728dc07dc2c22f0452eb070a26b2c4c62c7f`
- score hash: `sha256-4da2fd43374f3f71bb78cd7161def0073953102f6c697c7524af788fffb8b20a`
- bundle hash: `sha256-51af48f2d65813830e919554fe72ff51eacde715ce83c413c50c801cbef19eb6`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4151aa2b4529763822ab0b93f374b3f94033c0c22020133d889b1c07c28f1c03 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c39ca4e202f519054004d259e48e53f0d2d0cbd15feea1a71e3f230eb460f875 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-da3a55e4d1eb040e9391e3588f022739b5bb8f43939daff6b4840dbbd0150e7e |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-e2e8fc234ee7ee3bc243c64c31e8975d7d71cce7565258975017091454d9463e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7bfaba4e | sha256-912c3e3f04fc1fdf7ada4a0c6a575d2b220dcc8ce79aa172e81592a6fa5ecee5 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7bfaba4e | sha256-92664e70d17ac2fd66c76e025fa8269883c7915bd2348fc44c0b318a5ad26be4 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-c1efdb33 | sha256-668d6bc50cfa84e0d8b924ff2d8e1866640d5f1f15476e27096fe66d919c4e3e |
