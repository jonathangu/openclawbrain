# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-42569e83640d83d518ba29ca1a5d1e9b6e4d4199b30f617c4791a1dd88228113`
- fixture hash: `sha256-f48809b81bb4e1254f7d39666aaa728dc07dc2c22f0452eb070a26b2c4c62c7f`
- score hash: `sha256-fc92de29a602930220817cf50465d01854f6af72cddb74719b846c92c28fcb30`
- bundle hash: `sha256-483d4d9830e4adf2aa6af29ae8dc6a5e6db65470580a95a9b4c53fd6851a8261`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-944e49a6ed978e7c8b928acc153b9900e419b723f6f5449d1c9c6b3b10266940 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-fd1424b7c5fb6acb560fcc5e0f27e2395559828895b6c33ab22479619033ceee |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-b37f464adb5fec22180fe5e1d66d8105a54df3d2f9f112c893a1aa725a06dcf4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-c1efdb33 | sha256-668d6bc50cfa84e0d8b924ff2d8e1866640d5f1f15476e27096fe66d919c4e3e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-c1efdb33 | sha256-9f5cb1e515adad1a4b30698f5237efc1889f714609b06330b044e42bf849187e |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-c1efdb33 | sha256-297bb812175a402214036c0328673bd712b45d6d6899e15763488b28594ac106 |
