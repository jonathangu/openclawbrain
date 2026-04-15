# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-020e5fa0ec60c9180b8ca12d4a8cde03c3eaf93efdc6e1249456178218366170`
- fixture hash: `sha256-2fab851b07744bef46921e5dde6e3c44cc707f0e47e7a2b971ff5ea69c88de53`
- score hash: `sha256-384a1d5313f92ed1ef316a7e2109e63a84fe7f52c916af2816d9834773d06781`
- bundle hash: `sha256-2471605f7afee948046ba8d9945bbb88f98b44184ceb10b745d87915b3882557`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d84e9ddc31e34697064a9e60de43374da82ef3d65551bc6676137ee0e90f5d63 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39774a34ef2d360c864cfc4c8dff731684fca37f62f9c81c22a5607fe8ccdd9c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b2e2018d14f6e4f7aa42463eb59f6dd806900d1bb68ca8ed4cbe3317386d18bb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3b94ae221faa8f26c21b605f394d502d38af124838a03a1a89fea1cd3634b60a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b78dcc53 | sha256-8387e5d6a0182bfa84f919fd5f355358e91477a72198bd5ef09b8c646cc4d2d7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b78dcc53 | sha256-80a59a36a05a0a6c93d7d12435279d07d6ad54397a9d25c0a30d0c6b0509a605 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b78dcc53 | sha256-8387e5d6a0182bfa84f919fd5f355358e91477a72198bd5ef09b8c646cc4d2d7 |
