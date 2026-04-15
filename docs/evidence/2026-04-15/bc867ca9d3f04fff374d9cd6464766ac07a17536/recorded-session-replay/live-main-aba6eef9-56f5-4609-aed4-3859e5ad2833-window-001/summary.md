# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af95153d1f0a3be68251ed9ca1c6eec687f3524276f083ded8a5b5ed5deb8173`
- fixture hash: `sha256-e08d20ecb487c4dc497560c31f0ea6c918c59692d8f02eb17f1047383fc56246`
- score hash: `sha256-84828fa6b368f492f1d9173fdbb77da93a5d8596e166d060cadadd0ecff15aba`
- bundle hash: `sha256-dafd0709955118e9bfda8e913d4c9821c3f1799d624db0b0654428d7547603c7`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-af28d7bfd3a7d04b34389f31647ac0f041f3d52e501bb74630c37fbf1936f421 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-9f4bda89f416da57428b66584a42037314b6c66e112c5e31b6ddbfdabed81e28 |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f40d31b97a5af52d22ce97a8cd50607bda738056307b28f3fe293448e802731 |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-2d6dffcfbff6ab22574c3c5da418e1f63f5c58c0438c5a99502cfc3f241a8736 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-0fc0b177 | sha256-7df7b6de9754a0d4df2729ceefd2d2c1496660f45ffd07dea65794d06e3d8dda |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-0fc0b177 | sha256-7df7b6de9754a0d4df2729ceefd2d2c1496660f45ffd07dea65794d06e3d8dda |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-0fc0b177 | sha256-7df7b6de9754a0d4df2729ceefd2d2c1496660f45ffd07dea65794d06e3d8dda |
