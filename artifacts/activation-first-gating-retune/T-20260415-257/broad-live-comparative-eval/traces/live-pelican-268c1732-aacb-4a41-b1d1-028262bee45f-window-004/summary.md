# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-380e12e9dd757771937f4748557c11b50a1f9a231591dd724ca65839af3ce6a8`
- fixture hash: `sha256-86ffcadee00971f5c46315d2afa19ae2e85e45bae4dad0e458c42f57f711f9d0`
- score hash: `sha256-8b8cd9fdba2fed56a370795225c7af513454c7916647da028face96d751cfcbf`
- bundle hash: `sha256-75a96a286b5b977dfeaa8147abde2f0b09d04e5079db862af0caea05e00b47c8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0928afcbb7b85c41b2a1d624e920cbdedd75575cc8baa6c3ef5218e9d291b99a |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-6b283f0c955a8556ab20a049d2487be6a045d2c903e45af734b9574bc5c45ea0 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-9df6b19e6216dbbf85d01e8ab5f4095371c3eb60339d277b93eb2a4b6f1880c1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-305a1545049cb6d6afd690f17fc685c28f5604ab17277fe6b30871f58b9ff558 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-11509e1f | sha256-4130ed9c2ec8a5796253b71c1c403e96dfda173ebfb2f2d9b6efbf9fda61426a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-11509e1f | sha256-e0e5d1e2218efd9bc7369eebd5c83b16ce3b10684826c9e3cd8231f2ef9c7489 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-11509e1f | sha256-f3fa878b28e604b7ddd19f42e592409403b44b9e32afdca251f7c1ed09637429 |
