# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-02e7d65886ec7c9c662814c419da7cda053558fd638bbd6332be92bb767d2ff2`
- fixture hash: `sha256-6508603acac99ba814ef6cb1f3424ef1d7247c3d69d4612af9ca33edb2806300`
- score hash: `sha256-ba39769bf96b20d58854d13e4008c744fa45579cdc43321b702cc85719606ecc`
- bundle hash: `sha256-1b92c190b005c5a3892f3c11a7dc733bcb0f38c7a6b2e6763c4c598d6d2d08e7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39c1106c4dbd631708b383c103b666734b15b0322fe3241c472e8fc7fee74258 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-16f44b38bc0f079fc46c795072a05bbc6fa8c09097f4ce3d7b96b8278b0e4181 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fe80f310a512d466e41ba88f96a1a56e0c275feb2653f3ec62fd839febcd72d3 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-91ada8b0f52ce492995271c518e131ae5e35160d0f2080abfa3b16919c6cba53 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1be14d42 | sha256-19d145fa3e7784dec52ae75aed0ce9d5ba1044fc378c0fb1e8641fea0976e8a8 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1be14d42 | sha256-6d4dfe9e141fca588625fd921d8b3cc4d78719c578e27f29382b8ec32214ceeb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-2b18278b | sha256-3de70da0ca3e1b09f0ef04d994ce94273a1ed53faa30a4f839881ea5c3666887 |
