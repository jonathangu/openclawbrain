# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-01a3188870712e041c94dac038c1913b6a8275f11c9b961a30d44d4a9193a2ad`
- fixture hash: `sha256-61072aa6754828e3628c89803b9d747baa926b5fb67bd06b8dcc6e5a7d888974`
- score hash: `sha256-e89af162feb8c142654655dad662ca8abd2c93a4ac40bdc33dcbf953015758f8`
- bundle hash: `sha256-67fda41f5543b21939c75238a921b4021b4f7896a6f4907f15b6c6898779c81e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2838eac92b5037b9e7a88f6a187516f6cafc0d1eb9fd70438eb0e0126665d9b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3870588d0a414a4e9ce443ca418b035a4fa170450684cc91f719f417563294b9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9c79197d57229d72959383c8139b80e19e1344470ffc45b8d7471979749d5b8f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8bd95804549d2e3aaaa82c8b3bf03754c84840745cf634d5612e7d0ee473dd63 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45bfa1b8 | sha256-9df08da761ba79815169bf26482257e4b9327e06a64ce57b492352e47f954714 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45bfa1b8 | sha256-7509dfa06953a835756c225b4536e56a38e070fdd9d19797c11b981015937ebe |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-43a3a1b1 | sha256-5efa614c23f86266e706db219f5695a449e5171c6f2777f098eeb4129f117d21 |
