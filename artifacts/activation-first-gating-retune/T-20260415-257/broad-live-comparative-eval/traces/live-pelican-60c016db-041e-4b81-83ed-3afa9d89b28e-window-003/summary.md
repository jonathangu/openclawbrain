# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-954e7370fea342c102d12700848d801f9cb4f766c26a152991092af096940dab`
- fixture hash: `sha256-71ae1b688e319cb6bd41b60a17bd7289e838d68bd59d5831fb46d3db379a64f9`
- score hash: `sha256-390d0cbf2d0a07ec7da2a6b820af1c7763b35ae47810e0db417d13c04c20ba52`
- bundle hash: `sha256-fb06fa388b5bdfc99465af1cf3cca9de36c70ec7f9175b776af1170d85c2d367`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c508d0d953d2e20d8464c0689a6d8d9a5c0442d5d485367bc8ffc01689888e09 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a1bfbc99cb29b8b91c3dd34cc1feecf7f0cf37d0ddbc8da0b419ef83cba3450 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8d4f4a7b6b04994f2d4eeac567ea54f3f2b39d7de8154a4dcc3d3ca054ee78d7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-af8c1964092ef91866038aec89952f975e8559ef9ee9cd22af2ca690f09bf837 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a2bbf9ae | sha256-ca97f618c50adb70058d157f12a90811fb47df13da409b7a72d425caa2b9d182 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a2bbf9ae | sha256-05e511d7371fe0295e1a4596f55e315e00deac4d0cf78c220ec2a9ab9435c8ca |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0298b33b | sha256-15480c4ff62580f0afa3c936a7febec2c83b68a588a80afb3f420755d6d98972 |
