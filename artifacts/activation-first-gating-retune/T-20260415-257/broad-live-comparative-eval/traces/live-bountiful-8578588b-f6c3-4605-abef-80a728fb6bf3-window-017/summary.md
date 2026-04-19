# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24534757347a93b75b187ac38d7e9e86602b361f9d21c2720eadd2aac5437955`
- fixture hash: `sha256-599fe6907f3cd26dea75cb20dba6e419b550fd93a91244dc2a42f5a954807c1f`
- score hash: `sha256-047db00fe9512efc1d5795438ce6b030abb8b1ff99b34c5f2263d61c7637c932`
- bundle hash: `sha256-8e057c83a1dc97ef0dd6f349d571641b5ca9bbf0d535155dc034cf728feb6eef`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fdc42d0f374b0d99f1d90ad1240350cf6baad53baa8fca8aa6efbbc417c89845 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-d7fcbc7708d42f1ba6ee2ed277e1d1d4627abdeff1fa5f75f0d17845a428e364 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-0882a26b8ffba2d04f5738c95776dff58ad49041ebc0623ec081ff97b86e13d8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e3775283dac55a19e7915264356cfdec4950ad3180cdcf3f8d532c165e19cb13 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-44896b34 | sha256-c270a1556317c6e7fcc80c93e0390b0c453223ede0867b4b5307b071891560d0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-44896b34 | sha256-a1e16566052fdb61306928c3ce41fb07b6e89ee4fdd18c4d627334bb318f7687 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-44896b34 | sha256-c270a1556317c6e7fcc80c93e0390b0c453223ede0867b4b5307b071891560d0 |
