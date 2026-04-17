# Recorded Session Replay Proof Bundle

- trace id: `live-main-ea1c291e-11db-40af-8a15-d4d00cfa963c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d851cdfc065d530ff6a05cd12aae1453cc6c5cc252f286f05c63b39f7b7ea103`
- fixture hash: `sha256-add4e01555ea0b700f89e1179ee076e863d3216d180ce57f607f066d853c468e`
- score hash: `sha256-670d61f94cefe9725a01ca08aa91a6e6f597cd3c4afea2b0a07a100c9ccdda6e`
- bundle hash: `sha256-b2e4804646caf916c9f9eada13f1b7309deb8c8b9dbce607204189bb1d6d487f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e16f2c51fd8866c40ce249b661c20fa44d3a586d3c45a550284b22e35e90bd83 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9136db29f9c4794e2a7d287e99b91cf3bbeeb6a8c8ec529fbef383f02b66422e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc0687d9eb3035a224576b46f94dbe2c7199cdf2130843cdd350b75350c6b9ca |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-99816a82e8af3027ea87c2057c6527af4317289eeb46bf6de565bdfe49ae1810 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4e3f8250 | sha256-93cc6e779d75461635b3515110476452ff35dbae4d13a705f967225034d42e85 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4e3f8250 | sha256-7e9b440ab203960f94d9dfd8adbfdce05654b1da435032f3d716521c67a0d6e5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-bf72a64d | sha256-37d6363462d4d93b6a3da8a4be4139c1d86af2a22d7d3fbbce101841c8f7a245 |
