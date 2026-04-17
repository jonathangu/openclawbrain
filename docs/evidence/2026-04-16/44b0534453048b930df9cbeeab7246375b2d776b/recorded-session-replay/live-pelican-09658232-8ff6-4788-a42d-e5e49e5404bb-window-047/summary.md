# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d8011faf3d69f32bcf0e92bcef735c94f96aebd8322b667cbe52a25917f1a6e`
- fixture hash: `sha256-f28ec0241ac4efd4c1f97733d381efba161e2d4c7cd778ddce2f415ed4529529`
- score hash: `sha256-1a71cc53da535cc99c8173f49d629d647389aae020e87abf1a8747f478cabf99`
- bundle hash: `sha256-f16e4afbfc85183bf38c2aa1979c7a76fe144fd062470f2e8923d97131882f83`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64f932981a8a1428e017d3b3bf8eed9c04a8f1b43e3be668df16d36de77d3b6f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0bae1de589d170468de408a7c5a798fd290ebeac21a6173fcd214c25bba4d5ce |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-424acbce5be67483c58d9d09905ac8c2558116f98392b53d01343d960058189d |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f8867862887f834da6ef2b1bfe43bd456dd9abe459c6b8bac30f4e2f774a8df5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-67fda4f9 | sha256-1c61009b09888d5b3655bb2151cbb2187d1775db1a8f275137da95a69770c59b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-67fda4f9 | sha256-5a5fbd84f3f1514f2f5bd25141025953a4c1dfa2bac436d2b340fc314e41cada |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d033c304 | sha256-6414f78e7b35522a8ee490782a7873e97b258c7913ebd014196d1024af242692 |
