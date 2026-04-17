# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8836c19286d7dfc7e25365f74f5e0786007d5f48b08d8fcfba5fe79b0f03c2c`
- fixture hash: `sha256-0ffaff36365448396a5594a68d8364ec6eacdae9fdbcb2693a4ddbea65547f4c`
- score hash: `sha256-0b4a5dc9160450825df47a9353e4ed29e9f59a4bb80b9d029a1ddf05b4cd8139`
- bundle hash: `sha256-04594d515b6187fc3dc7ac38db399404af977e13f895540281683e21d268d5cc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c4e67449219060e0eaa53a64e9ca0f2f7168ec707e126564ccb072cf633b7d0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cdf0c0cacaf17b75ecc3fb17853ff2cc06bbd02de226467483dc84bcaa1d773c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7d92257a4ef177485e11d9e23999aefc5dc424832a02d636aadcce46e85fe083 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-cda05002967906bfeede27609544ffa14e4f3cb76655038b56b1da68529c9d96 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-82eb76e3 | sha256-6896bd24c85979c321abf76dfa72d7cba784e8b5a7c6975e3fb147c8a9175382 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-82eb76e3 | sha256-09690ecde747a61e673bb583eea3d31e3100bc2314b5d96cd1b265f6d938b4b1 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-260f4ab8 | sha256-98bcae23c519df411f0aac50e77431f135e49c1405342867928a08d4e0301747 |
