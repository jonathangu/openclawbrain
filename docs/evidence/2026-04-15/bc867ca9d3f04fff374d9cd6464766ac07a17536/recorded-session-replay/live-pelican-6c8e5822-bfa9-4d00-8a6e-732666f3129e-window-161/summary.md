# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-161`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b83c4e3036cd20627db9b3b691867f0c4cf67798691db96c05537d9efc454f0`
- fixture hash: `sha256-d883ca17da8d181a1200f08513acd619f27d5b75e1c49c4953044231381c83cd`
- score hash: `sha256-2416d49b6214a6dcd1826d384f8e58de6f61a8f9e7d09b987a90c68df1fa9004`
- bundle hash: `sha256-2eeaca0f608008645418be0237458692dcda6e464c211ffb118128695f17fec1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5551839106976537fe1c9ce0dbda883b66824b3f67f3049bc2763f475be1647 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-482c169c13ad67d8a5b26d23cd3613666721a36becbf191472d71f3928836350 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-736eff331cadaff7538e0fa8a05aa29ee524be3de2810e71ea19874c8fbfe661 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ada5a2993868ae357a50362ea39cd0d67c9937649df971fc2f2013593cbe869e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33ab5660 | sha256-abdba7b7e19e3beef6c14b800670208e6c0123e9fb3700496e2ee7721c357296 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-33ab5660 | sha256-8e383946f52623417654a50ca468f8d09fbc81569b7b558f40639392e9750a34 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-33ab5660 | sha256-abdba7b7e19e3beef6c14b800670208e6c0123e9fb3700496e2ee7721c357296 |
