# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8836c19286d7dfc7e25365f74f5e0786007d5f48b08d8fcfba5fe79b0f03c2c`
- fixture hash: `sha256-0ffaff36365448396a5594a68d8364ec6eacdae9fdbcb2693a4ddbea65547f4c`
- score hash: `sha256-a1fa7ad0c8c3d6686a85369ee861e191e9561c7638d3fd53fb7203cc55e9034f`
- bundle hash: `sha256-e4089cbe060332b9d86c69ace307de05d012d895afc0d3f579754f98f2d42cd2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c4e67449219060e0eaa53a64e9ca0f2f7168ec707e126564ccb072cf633b7d0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a84aa0c996e3cec739eb6208c797cd357e9d7ef322e3d8483d6d5d5c601a8610 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-018f10f8878b9ca7f3b13c0be2648300f138377dbc137f180ee9dc3dc12bb200 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f276ffb208f1c69c5ad58c31015913a82afee9a83f4f924927342ac8e35a5939 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b753fb7e | sha256-765014664d698220e542c9e42b77222ea48cc6c87f8393852ce55080ce1bcf2e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b753fb7e | sha256-50722abc0a1b8857634fc4130794c78468cc588b16dc48a799dc7df322e1d6a0 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5a77cf53 | sha256-a4ec6ed6792757ad23702fd9516c44dd89f4fa7f981851c85919018b9be71b7c |
