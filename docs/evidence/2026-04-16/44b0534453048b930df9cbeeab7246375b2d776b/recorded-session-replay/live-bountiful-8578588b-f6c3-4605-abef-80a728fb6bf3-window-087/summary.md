# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ddbcc39375c809330116a5c0c8dfb6ea4b6c6558d69583524e0e4b68dbba125b`
- fixture hash: `sha256-39075580e6ba979bd70972045c7a58c70d209890faa8f7b19eff384f00014d96`
- score hash: `sha256-2e40cf56fac741a81c189bf774531d80247795faf27d7ac772597592ca228739`
- bundle hash: `sha256-a9c8805ade9ce045f70145e1ed95bc54d43e1a3c77361a53ff2dc58f3276ed07`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9a05701b4e1402a8dbc9e3acd4b5bfc8e25db9ea5d315f0c5f4699b5421fa36f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a70466f8ccf1a3d516a6144ad02719741eeab6a2d6c0a996e374814633a5ea65 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f0bf3eefb19507cec6981881fbad9288520ca7bd0946394df531948db345975 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3a6bb48c76063c576f05e60ffea6c852a991dd4d29c45c773ea5a429cea3123d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-00b6aaf6 | sha256-bc7bc62058a5fe1bd6489cc759d9d2709e22b578a961494dce8ce8532530b337 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-00b6aaf6 | sha256-6d593712c887c1a03033dbe23254ccde87efc62278174987ce3aaf720b9f9b56 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4eceebcf | sha256-aa372319278cfd4d5e6094718003783a98827794f335a6ddaa46367fcf11a8f3 |
