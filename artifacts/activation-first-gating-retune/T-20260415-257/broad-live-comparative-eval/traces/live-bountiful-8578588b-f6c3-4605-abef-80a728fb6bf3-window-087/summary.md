# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-087`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ddbcc39375c809330116a5c0c8dfb6ea4b6c6558d69583524e0e4b68dbba125b`
- fixture hash: `sha256-39075580e6ba979bd70972045c7a58c70d209890faa8f7b19eff384f00014d96`
- score hash: `sha256-f85964dab02de07678f38b99ade4394324c54279d86abd747ded2a03f68ed9f2`
- bundle hash: `sha256-1ac3dc77ebc5b9ff6928f38a0f0a98385b87043eb66e9ce6987be70b29c1a47e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9a05701b4e1402a8dbc9e3acd4b5bfc8e25db9ea5d315f0c5f4699b5421fa36f |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-015869189b2841ef69af428cfdd246768ed36ce68572d8f30e71ab8118173461 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-1b44a17bb927fd51318ac014f87fd72866f1e1f657fe052d96c8ff9985891f60 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-cdbdb61222389027ded6aefa2c712b36cc17b12c9b65b036a4082954b467f1df |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-4eceebcf | sha256-aa372319278cfd4d5e6094718003783a98827794f335a6ddaa46367fcf11a8f3 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-4eceebcf | sha256-692928076e93447a4b0a14a58bb48c8d8c302db0a3aeb86ca2a14144f7d22f47 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-4eceebcf | sha256-aa372319278cfd4d5e6094718003783a98827794f335a6ddaa46367fcf11a8f3 |
