# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df424694932b0793aaedff791f54d5ac971c24ed551452ee216f10c505396c8d`
- fixture hash: `sha256-cdd5cd85fb616c8f44b236f115a79978bc2dcad4597a177039207ba517f1bddf`
- score hash: `sha256-8641d7befdd1bc634759207ad44312cf96c9b5c6b54fea3da360a99ffaaed6fb`
- bundle hash: `sha256-d8ad1110ce2faccf3133f821fc7f5ca8045a4ec3c00f77b460e8876821055f10`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df3745ac4e10090248775f0174e4f7f9517bcadad1b8588a0276c1d2f867a57c |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a0c6bc5c2f377a639d2f4df5fc5644251422d9c00e9974eb018fd453f4d2bf77 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8d218ea41b22f8bba218d6fb39521bf2cbdac792ef026d69f7fc02118a904d2c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e884d11d2db6e30992b95ded5d06846fb0d079647d6c4fc63c3bdeacc8a88335 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2c6103bc | sha256-860029e383ad5df397a599356eac37afcb6ea01ff06a0c60ed58dc56f8ad4020 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2c6103bc | sha256-860029e383ad5df397a599356eac37afcb6ea01ff06a0c60ed58dc56f8ad4020 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-378ff371 | sha256-40906b502eaca751ac664c500ec7202fa758754e66b155f14b4e4b9b6bea2502 |
