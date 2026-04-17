# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60561f5cd4b9679d1d07ec70fb93c8ce09ef36cd5a40b0352b67931141e9e246`
- fixture hash: `sha256-d68e77ff5a53346b0fae859928eb6131851ab9f7d88f52a94509c0f85b109391`
- score hash: `sha256-dea60efd14f314e3c1eca1d45f48b06ecd14f67110cab3a1184ab6c6a1879c2a`
- bundle hash: `sha256-6ff86611eca16338a583bbefb716958cf0a175e0e2589dd741872b7c6cfeee5b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-350a961987884e875cb36c29ae1cb810ef961abe38158c92bab3e2c95369cbcc |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dd14e35b85f428c0098e6f544136573bac3c97f6d4606de5964624d0eb067c80 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b6a1036a601e06fb68d2583872e28d29e56ee3ffc29497eb1b7f4741b91bea00 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-615de1b76ffdcb3054a94a62be531b42e43086bd15af1b37988936006ebadbb9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fdd2ab54 | sha256-af35a29bf6519c7e69b297430ad8aa5ad0c56da3d8ca512c7befcb3b5ed5293b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fdd2ab54 | sha256-b39212a2801f490eba237d19814569aae51190724ad35d38d105262e4fec1ce3 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-55caeca1 | sha256-59d725502db9b30ed52548fcffb066c6ab0a96e9ebde64425154e4a272c3e1a6 |
