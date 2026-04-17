# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b0963b5e18b40bd244437e7e0701feca11058e18371c7c7830706c53a28f15f0`
- fixture hash: `sha256-ad1fe96e9866fc7227d860e828f71679e46d996ff90526f19b5279748e32ad9b`
- score hash: `sha256-9f10a863469cf7d26675518b7c4d8e035cd08156fc9423f52dac7c8cec6218e5`
- bundle hash: `sha256-f1de7ef850a6a8faff402a198d7dc8f9c6d12349875c7f4dcb3d494723275676`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-63768ebf632f25773108234ec4f8850307fca437412854c0aa69b01f97c1eac8 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-403d0523ace973e64bd785de515e9bd21f870e8a0cff2b81cf343160c3fb5369 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-2a85b2fbbd93f2f311eed5d6bc81b7108d0c84170432f52abfbff1187d23234c |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-615a95dada2f1effd9fdd8f169a204510f2747a06833c183eeb20e1147e3f8c8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-735284e2 | sha256-61fa05092e3f48e5eb412fe51ee501c6b0ee4ba62ec068c4e6721b6ebbbf366f |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-735284e2 | sha256-44825c9e2fa0edd8ba2a0b2d94ccac3b5beb1dae9da1dbd5f463b9e95f7f6ffd |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-24aece49 | sha256-9898e6a17c386f2685ad4c3bd9e975fbdde692444f254051543a8f19e000fc62 |
