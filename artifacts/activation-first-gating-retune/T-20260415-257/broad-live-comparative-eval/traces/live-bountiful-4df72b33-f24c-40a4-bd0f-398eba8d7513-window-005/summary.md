# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3abef879206da47064eddd47e25da3ed69b90db7cd3c4a8ad4966415b7f00bfc`
- fixture hash: `sha256-9918ac1f02e6942937a0c165ef4e1221b4c237d331f00ffb8e89f19fa2868433`
- score hash: `sha256-b111b46583d87f28a094826981447b9a8f552718e1f42b5d9a4e24f4a472cc59`
- bundle hash: `sha256-33305710cf03dbc1fcbf237bed31c11169b9f0931966e2baa2ae1206b5c68715`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29b14ff43615dac430701017e1a95d84a605d40df7e69393e02bc78849368384 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6f2167c15d1d52d07c3ba945e86de0b21b06b69346f288e21ae0699db3880f24 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-29c0b23f7d034aead6fe393daff329b819a7360d47af08f4789d859cb5336c99 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8a2d0c3facfb9220e92c31aa416156ec5b7165bf3f8f742bef800ccc4c251889 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1cf9bb97 | sha256-9de5860282548a710f42642d1ad3fab4cb00ea8e3c562a9d61f5cba56bdccf85 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1cf9bb97 | sha256-a8fab10e008b38d862c31dc9cb79433ba4ea32fca3658f893c146e385ba61e96 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-00a7d64a | sha256-57aebdcfc8f4af95aa071ac54613097ae31068b77083af20346004c09b313c1e |
