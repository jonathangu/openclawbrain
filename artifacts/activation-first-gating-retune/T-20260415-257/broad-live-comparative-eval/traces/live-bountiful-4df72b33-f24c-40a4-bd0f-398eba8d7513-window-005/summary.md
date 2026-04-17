# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3abef879206da47064eddd47e25da3ed69b90db7cd3c4a8ad4966415b7f00bfc`
- fixture hash: `sha256-9918ac1f02e6942937a0c165ef4e1221b4c237d331f00ffb8e89f19fa2868433`
- score hash: `sha256-1b628015914baa759947c304e59cbff9cf702fdb02aee8498d0a56275ace479a`
- bundle hash: `sha256-37b5912eb7ab029248cb22088a9abce2bd7c9c97d6b51b970a49e71124e055c2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29b14ff43615dac430701017e1a95d84a605d40df7e69393e02bc78849368384 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-683c632b5611023f2da9a40b2bfce45656718bafb4f722ec23353470792904bd |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-3b78eb8c39ccead68427064b8b1cf24e154a81cde3cad848885e0724e4d1c0a9 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-26679a84429336f34ce22460c168dfa42c55f7891aeff171b3de275e99c6a631 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1e7eca63 | sha256-18b0b5078b0cae6d61ce3f992d7f6af6ff72b9905eb446615bd900108338ede5 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1e7eca63 | sha256-f80202a76f8cddf6d4252c1144a7b3805a310ec490c739aa093486d7160620f9 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-022ce516 | sha256-1a470381498e528a68a6cb2c4a1ed123b6e42fc6324402ed274e5f4fa5306aea |
