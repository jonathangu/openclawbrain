# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3a175359354be687e420ebf0f5b7a7ba6d56063e99f8d68c1b9194462319f20f`
- fixture hash: `sha256-598383fefbd171f64af80d75b4d0910cbb4c4236c56c222c3e6f677cd87ecd08`
- score hash: `sha256-63428f7cb7ecb13f12b0811bc94881cde35ddf7bc4556b81a5f1136b799d2f5d`
- bundle hash: `sha256-0ebea1a78f91b895491a8d2009e1bb44f1498dcb75d6f2f4d68ea929ffca4f16`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-566dca0adc9e80f718fe72c8b1c0886a7b904a433ed25544bd407de590aa9335 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0a5ecfc2ecf83c6a831c84f16a4b3c136f6c7e403b77a91e034ca749421f0f13 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f988aa554050c9a5fe91f154c0fefb6a7633caabaae91ede8a9701bfcd203576 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ddf32fd38c6cfefa05ec1fc4de2e24be3697a0f50cb9ae5eba78565bca3ce755 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4e82178d | sha256-f7ce9264e9c8f28b695f91afb890de443abbb8c1e1c81bce004f552dc6d1b0c7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4e82178d | sha256-e6ebac5c584322240d9f9260bfc8653218694a91b5742399bebc2a735292985d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c432b1fe | sha256-e6403cfb0623cf0f77aa95bebfd6689c5ec41f752f2f4928157918a6b2519be1 |
