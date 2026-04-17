# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-064`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5976790ca942312652f7ed18bb2acfc7dc50f422300df52366315661beef5eed`
- fixture hash: `sha256-90ca4b5e31599f276df6b4ae45b8fe949a2ef12d77f2e3ecb7cb55c21378ce2c`
- score hash: `sha256-f9497ae0ea478ad5293adb95592a4e66038f85fd6544dde73157042737091bab`
- bundle hash: `sha256-c900039493c1753950c7b105812ade390f03afa2d3beb9e041e84e1a0f442b96`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-df55308d9f9f7142399061b3ae503fa11ed1103552e4d35f047c53cb2babd5e7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-02c831b80db59f353158bba4d1bea511154beda8bdee2ff3e5f2b09f2cbff61b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ee7c78335f0193e9c6ad7a2d2b09eb64d8b48cebf758a34fb1704c1851639446 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ed597b472ea68669a518d8acd2ceeba9ba3dd8eee45796c31e1339fc108d5ec5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8b821a4d | sha256-c0ec1b16257e0f273b2377c7268c691bdad84f62b3bda3757728d17efc0ea2c6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-8b821a4d | sha256-2e5fa0fa09f146d150b576514be8d30d780ab70663e9c53f929b7e65ab86b7d9 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7bce2fe4 | sha256-fffc3b0e2e966f5c5a951c264d44c283dd9fe5b63b381100573d294670eed6e6 |
