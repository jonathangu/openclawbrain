# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f69683abc74146be49e8afbd73d2f629322351b8f1ff326bedad7089f23b35cc`
- fixture hash: `sha256-78ae89352ee0e2620fdc9e4b5d6b74ee70bb4cf28775ccac9315ef7f4b6b2525`
- score hash: `sha256-7f23b625c865026504e021832051a2345e9af36dc6e8c35d0db3fa560a705b0c`
- bundle hash: `sha256-dfac9c89611b5e7ce0571cf7d992cd2cee64accbb09c51f29ffd58d7a0b9338e`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81a98d75515ca1c6519d32d4f8b5120338f9765022c93b90e0504e9561ef38af |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-267248ba613f4dc27f05963e4c152ac9e34e77acf44ae1348a2bd9f0fbf2cea8 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81887b924c8540208dadbdeba88207aa93c1b384df28c5efb9a3f1e1c779c0ab |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-05eae26ea626182e07ca66ae49f44bc4304d18277882b74edd85e464051bdf25 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ba1a0979 | sha256-dde7355f612d26a89d4d5757f6e7f750cf16cdc69d1bc00b3cbc3bb0f2e5ece2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-ba1a0979 | sha256-5b9b6dc13990895bbf18e9d6126310b013d2d346912c9c504bd9f250fb101376 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-f06d3f66 | sha256-8c884c7acda9b77fbc36bbce1c82dff358c7fa79213ced306a592c6180f466a1 |
