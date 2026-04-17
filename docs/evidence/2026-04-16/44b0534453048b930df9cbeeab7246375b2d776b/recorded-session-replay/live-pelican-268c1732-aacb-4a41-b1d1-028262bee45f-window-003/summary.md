# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-01a3188870712e041c94dac038c1913b6a8275f11c9b961a30d44d4a9193a2ad`
- fixture hash: `sha256-61072aa6754828e3628c89803b9d747baa926b5fb67bd06b8dcc6e5a7d888974`
- score hash: `sha256-6af8bb85a1c39c734237b96aad7b1837926cdcb3b8a1a53b9bfc0584b30b407c`
- bundle hash: `sha256-1663e587cf07cf9dbfca19d98e840c8b7439e6c037f1fb28955ad275baf96f37`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2838eac92b5037b9e7a88f6a187516f6cafc0d1eb9fd70438eb0e0126665d9b4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-af7f1673ae9174d972d12c9f6c79289e3cfb658ee776fd0d84114572eb7161e3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1092ff1ba7c2c81f7a0e0c713358fd15820391297eea87deea024a3ba1c74a7 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-44eca5354e95f423a3d90052ac29b910f45d1b9a28dbcdfc4c252ffe859c9412 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-eb9234f6 | sha256-d4983fed046f452c55acf44e6646505c1a47631d6be884d68e71284f961f2a5d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-eb9234f6 | sha256-33984532d2d52aca5adb2c38de0e5a2cb784e56b33ac8e2223b2881d78a439b4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e97634ef | sha256-aa34eb6c719a30fb1a258fdbeac0204af761b8037720740664d4161ad9bfa7e9 |
