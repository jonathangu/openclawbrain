# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-27432a79558194d97a71b7fc8ed69705aaeaffac6684c66f5d3d996c91fd30b9`
- fixture hash: `sha256-4ee00dedd58e8761fa82d5e969e9f577592259713a36cc6112630837d5f1e052`
- score hash: `sha256-68b977cf7b6e0c9fe8651fc54c2fa2c4cae3055aba7dfe88c29cd39111164a9f`
- bundle hash: `sha256-3b07e563d698634071b099249624a9c6bc85e0d139b958fb7982c09e06db94c7`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aff496b7da6b579f98fbed5214b0d11a4e8b4ff3f3ff678a80a14299990c301b |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-856d489462321d2fc669c4f0d793eeabf1a03465df34af755ed5c4b8330ff6f6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9badc9aabd56eb90d297852b160eb3d6ca92b412eb7471be3c3c7697f022190c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-16fccbececb55ba1fc46d93f81fcfffabcf5b6642763f3708ac16cb139c7bf13 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27f90a6b | sha256-f584c3732ad7cc3560f315598955785679b4844903b9a8bd0d94e6560df20304 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27f90a6b | sha256-33afb33dbc28b6ed0f902b55521193e385e035816dd2ef7c4ebf9a8dad4e46fa |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-27f90a6b | sha256-f584c3732ad7cc3560f315598955785679b4844903b9a8bd0d94e6560df20304 |
