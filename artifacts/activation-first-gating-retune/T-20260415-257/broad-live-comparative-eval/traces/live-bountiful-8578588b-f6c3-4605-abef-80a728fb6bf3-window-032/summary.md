# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-27432a79558194d97a71b7fc8ed69705aaeaffac6684c66f5d3d996c91fd30b9`
- fixture hash: `sha256-4ee00dedd58e8761fa82d5e969e9f577592259713a36cc6112630837d5f1e052`
- score hash: `sha256-d8c8701a2aec1083c74f71ad27fc4914d1dc78e6f761fa02ca1cd5978f3e38c7`
- bundle hash: `sha256-a34ad34d62542a691246c5910923f85262b8924c77da7310d211ee552f265b70`

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
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8a1cc89b54dc634929b1e552fb9aef16c43a6cb7ae025c2385031a69c7865d2a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27f90a6b | sha256-f584c3732ad7cc3560f315598955785679b4844903b9a8bd0d94e6560df20304 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27f90a6b | sha256-33afb33dbc28b6ed0f902b55521193e385e035816dd2ef7c4ebf9a8dad4e46fa |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f5a38d5a | sha256-114defe690235df656996a73e63633def5fbf435638237357c8f73c653a10860 |
