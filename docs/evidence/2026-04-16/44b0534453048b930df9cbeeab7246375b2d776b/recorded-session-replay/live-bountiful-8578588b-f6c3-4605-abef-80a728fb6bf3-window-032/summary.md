# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-27432a79558194d97a71b7fc8ed69705aaeaffac6684c66f5d3d996c91fd30b9`
- fixture hash: `sha256-4ee00dedd58e8761fa82d5e969e9f577592259713a36cc6112630837d5f1e052`
- score hash: `sha256-e89d13e270853b3d954b2b69daf7c70eea054502d2a6a0227d402759f09e5b38`
- bundle hash: `sha256-eaf502198cedc2f9f70c94a5369b06813127c1efc034cf431b11e381867a7a97`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aff496b7da6b579f98fbed5214b0d11a4e8b4ff3f3ff678a80a14299990c301b |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-856d489462321d2fc669c4f0d793eeabf1a03465df34af755ed5c4b8330ff6f6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9badc9aabd56eb90d297852b160eb3d6ca92b412eb7471be3c3c7697f022190c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-51603005a63ed7d111cf50c1ad1275b14148af5b73fbbda188f09af5e0a6d11d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27f90a6b | sha256-f584c3732ad7cc3560f315598955785679b4844903b9a8bd0d94e6560df20304 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27f90a6b | sha256-33afb33dbc28b6ed0f902b55521193e385e035816dd2ef7c4ebf9a8dad4e46fa |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f5a38d5a | sha256-e2ad293084c537cf2d2fcff70d353221d6667800ed973b1c804944a7a6914864 |
