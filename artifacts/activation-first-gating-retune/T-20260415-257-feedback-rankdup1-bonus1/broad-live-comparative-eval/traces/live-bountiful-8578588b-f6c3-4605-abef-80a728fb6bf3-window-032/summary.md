# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-27432a79558194d97a71b7fc8ed69705aaeaffac6684c66f5d3d996c91fd30b9`
- fixture hash: `sha256-4ee00dedd58e8761fa82d5e969e9f577592259713a36cc6112630837d5f1e052`
- score hash: `sha256-5d188626b1e852b64ccb4f42c3b609a478469ef8fe443ec7bb89dacba4c3b8f9`
- bundle hash: `sha256-5b60ddab566bda320687f42761f8de6a5adac09aadf4d892533b333b5a925fa3`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-4d81150d8010579662ffc1f98c25458967a6d4099d5eafbef95e912e12ddc88f |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-01a9c088d7b83dc0e4ed7b47b05f98ae6c5bb0c87d8b636beb65eefdd252ebc5 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f3a7dbfaf5117e1d2a0eab4679115f579feb85dbf1850c2b48051035b9be8b0e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-f5a38d5a | sha256-e2ad293084c537cf2d2fcff70d353221d6667800ed973b1c804944a7a6914864 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-f5a38d5a | sha256-e4c2a5784ca1b4f9cce80f08cb2d0e353954d88d8dd7656e535b5ed65f458e3f |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-f5a38d5a | sha256-e2ad293084c537cf2d2fcff70d353221d6667800ed973b1c804944a7a6914864 |
