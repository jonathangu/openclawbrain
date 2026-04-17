# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22f847bbdb3bff7bf5823cbe39964b330b3ee1ba23484549f7f4546fac1981a9`
- fixture hash: `sha256-a7f2ea82d1ad7a3badc44ebc7ebcd547c985d36abe3fcd06170981ec576de057`
- score hash: `sha256-38ea451ec62551e1253542661dc97ded0097de0569b551bab52a10b5544f7479`
- bundle hash: `sha256-35b4bee3b72b33f737b5a15a319f034dc5a6cf52808de42d1fa7ce29cf77ecd1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9025e023a7eb98100239409dc6df273a8fbdc8529118429bd0cb2b4995877ef2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-92c343f0235368e82320809ef807c18e1908cd86fac9df20e77a2c6e05e1c411 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-44f51e6db966abe1f528c513a26f41ad6348c417594e3cdc565e92cb11decf09 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b2a899bd3516dd41de6ff76042bd1f89a81649e3225cff4dda1edf3279169a55 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a70a8c22 | sha256-b7c2b8bd1b62d5a5c3d8ff87a71a7553fad32bee38fc211a8c69a08a54077f74 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a70a8c22 | sha256-51de57e71ead92caedf1b513620e054eff1be66682688828d90a65038a46900a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5314a9ef | sha256-5fa74db9b5e592a30b4b75b99243d3f77795e14af393a0136fd2aca8771c5d63 |
