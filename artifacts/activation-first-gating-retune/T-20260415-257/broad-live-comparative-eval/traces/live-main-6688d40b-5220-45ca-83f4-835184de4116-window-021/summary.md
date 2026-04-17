# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-13cb1bca8722ed39c54b48c9d170af84a0229da5a1be3326ad569cdcb6c86e93`
- fixture hash: `sha256-3649ce5ca20580b372f2a2005a8164ef24eb19856bac1831bacfdfc2aeeebd5b`
- score hash: `sha256-2f9011f3b741339473cff40c972d77b8b107b62a5ad16beaa145701f051b11bd`
- bundle hash: `sha256-d979c55f4eb31b466b9382cff2683eeb2548d9e24b4c58afe38f23ec3a0149c9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3fcc253b9510f29399fe22001359326c4d47b1fc87658fed51c53d2aa08bb9eb |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2c17909762fb485083d81d6df45ceb729f33074a00d63b044eba61a80b8814c1 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8c16ee42c719835d8a2bf9f13f9883ee699242ae319f59b054beb332d4064f12 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-d3c966e581bb4f0410070c081c0209e009a1b7e3d5f143aef2316520b778e76c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5de646cd | sha256-21e42bd71505f842a94af39b0624da0eb04389baafeb45b84096ab7396a8cc84 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5de646cd | sha256-e0ea208d49d894945ce976b280de61ec977569a9b12375752e14342a328c9cf1 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-4ae745fa | sha256-909297b3ce0668d952ecb2e76db6ffeccc3ee65c93fc5ac808f156c61264fb2f |
