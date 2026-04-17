# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-022`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5d4b7ac6ed69712b1588ada5d64482dda6216ae5bbb670a70c4e5011448ae050`
- fixture hash: `sha256-c583a0a30dc7272198329e0ce06b64ff4fe39dce1f96b56a4f82e04f4a924ee7`
- score hash: `sha256-6803fc5d76aae9b4a6d53a79a5f376a11c217cd160eddd90f32e3ec3e195fdb9`
- bundle hash: `sha256-61e77ec469fec803d23aec13919e86f8750909f78595392cbe8ec329a1e1c932`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-646908dce1c2aa715ec563720c445a9dc7233e215511f30956abcb8a6c0f9113 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b6ac92f8d17574de83e1fcd4d5385c16d2c352eb26f67572ed9292aa00118947 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7bbfe379d6c576459412a33dde7f8b59745db33a20320a9d68f5a116f31510f2 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d6931112214a9360c46353495070a0e8287d59f94b3d44daf2e524038b40971d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f239d04f | sha256-fa13dcbc18a8b57a7b18ed1a5db4c3df78ed665822acf14acafc7ec043a5dd8b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f239d04f | sha256-32ccbac001dba426d07ebc8fcd95f513ae293ac498e3af08d4117b6e5af7c02c |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-dfa6f2d2 | sha256-e737794128f3b1282df765f654b69633e90a78d3ed77778c1317a2b78c4c0801 |
