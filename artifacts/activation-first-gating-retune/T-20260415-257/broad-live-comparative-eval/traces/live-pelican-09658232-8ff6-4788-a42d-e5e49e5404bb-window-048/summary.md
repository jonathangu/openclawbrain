# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048`
- winner mode: `vector_only`
- trace hash: `sha256-9c32a87b231e4d5848a772d9d1cb8d355e8b17c5c883fc0f1ca8776ef042ba2c`
- fixture hash: `sha256-66d4441e9cd89d5df06e129fcf70accf27e8123573950bf81a6f813e2979adc4`
- score hash: `sha256-bacdc4d92940995236c177d030beb6ff2c36f247e6d148011079e6864bcc60e2`
- bundle hash: `sha256-63078c1aa0c0e73c57686606004d83de63309cdb5bb72ecaa397a70f69473368`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 100 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/4
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b9d9843197c6ea9cf1bbaf94c65647f4ecfa1e2224f8678711a552cc896cd7e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-2ca040a52bb01fb5c0f6eca512dd1460d0cc0748e83471757cb1c89fcc3d044f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a6e6f763610b0352ccca78b6a694e1431ed0bf5d41857f0db24e06c4cab4f030 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-089918394a442ccf9a5983bff3d30d5681ddf2842607c1df3b31f91b1ba275f7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-9d3fecf7 | sha256-6a9ac10c1326ecd19816d1f42f7bd1411dce56e6622ca44fddbbf4fff0b2c099 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9d3fecf7 | sha256-43102371af2ec96fb1a9408951adad6ff8a105704db5050b81c6c1ce7f4cb5dd |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-ec5ad596 | sha256-bd1edf44d3f5449212f45d0f7239140ff669291322943bbddaf30607515e8706 |
