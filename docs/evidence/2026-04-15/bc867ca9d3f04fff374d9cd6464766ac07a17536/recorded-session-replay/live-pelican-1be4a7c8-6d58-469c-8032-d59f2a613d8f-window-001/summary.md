# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1be4a7c8-6d58-469c-8032-d59f2a613d8f-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9c0b90297f99ade602878feaa8cfde6e3a19db0e47440bfe22629154903dab61`
- fixture hash: `sha256-1baf21d3d9b73bfb53336d6a81b7f65e4d6e7e9fb603fe4e8af018eaeb0d47ef`
- score hash: `sha256-c07357b93ba3cc221f195ce0723cfead32513fd359af063a96f9f46aeedf4aa6`
- bundle hash: `sha256-8f28f3be95f3f5b0dbbc99ff39c69b8b381095d7a33595141596d7d1da937f5d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-27b2a69331fb76743637a0a59a8c052316c43dae2eb924cfbe90678912704fb5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-07d81cc7943648e5d6daeb9b5afe08d8fe0b03e5bbdad27f6c86117f0524ad56 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3ab263b359e10e2e25b34f88804c765c06e2c32c36c4aa968d6585e4dfe99695 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-52d105b255e6428f2ebedc0333a4c3bf1dd6ccd57726195c87a4a2a80c7c1fe1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1c37d361 | sha256-1ee89db4dcc1a5b74f49f36cb0f2ef3f0456bc3743ab70fe562b7f5b2b99d77c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1c37d361 | sha256-1ee89db4dcc1a5b74f49f36cb0f2ef3f0456bc3743ab70fe562b7f5b2b99d77c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1c37d361 | sha256-1ee89db4dcc1a5b74f49f36cb0f2ef3f0456bc3743ab70fe562b7f5b2b99d77c |
