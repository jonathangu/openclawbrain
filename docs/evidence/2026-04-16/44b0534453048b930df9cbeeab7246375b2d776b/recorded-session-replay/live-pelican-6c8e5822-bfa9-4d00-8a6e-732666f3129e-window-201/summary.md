# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0daac77494acc6aca056ecf3e9f12fd58f33e4988f5a646f0ba5dd6ef080149`
- fixture hash: `sha256-267dbae4de2075656207ffdf48cdf822d6c7cd1996c42f8535ce786c53f3660f`
- score hash: `sha256-88a4ed8181c4b7a6e2f9fbaba3c3f5e149c6186b3db9aa8e899afe2ff157e5d6`
- bundle hash: `sha256-762caa448d2c51adc90505833ab90003e0fa43080159bee246782c58797b2d60`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7fed91035473e3fc2f9947814563b07b47451ce7ca5cc7e497e3f1c68f58c389 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-122c6f1668c68de84ae4149a0cdeba735f6b0e1f33dfbe6bd1bcbc85b3c5b4b0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ffa9554060c939d1bda65222e4854f651d9d84d44afdac26ec9d7a5d555e7865 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-dbdddd7351f784d13074c8b44a36b05b379db85638e94724ab3686728f443718 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-06862316 | sha256-15c5226d135c48528fb90af8cefd57efda60587d7562368e83f1aff7f38df758 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-06862316 | sha256-40094f56c475ab4613b184b4ab2e86f3d07c3277e872c007f8c65c0a3eb2bc9a |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8962cdc7 | sha256-8140f51b6946a054140ec1b6bd527de86dff7e864fdce585781c6f336d5a4304 |
