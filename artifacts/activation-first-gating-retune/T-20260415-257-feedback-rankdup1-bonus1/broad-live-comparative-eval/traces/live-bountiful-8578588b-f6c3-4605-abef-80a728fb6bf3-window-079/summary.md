# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-079`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ac9290c0886604313b41285576a01cf7cf17f3dff1c7512072e0576b64e4b6a3`
- fixture hash: `sha256-e8553284ec9b39217012ceb74491f30f031830486fe62cf2b317a9220acc58ef`
- score hash: `sha256-5ba828aaa2831173d96295b873b515f4f4c0cf75c60fe99d46ffe3f79acb44f8`
- bundle hash: `sha256-09914d3542060820f72a037e4ff905eafcac9603e26ec3a5d413c86f31e61306`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3c9506ef693eb127f9b82369f4aa560da295df39e9e3ce7446b93c59e3f245e5 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-48838537634be42937137ab0a2f13e68a4ec9f8d37836994dcc30223b1faa7e6 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a791b6c13ef5f33bb80c7e9f026820a26b2a9814eb83a3f67bf17229d052fb49 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7c2c890f406b702714e7ba3ab1963160b9b8966cb3eb817edffc1bc552fe3803 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-e1b9e072 | sha256-eb0e97b9180cc2a1170f2ccdb4a8e2b7177924c4a0bf0c0c05e67b61be4877c4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-e1b9e072 | sha256-ab615c8d12f76ec069acdb4bce71750f22a9096468f69a595af0d06742636383 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e1b9e072 | sha256-eb0e97b9180cc2a1170f2ccdb4a8e2b7177924c4a0bf0c0c05e67b61be4877c4 |
