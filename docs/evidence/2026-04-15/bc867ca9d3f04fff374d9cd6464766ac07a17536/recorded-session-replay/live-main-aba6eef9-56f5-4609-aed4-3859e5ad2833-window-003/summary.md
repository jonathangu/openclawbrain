# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-003`
- winner mode: `learned_route`
- trace hash: `sha256-7206817fbe9864fa741e2aac4263783623734861273e6c92294a7e71e4bda31f`
- fixture hash: `sha256-3fcf85ac262f6dca9a6b48603643e7ca5bbe3663229b7fc7238b9b7fb3303591`
- score hash: `sha256-a85684f5af4688f93ba4f9fa772b5909bd0ba8f79368219433cd29c14920942f`
- bundle hash: `sha256-5fb9fef27d234501ba629153d9ec71d03d8ff61240400293aff1f6b2dedfc42b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4a6acf0ea4807b1384f37283996c98fb6c5d3cf32e52bbe94b1a201a85fdc539 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-07740041ad3a6c3119210736bc76ac586a5fc7f7a534ca6d763fd02c784019db |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a2a45ba25544b503605f645bfb56e766711854a73895fa940dd7cdcae589c2fa |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-7f573ab95cf7a57cc9fba7114a9d928f0d3e255f377af57760051bb4d82f69f4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-ffe177c8 | sha256-c6f4d77b2d7fb2ca0c60cc218dbbbb31b1588eb65158f2fdd9edfeb36b6afb56 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ffe177c8 | sha256-5eb946d44cb4e3f649ede6961150afc13234e51609e7e618b51b7767a3b0498f |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-ffe177c8 | sha256-c6f4d77b2d7fb2ca0c60cc218dbbbb31b1588eb65158f2fdd9edfeb36b6afb56 |
