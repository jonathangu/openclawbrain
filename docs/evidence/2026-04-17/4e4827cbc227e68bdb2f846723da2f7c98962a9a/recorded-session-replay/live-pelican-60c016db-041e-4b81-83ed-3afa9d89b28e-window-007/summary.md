# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bfcdb554e6c3bfe187f4c905f92a9b282d7821367cef535897c2815e123fe75d`
- fixture hash: `sha256-3907274214cdd60210f9dcb9d9b0e865d090d5365a59db918b98e4ad4849f4e5`
- score hash: `sha256-67406e36cbccc04dfb44541a587cc470698e8b6d5d36f21f8f748bdbbd5e1f13`
- bundle hash: `sha256-3d1f1dc381b1aa04c9c72f0aea4001f202c74a05cf93a808ad29eb4617472910`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-18ae191771eabc01fba0eef9c0e7f277194aa1ae188e2e94481f667ee00cc41c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f3d255b66100f9a62e64cdf7f9ebb592c0ead1394d7206a3f61d4809fc7db88 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ac3ac61037845ab1c04235a1079f7b0f55abba2d1717613fda28099c840ba114 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-a4fdcb1dec2adec5b04f81025d20be76502d4f46033be60931643af2d4eb4dd5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6cc7728 | sha256-e8401026b46708ab4f09c8a1d799e9354ee2269f0695c3c3705e56be19a35829 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d6cc7728 | sha256-97888f344dcd5dba64562b14edb2f0a9cdee761959cc2397efaa32128db56bb4 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a4b38eb1 | sha256-4a0f2b27b87ea7e95bccaeec140171ccd57d019aa2c37137b7e1fed1d42b7aae |
