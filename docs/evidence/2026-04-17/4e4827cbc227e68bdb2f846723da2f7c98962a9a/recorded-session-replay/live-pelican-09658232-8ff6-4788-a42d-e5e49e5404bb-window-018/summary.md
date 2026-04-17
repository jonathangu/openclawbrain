# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31afb2df9c1a17ca25197bd8dab4006e37b9c5d5cee2757703f3aa5a6af3cc63`
- fixture hash: `sha256-998f6618e36f06829cb18a9eae15dbb334b923e47c420cfa28a2642db4d68155`
- score hash: `sha256-e2f522ab849b482afd730145b80632f4c5f57c9aa60e6fa11d122791dcf56c89`
- bundle hash: `sha256-f48c10f59c17dfff8438e384a4b4581109193936e49a47e82245cf80de6e266b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c0657d7b41d16e0a76bed7b5e5dcdadf4310444b0556eb5e7411f6141dac5dd0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-126039d99fa7c4d50ce2c75b607e43af3043c883ca49735f2e0f2b681765e016 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1195e947d5d5571ddc80d9bb3dfce7b2c5b7a4d39aa0f0aed150e32543a26b5a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-943fd6d612885f4febdecedc45b63e1499807652f228916a7e904d3212816973 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2b0c7eed | sha256-d5154bdcca899b8828c8875b9a93e15b75ebf123323de657509b3338edf069be |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2b0c7eed | sha256-208d7e37f22ab98d4a61be523efbb1d58a866b2949c0e7613f475cc68cec3e1e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bcab9b94 | sha256-ae5fc46f88afa5f4af117aaa66b508cd8bffa33c6db99d7500ecdec01268bd8e |
