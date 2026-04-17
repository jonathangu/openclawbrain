# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3f86fd026217c7d6458e87e96268ca58f7633ecf498ef1f8793a6a7617c13f8`
- fixture hash: `sha256-c25bf3a6bec00b35ab13366d1787d21cc5e0fb28011aa90689176fbd43238498`
- score hash: `sha256-f45d84dc04076e32290a4c48f1d66555935be04b1bd1c8ab33b454c23f0d67a1`
- bundle hash: `sha256-64a6b7a8ca5cb598cf0798666dde37dc412cfd81f8136fba3527cc196a7873e3`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1e36e95d3b902dbb1cba84b7196a751790c689dc2e631e7340724bc6d85c3a59 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bdc1138759222e8dfad805607e78a1baf5cde504f907ccf34c9e9bbbce043879 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3aa0009a221ace8fad271f46c61b082025aa09143b3006760cb57099979099a7 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-cd0a15af39e41fe6be0fe6a241b4d172117627c1ca2d5b13efb78cfc618fc372 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-791c276c | sha256-8bc7116228d028dd824a43228c80bef376ccd7e9bf9fc2b37b13258aec9df146 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-791c276c | sha256-626bff7d885cf1817a2b0411b89ed101d6c771b326d2e329e4a3631063473d00 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-f5cc4b11 | sha256-429673d96c16d71cc831218a66e32c946bd11f69e5cd21e4f958dd694a539720 |
