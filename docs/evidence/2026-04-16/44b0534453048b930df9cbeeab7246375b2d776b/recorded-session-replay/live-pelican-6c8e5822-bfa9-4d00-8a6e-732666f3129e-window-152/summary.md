# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-152`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d734aa4f8ff91d619ea2bd69d87aefdd3f36d0cec38d3997b6f1c5ab56a102cd`
- fixture hash: `sha256-b5a8d59003130cacb6d12d20cb7f35591a0ecaec31de33844db54aba06f55180`
- score hash: `sha256-cb5c8627d9098f5cbe15721c21b291be79e433fd3e4d23ba08231274c7631cb9`
- bundle hash: `sha256-66e66eedce2feb480874bc6f9c60d2577354a6de5043b6a5bd81ce70121b62ef`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2c822d5a812142cdf3ff00336272092554327fa9d0fe665c2253ac281723c371 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3ec86a6f2adf7a74531f0ccb681789c1ac412f742b9262e8788b9b45ae8847c3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3f6adf1a22bf601615f149f429958f1b5801e28336dd1970971eae695028b632 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b51bc6bfe18e233da011e884db58aa78e3dca6422a6b7e35c9c282bdba0f0fee |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c15cc6f4 | sha256-b20842dbc8be2ac106516527472b86df49dc77fb9d7cf93bc5d7543c99c400e5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c15cc6f4 | sha256-55bcdb370cde2263696c7f8f17786852b32e3cbd46635d72c46215c34d18aa01 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c82e7ab9 | sha256-14bd9fb909ec0c918ed0510fe848fd90ce934020e7659f46fed8c1db495fad67 |
