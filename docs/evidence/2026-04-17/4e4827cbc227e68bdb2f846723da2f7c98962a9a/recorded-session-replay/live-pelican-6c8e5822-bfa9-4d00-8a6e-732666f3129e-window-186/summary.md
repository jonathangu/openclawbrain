# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e334b58e5431d3b20f7572c904faed7d64f26bc6fd3cb1bf1d055e492134e8a8`
- fixture hash: `sha256-8e788213c51f0225abe30e2600382afc50022c57de7f08753d94aa61dd287dae`
- score hash: `sha256-e63db84355802522578a5dc8d267c3dc55e2e3ba3d7368956b98183fa6ab47a2`
- bundle hash: `sha256-7d6dd5c1c2b3a885e4c157a4b70923183c460efd67316dc770ea4e1401582372`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f4621a0c949a3fba62d418ef21dd1d6c65fb58e546b35333db0f8e5c2c8785a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9af409008e98badecb836499a738b667488dc2cc0f79638718419e86b7e4a4a8 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7c4b032bb6f5742e9e14ea2c227040362691757a95a760a125866069f269aa2e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-1909bc904ecd664a4c40dc8b1ba16395074a947578538c323abab1c82cb655c4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6e0ab4ed | sha256-b0db82e389e8ad3243060f4fbc75f11da9d03efb0097fedf46678f603deffbc7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-6e0ab4ed | sha256-4b55fe190e7ab00a785d8e7e8a8e1d298b0fdc98394150c5e22c36addd0beead |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-9963d73a | sha256-d5178c512496f3605470b3e84ef2f46614c74984a80694fcd822827a2a5ecab8 |
