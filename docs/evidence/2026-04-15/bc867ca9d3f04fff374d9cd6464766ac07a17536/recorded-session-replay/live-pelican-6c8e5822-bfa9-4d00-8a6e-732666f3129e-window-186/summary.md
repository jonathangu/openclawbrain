# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e334b58e5431d3b20f7572c904faed7d64f26bc6fd3cb1bf1d055e492134e8a8`
- fixture hash: `sha256-8e788213c51f0225abe30e2600382afc50022c57de7f08753d94aa61dd287dae`
- score hash: `sha256-b09551cac3c6455fdee76dbc53564d4e66a56e53b7f3655ce4a4a6fd2d098772`
- bundle hash: `sha256-7f6dc261edb633bad697667e7cbea235128953d54639cbb7b48e752026e9c385`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f4621a0c949a3fba62d418ef21dd1d6c65fb58e546b35333db0f8e5c2c8785a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e802ee9ee47ad880d698ed243e914fd306319aac08e883223d5d3c8e32f3332c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-85d1d78abb7852f43cb81c2ab660faabfdaf0ee50964de5fec892cfa9d890c23 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-348c8d40b37e9c33281d7f017d91dc0632e38a5b6e9bfea36a82b96152ce2878 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-28cf2a0e | sha256-817e1674eb37e4117df0384d48a18934deaad1186e9e154d456c21dc52695541 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-28cf2a0e | sha256-d62e528fa56853b085634e8901bdf02aad09e6f8436c1272edfb327ad158f763 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-28cf2a0e | sha256-817e1674eb37e4117df0384d48a18934deaad1186e9e154d456c21dc52695541 |
