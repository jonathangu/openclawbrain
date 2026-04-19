# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033`
- winner mode: `graph_prior_only`
- trace hash: `sha256-694cf444538867e625d49591796eda7824a3f9914c6d50782ffa8d2751091f0e`
- fixture hash: `sha256-cdb2b18e3a901c8928c86a3e5d6789c9de0d594dce56653b0cb654624b8e744f`
- score hash: `sha256-c9e18e2068ec8491ba486eed2978071edb10abe02e7b0c923742271fd0eed358`
- bundle hash: `sha256-4053587c326f651168a87507286fa63d4da072d771de2db22ae74feaa19c2b42`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5406b4db2619fd299a4dff36fb17ece03d149828bde2ae07870bc2e0cc31ba06 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-1124c634b1fe91ef8c067738d9dc8ae07ec2b5cb7c2db57b927eb29fb108a8f8 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-e2f45ca0f3b3a225fd7e0adcc2bc1ba3e394c2d7f86840bc50c88709b05a64e0 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-f43137a1f2db0c77707a47beca55bcb547c64a4d67aa10f1dcb904e7d9a835ef |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-31ad8955 | sha256-c76a31e80eba979e01196274eb08bd2492fc9457b76ae5c9d960edd78b788da8 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-31ad8955 | sha256-de043a482cd0953571c20c537d5e2c287608bc54c89f459747829d56550f7944 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-31ad8955 | sha256-fd2a74d71d4654d94509247d0533571c9633e3bf12fb3cddd91db0c77550b811 |
