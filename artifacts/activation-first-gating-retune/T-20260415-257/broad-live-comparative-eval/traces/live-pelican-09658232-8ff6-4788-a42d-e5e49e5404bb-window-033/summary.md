# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-033`
- winner mode: `graph_prior_only`
- trace hash: `sha256-694cf444538867e625d49591796eda7824a3f9914c6d50782ffa8d2751091f0e`
- fixture hash: `sha256-cdb2b18e3a901c8928c86a3e5d6789c9de0d594dce56653b0cb654624b8e744f`
- score hash: `sha256-b10ef6fc4e28264a29191cd49390aa84256732228ba6a717baf38215d705beca`
- bundle hash: `sha256-910f1acf497835d345930ab9adcad428892a8a40ff4e59ffaf1072b20b06b42c`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5406b4db2619fd299a4dff36fb17ece03d149828bde2ae07870bc2e0cc31ba06 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b85e5ea638d18cdc1972ff2c629db76606d1a9a40ab4e3131aa3a2c1897ecbe |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-2196246237503424b6668e7a203a0da2a72f06d6b4a3a8dab926b06b1cfb3bc3 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-77f3cf85a1cd88281610bbcb3b8847ce29aef423eb9799771b7fbf420f7db7f4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e00191bf | sha256-ca057e46431abb2ed84fde5fc42cf7a5baf89d7eed739bf6fe59067e62358f40 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e00191bf | sha256-77a404a3d8491f296958012ab81dea305b1fbec4e978aa757a447615ff73e7b6 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-ddbc949a | sha256-8898c66b3655432af0a7a9241a6a4a45876da0113124edf3ebcabfccce883d6c |
