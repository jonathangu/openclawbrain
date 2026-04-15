# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b236d1de0cb5cdb5c6745d8cf8165eea05f61fe4f12fe91030959e1a1f0a9ef6`
- fixture hash: `sha256-2918c2c3bf776980cb54310652408dcd4b80904c74dc802c02149421011a5050`
- score hash: `sha256-46db2aaf17c85b3ab03abce6d4a85704060090b24f2e9fb0f4884919aba45216`
- bundle hash: `sha256-e242d44d8c58dbd6e1530265d4d64c0129211220a4c995c98bba0d6cf0eb8f1e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b752caeb990c3acabff60b3183401c5659a9fe06fb13d30bacaaff23a3d4f453 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8b632325eea6a1b6c95c2437b69dee2779986e6e0890de3a5c5a8492d5ef5c5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8a40bc776acc668bc52e9d947f7b716e00686ef8ab5cc21502e5611a8aa2cade |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-483fdecb444182ab2fbd32452790c659021cea5c1a0c0e0a5470186929c7fef4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-290ac115 | sha256-b196bbfcca497e706aeeab58655ca8fc8e2841a7bd46cc45dd0f029e12da3560 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-290ac115 | sha256-613b52e9cf1272486e4ea886f8f231f895887bc2683a9374f2f4d639d37b7b20 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-290ac115 | sha256-b196bbfcca497e706aeeab58655ca8fc8e2841a7bd46cc45dd0f029e12da3560 |
