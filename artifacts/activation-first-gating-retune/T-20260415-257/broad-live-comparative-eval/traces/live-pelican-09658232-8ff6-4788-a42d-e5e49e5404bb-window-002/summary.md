# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-002`
- winner mode: `vector_only`
- trace hash: `sha256-823695d70f7872b1ae9eafb6d1d27250c7a30f3c8da0fb3fac149eb03366ef43`
- fixture hash: `sha256-bd84df8e56b4c53a26fb492fdea7511a22aab4ac1b787c58633c40d2b1aa4455`
- score hash: `sha256-5990c8cbb76b21a0a2584b7a3dba5528847d03c38a80d6694872bbf1a70754ca`
- bundle hash: `sha256-1747f336da2065cb911d6bd57f2b1ee34b65b8aee96a1f8e699f34d78c9c908d`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | vector_only | 70 |
| 2 | graph_prior_only | 40 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 1/8
- phrase hit rate: 0.125

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a1db4bbe90ab058f57bc7ae6a54f5aaf2daac0fc5ad242f5b0e6f3a965eb8e61 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-ae637e4c10b377c5f13202674ba19620fb25f60aa57c968a376df0408ea5d8d4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-819c2ce63044d4decb13705514cd54b79ba440be8973257b566cba91800b5ab8 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c74183ed6431633fd5b3742bc976a0b8fa151b3f69724054c45c68209ed2e5c9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-e4bf819b | sha256-7c703ff68be216e3b8ef2df95472bada00a10b291455345373f7b7fb8ea5700a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e4bf819b | sha256-6e6be7530c346fe6f02859fd4c043a5b2cfa253fc78ca0125321af0018447771 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-6e3246d4 | sha256-bf04aa2bd6ef712da399b20eea9e9f91799185868d4cf48863e7f31fc6f6a601 |
