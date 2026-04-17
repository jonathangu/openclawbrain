# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f8ed2037bc2ff0feca7432af422fbb2b58a869f3969fbcd41ad42699329bf723`
- fixture hash: `sha256-b54fc3ca6fe17a912f89c0806fd3df709e1f6f80264d7323adb898abecd00677`
- score hash: `sha256-2970b812a99512fec332dc64d8f3df99abc934ab8c7cd1d08a66513f0674b4dd`
- bundle hash: `sha256-ad00a654a276a7a91955a5bad9ba0a6245292541fd5a2177bf0ba8d333a8f0a9`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-34738adcfd10c341d46efa61990a3844e1795a8fb18b5ebdc9694342a06a5142 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6fe988b1a461b982eead2944218b6e9e487039361a0c0545ea516a4dc85f66c6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-413eb8dca3017f6a7cc5025adfbcc6e3308773fd0252fa3072a99426721fe773 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ddbe11c5a1c0958917c4818f642c43cfbd1a3e04321d700d57e3b7c8241a5b4b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b31a0c37 | sha256-bd1149eb7c3da41e3aaef0e942bf2b56fb9e62fd32f58989717afe90e967f895 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b31a0c37 | sha256-920c753513f26bb625456712a0113bbb2373edd5eb2c22c20521af1cd0e201ae |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-ae095b48 | sha256-4230d89698a60d18196468893fdd0c01043d6887620f5d40aebe60709259e259 |
