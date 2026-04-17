# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-41fe4ec7878578e538ea87217d0f1ff26ff7c1c5009495fddc6dab6258a2bbbb`
- fixture hash: `sha256-49ec3fe73495575ef4a5edbb2b2c58d86b67a86df5a7ca6830045265a7717b0f`
- score hash: `sha256-9aafb9195cdc5f72b333b7e451e306e4ac513d548c0193bf6a7b14a43e8e4958`
- bundle hash: `sha256-898b4381099d6e7fd48d2187081df20a59392708382a5fb4a86037d29854ced0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d5a96d1404eefca9da775b3bd1f6864e8e794d06c6c90d16aa5e90455db3aa7d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1ab8d2520032f05e46e054fcf06c8517bdfb0e9175f12bc6ebe4ff0a8884d98b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-208b887f0b43b35039e2e7aa104a9abc7c9ce4d803ee0dd953eae0e498e112c0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-535a5922ff411cc49f05af5a1052c80e2d75beada861fe7067dcf6b4b9334da3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ac29ab8f | sha256-c6a1c36ec0c1e372cc92c90cd63ee8019fc209a71ca5eda1e614c992dab0ab08 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ac29ab8f | sha256-ba9920d19658b8d20ef0565ff3710e0e69c2db2f2687dd6e172869787a270434 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ec1f06a4 | sha256-18574a9b19058f7edcc6fbc80c0e2c5f182b4d20cd280ca0a57db9b815d800bc |
