# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd603b557f17857772a54a908d5f0ba5df5b9405e501fb3b65a61cd496b30680`
- fixture hash: `sha256-a72ba01cb2a77634727f52aed3858de560e72d44f77e52442f91249de387c84b`
- score hash: `sha256-b06587fb491d27192debaf06cee708073a6754558eb642d6d4db9762eb44f7d5`
- bundle hash: `sha256-18db4531f846ce369fafc66e57e4bec0f468bc312d21892a6fb1ddd1025e57b2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf4126f9f6708337f7f0c62a2db19e988397589b870e571ff16dc3ae73782dd0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-48767cf10cba67280d7db2b79f8cb83fee94404cebd4fdc83d46f7acca85b7ab |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8797bff2d0448bf0be9c14f46a914115e23db0c4b9bb2c7268f35abd6f368fc2 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-2564545007f525f943d56f5e9441ac5f13d719c33079c67ac514a5a154a47c02 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-090f9c37 | sha256-f7ea5ba7029a5f700e786b0d44a58edd9ceccf9fb4170479904065a83aa94cb5 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-090f9c37 | sha256-16a85c257884c49d4f21c1d6002aa14dd1be286b4726a70ad29f8a9f8ce6533d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c8faf36a | sha256-43a8e4bbd8cdd840e04b8f73e5bb00fe524537639dd6ca6f72019af330eded34 |
