# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-235`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd603b557f17857772a54a908d5f0ba5df5b9405e501fb3b65a61cd496b30680`
- fixture hash: `sha256-a72ba01cb2a77634727f52aed3858de560e72d44f77e52442f91249de387c84b`
- score hash: `sha256-8013245842de663c41525a900da073ecde2deeaa9a23759f8c584c4aec06da2a`
- bundle hash: `sha256-94155f43586c2752f3438bceed36e5d6a6099f3e55bb8e061cb03ceddf801efa`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bf4126f9f6708337f7f0c62a2db19e988397589b870e571ff16dc3ae73782dd0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b8cb6ae92509dc01ab3b21d2566363acb857eeda5bf37e49ff26c63fc2e4ed1a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-527d51932f5ee67e411ed78298791828a49b35e315a7b73407d00274da586f2f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8ba587983e17c650eb4aa715a6e8f0f53176092aa79d2a6d6c4c22c8f31c0f5c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9103799 | sha256-35dd622ea4c9b76ac672ff2862a08a511a546f25d3506dfd55cd9e19f62c09c2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e9103799 | sha256-979c91e777fac2605e0a83bf8e2656d31bdeba84cccc2107fb4d59142d827df2 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a8fb8ecc | sha256-aec3c1c9d15dd507feeab1e16f402846151bd15e64254ba787bbf57ea96518c6 |
