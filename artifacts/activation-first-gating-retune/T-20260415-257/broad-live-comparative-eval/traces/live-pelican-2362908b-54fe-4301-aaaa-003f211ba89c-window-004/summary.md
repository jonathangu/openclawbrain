# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ae1abddec00632179423e5d665c773fa81ea75d92b306fc15251840d9f53ec48`
- fixture hash: `sha256-c2c90149661c99c58bd2b000a17d70b99f16ed3daba941c64a7e5c1b67ab99b9`
- score hash: `sha256-7e979884cb277e4d8eea2e9adbaf03250227724dac581d0c6caeca06f76613b8`
- bundle hash: `sha256-2fb73afaf75d56a8a61ece47b877043423cfbc9c3d788438a6b55a7049c27af8`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-83c42e700005538dba5b3a6d69c6c5e443ab91af8b598837eb4ca6b5f8135237 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f3d0e7a875bf8da667da716217e49d2874b1c39d4639430286b4b39bc91122fe |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-482a21200e32ae05ff852a26c5f1b182e2516b37c4f9ef95519156a700a8d7ee |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-3ff62572fa0e4d93e0f303cb3099be8387380754883ed5635d9d31d98818626b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-385390dc | sha256-b3b4d03fab2fb2e80cf5701254f67ce989bd0800e1ea3de193c689a38b6a3b09 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-385390dc | sha256-66301cd9e210689a068e35cbd98c145f61e878b0c325aabb648fe3cd4bfc0aad |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-9964434d | sha256-205ef7dc3c0a5551d6d8893a77243755067653843bbb82e457577fb3bca533a7 |
