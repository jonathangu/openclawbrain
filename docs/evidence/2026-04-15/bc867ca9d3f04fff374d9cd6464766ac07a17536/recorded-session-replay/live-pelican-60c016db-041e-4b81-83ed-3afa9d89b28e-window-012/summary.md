# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37ca78c4f79af1f5ccebd457f09d9bd9f0b270ac0d1c7dc3ef10aba20d199a04`
- fixture hash: `sha256-221b36f5e3c3b83dde39237b8133ec3e68acdd74bce0b4e3672a3fac84a8cce9`
- score hash: `sha256-3272c43394174db5a47647115186a02c2313c3b001277587a4a38eac087e8413`
- bundle hash: `sha256-5ffd189f47f1d78dc9e3d527fd527002b7208f85b17aca019d6d8846aff6c417`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fa6a30b0b756b7163e1ab0f1526218df1fd81b134bd908830d7627bb5155f717 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5135588154c225d8ee9d16a0c1cacfb20249c9a6b6973658c5d2b4bf0c0f6e0a |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-444c19cfa79d95b3c55e0f6c746db885a94c9c86663d377e9657d2706ab1977f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fda6e6dea0152e112fc473e30b7fe1426157f37bfded1eec9d78e2f075bcf3ac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f1d7aee | sha256-ad49e2f9394d50f5ba5bce99b7a1c65f5cc480437a1860ecb5453015266d14c2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-0f1d7aee | sha256-530e1a4bf441b5d73bc3f8399a540ad2434bff46e9f4f451b9c19f0f8ee5159e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0f1d7aee | sha256-ad49e2f9394d50f5ba5bce99b7a1c65f5cc480437a1860ecb5453015266d14c2 |
