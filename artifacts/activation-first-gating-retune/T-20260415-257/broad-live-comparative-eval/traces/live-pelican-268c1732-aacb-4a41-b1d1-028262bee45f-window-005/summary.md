# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1d9740289fc2adbace7590e78dff24d1d94c6a419d6474e3af27754996da05a`
- fixture hash: `sha256-b9420b72d3a2c2c9c62adbc0b7f3ef24407bf200cf73b9c382cce44e2d33fe6a`
- score hash: `sha256-76444a4dd4803a7a505447038954357e268556ff777b6a210159bbb3c67af331`
- bundle hash: `sha256-896e88ae86cf862035c54bad9ba4fde784a1810db1b6bdd088b0a5560d7e1b10`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a36b1da37ad1b5e7a8a6bf9a89b082e6da9affb9cefe62c4630aaf0bc52cbd76 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8226d764c5888ed93b25fe0e01a46b7bfd3861bcea96143ec505b13fd88bd3bd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7037b19779d964d6c5c59a1ebe642e1e5eb41f992899cf4478ede91d64bd2b6a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b0f2532ca0d32106b7b01cbe851536c83f74e0b9091ff6d64abf59771ef3b7de |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-92c674c5 | sha256-f332d99c4fc2ddee13c9c079a33d2bbaa00245abf31d1b0efbea52956ca18f71 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-92c674c5 | sha256-c7057afa679e68ba9048cf6a8076189247c069f0adc51c76c246f3bd7da903ae |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8242102c | sha256-1a347d43806935e1cbce7e6e928f092a1d0b8e542e5c9263b81a363ea153a7bc |
