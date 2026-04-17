# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-24701235d9bef68e6850974201809e3a73463fe7ddfd0b5cfe74a867885dc71e`
- fixture hash: `sha256-7c9db0ae094c3de40db6d4e0f20c52b15a3dee97c3144a7a4c433e3dd89b20b6`
- score hash: `sha256-ca26af77da0592c75b05c825690b9477c6b4931755a92aac0770b6d48ab802d6`
- bundle hash: `sha256-d81dee9d993b39916695c3615e4e904b1bab8163641b8cb002bdf5c6a87f7faf`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0f3c8d6c272d7556d73fb57fae65bea8046db993f5ac8290705eae6ece09a508 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ed280e270b826acc2336fcc3bfcd0b29c41fbe33072e44a62b6fd0f711925ab1 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5c4e55cc45f73b3787fcc3d1d74605f59d0604998ffd3fe1097b2e34758f4b7 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-d099df5cd3da4c64017c1163142fcb146b0e7783f8a4ef5d4f84667383953868 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-70762779 | sha256-179a5ed4c48996630b14272e5972bd4b96635aa870760e08077d7b1baf0dbc4c |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-70762779 | sha256-6a2bb8f0d31e7a551bd5b815b8aed2b6f007e14ed1eb0f7f70442b4c2400b6ad |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-0dfcf72c | sha256-5c945663081d97d5c98d8fcb32c18fb3eb040a0c2be81d82e99ccd8cd04ea5d7 |
