# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-db3086ba9270f5f47434d4a3c708e73ff2624adb056b71992e75ebf839a91592`
- fixture hash: `sha256-2a8d321cab2bd435ac998d63d68b17b7fce95e9a0ea6d02ef75e09676d4240bd`
- score hash: `sha256-5b4ef8f274fd50a44a0d5af1f3087f233f5cae5ca63bb6fb5c60b1d254f9db51`
- bundle hash: `sha256-acd51dc3718975c652ad5bd8d81b8c5c9d943c805dc95f33473fcff2134a4221`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-da6633515ce74e28e9f8bbc2cc587b6b0548deffeab6470c77c67fc675828106 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b859a19f58e7963ead1fd054b6644e24ac93a10b2b7e574497f8e83259114c57 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7d5b403b1d18159eb4c99120a8e4ff97b5aa9058d5c7ae5606fb4d0349043c72 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-cb615ed98a3d4d794412c08a81690cd8d4eaaff47b4e4593392180ef56bdbde7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-13e03e8a | sha256-cbe78ce8a359be133b138d280a675b11e952517cd20dbbbab76474ff3f4a5404 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-13e03e8a | sha256-5ba813da8d26abe57b96d373c063f71e42b7450bb5160f81dbab95a87bcb940d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-31042a27 | sha256-42cc3bc3eee51f64d3673f79bcd4191bec1e981817c0fc14ba73de2e632c2c61 |
