# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ec605c7758b471d35c95979aeb2cdfe7a4674e948b05ffbdd6046eabf723431`
- fixture hash: `sha256-076f85a33a3de7d14b01739ce6654a252ec79b49aa247d0b8cb77da6c5a8a9ec`
- score hash: `sha256-1047144e35d9c0ec75da8ddd5f148dd9d5fe800f2a7fd84f0f7bf256507772a1`
- bundle hash: `sha256-22056cbc6a16faf537d4dd40f02968c4f17407125695f159739b6d90c5065a71`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b4e2e11e992d3b83a5df7a249ce0dd37bdac79f45db7926d41f83ea82d964f78 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5a5a0d85364361352bae87e3af0a14574ca10c5a9c116f73a3ed0cf8ee06f1f7 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5414dda11b7f121df585aaafa81712d72c46fb662765f0c1f38551e682211419 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-e0faf92927b7c3bca8b5d68844f40a81312dfc514f06c4055edd886c792d063c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3efcdd3f | sha256-882d943c42c7942d4fdd143a9d44a0a15a30666c911d4105603387cafd5e0068 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3efcdd3f | sha256-3296a9a21d7aa5318e6e1687cd919e8e513e52505d6710fa6713aa2421369943 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3efcdd3f | sha256-882d943c42c7942d4fdd143a9d44a0a15a30666c911d4105603387cafd5e0068 |
