# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ec605c7758b471d35c95979aeb2cdfe7a4674e948b05ffbdd6046eabf723431`
- fixture hash: `sha256-076f85a33a3de7d14b01739ce6654a252ec79b49aa247d0b8cb77da6c5a8a9ec`
- score hash: `sha256-f5a08005bb3067872e3c2fe5258c648e942e1eb926e71bdf9820061643db34e3`
- bundle hash: `sha256-5ce53081eb1184cabd132835ee8ad96eb3fcd9ac887e564800d24d013244d452`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b4e2e11e992d3b83a5df7a249ce0dd37bdac79f45db7926d41f83ea82d964f78 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-17a9fcbf32672465154c102eec3c065ff4e029938ef789bba9fd8a9e5789201f |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0517a78f2239719ffed96c9b4ddb66fe430baef840792963d39a7400f607c919 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-d2e76061058b73f8a5ba5d9920d00412a6f6b38c2af33e0fac9f51d17779ccd0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-fb0c3c44 | sha256-228d199853753dc9408d42440ed025de035d1ce9a2c2b1d9f6ce6446d0609bf0 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-fb0c3c44 | sha256-c5b37ad861153003145c63f41e29ee35ad33b462f93c8576b2b3b1b17834909b |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-4be175f3 | sha256-3e9ebce25ea6393efd0c02669cb71bf4f3d0b7e4d0c7ab003a5fb745add46b0a |
