# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ec605c7758b471d35c95979aeb2cdfe7a4674e948b05ffbdd6046eabf723431`
- fixture hash: `sha256-076f85a33a3de7d14b01739ce6654a252ec79b49aa247d0b8cb77da6c5a8a9ec`
- score hash: `sha256-64d0e1ef4b07584885d42843f62530872ce67cac6a8d6e1652318f2d31ed20da`
- bundle hash: `sha256-190725943fb38d919777f1ea983e618aeeaf3d9ece134b606de53dd2191c9a93`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4c3b50e16ae17d558532a27467bc1b14dcbca4d7917a5b3516748a073e962661 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5bc65379f7e61b7ea3e0dd5192001daf113c28c58b059d8f64ee41c6e5bb0e62 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-795e990fd6cfef67c5ca63bf0cff06ed7e4ba8793e44ac39382e67b1c36cf856 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f6d5b3e2 | sha256-7a517d5de7b3465f6b48146dda40a38e18b5edc9bee8be0ac4dbec5c4c49fcdc |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f6d5b3e2 | sha256-da98d7b174bdfa5aa5787563ea4d8471cb5e503dd0a3f627d3b44a942cc6c0c5 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-47aaed91 | sha256-6c635d3afd77f218d0f351c5da8d92583d97463fb7e059927102c31402073e55 |
