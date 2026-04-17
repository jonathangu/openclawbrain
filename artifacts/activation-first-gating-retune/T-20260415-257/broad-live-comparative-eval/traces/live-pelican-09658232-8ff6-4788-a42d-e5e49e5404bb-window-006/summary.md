# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e1899273f160788957979a298d976827dfb7d2c8980b1c161a6e0c69b405f12f`
- fixture hash: `sha256-e3a4578dceff89673c40bbf12c9b294dd97be3ba2d82b9f266209970182a5648`
- score hash: `sha256-a6b4925a56f62e787f58925929680887b16a20491099858de69fb09b3c1d73b0`
- bundle hash: `sha256-439c072dcfcc86b1be8cf864e3e98d55737f0e2ddf069585b4bd4a6b695aeb46`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ecf5a06a6508fbef20c40ee36944ffad441534c7ec83a389bf5c81a0f73bcb66 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4006ce3d3eac281222bc8c26c52f294f4361fd4060abb29f86bf3cadfce6c33d |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e3a943c3c61e1439589cf8a8f84ad2d174c889f9a65d20ed7d732792286c54c7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b9a335261fe8514c2000eb7c626a9cf19de04d3d1b2dd2997f288906632a42eb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-68ba997d | sha256-9eb14cbf21a5956dec515d9cf8552e9000660129c2b2294fb2d09f496821b35d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-68ba997d | sha256-39f891b6269c62b7164f9fe65709163226959c04983fdb27c2ac9821bde02aee |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-689c3160 | sha256-b07d666a05b26d0460b719c930fe426a7e20f69ad8db43315926213672d38256 |
