# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e57ec68d935737d06aef21380bbe661fdacb97877b7cf7bc1a4c9589d64ac9ac`
- fixture hash: `sha256-2dd65164f5290b5ce39a35df1037a21b22351c6c824fd1679b3dae413abe2583`
- score hash: `sha256-4a369615bec66d899f37921ff463c50ae88388db84ad3cd9d202f074d50f5c4f`
- bundle hash: `sha256-6620936aa966839d127261f86f6663ed4b6cd272f287e5de55156b4ab5d339ae`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d76563baad54aabf2052fe083a937d82e3fb9d0d74cdb60aea31ae988aaf3b8f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a4dcf3417ac7a8290377ffe1571db759b502160446fd8a676be50e4789707b15 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fb8655788d2b835d09ac8d714b4fb6a9b0a5b9a3d51886ad0ef2481282f284ad |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b7566ecd89e9f45146769de3c9ece75cb6999ef43a18e0b0801e4cda6d9b7120 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a13e1d86 | sha256-d26ad61be5ceb64cbcb1d94e4b79b71bb3062dae21beff166be4ab2aaafe8a39 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a13e1d86 | sha256-3ce2ea5056abb9352e85d05162ab30670e77f345aed15015b13b5072aebd9d86 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-cf0a1907 | sha256-bac893f37cd41a455401bea13f1089c345ee04182642b1d05b65fb3de2a0c30a |
