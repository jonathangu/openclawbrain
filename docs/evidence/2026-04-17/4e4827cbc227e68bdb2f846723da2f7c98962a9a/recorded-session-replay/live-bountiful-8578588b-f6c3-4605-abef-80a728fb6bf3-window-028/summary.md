# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e57ec68d935737d06aef21380bbe661fdacb97877b7cf7bc1a4c9589d64ac9ac`
- fixture hash: `sha256-2dd65164f5290b5ce39a35df1037a21b22351c6c824fd1679b3dae413abe2583`
- score hash: `sha256-e519a7a7b811da5edd97f87c948f80cbcbb7710e4ddb775b48b66eb5239d9f0a`
- bundle hash: `sha256-ad033b7edd25691b47e36421fc570abc1d5ed05237ce2a092fc7c7fa1e02c170`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d76563baad54aabf2052fe083a937d82e3fb9d0d74cdb60aea31ae988aaf3b8f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a4dcf3417ac7a8290377ffe1571db759b502160446fd8a676be50e4789707b15 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fb8655788d2b835d09ac8d714b4fb6a9b0a5b9a3d51886ad0ef2481282f284ad |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3d8c1b1a89394d867b74377e8be30ff0890784560e112b3444af319e399fe175 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a13e1d86 | sha256-d26ad61be5ceb64cbcb1d94e4b79b71bb3062dae21beff166be4ab2aaafe8a39 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a13e1d86 | sha256-3ce2ea5056abb9352e85d05162ab30670e77f345aed15015b13b5072aebd9d86 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-cf0a1907 | sha256-8dc0dcc00255f16ab871e210343b5a81af24830f3621e17704dc7546d3e7be8c |
