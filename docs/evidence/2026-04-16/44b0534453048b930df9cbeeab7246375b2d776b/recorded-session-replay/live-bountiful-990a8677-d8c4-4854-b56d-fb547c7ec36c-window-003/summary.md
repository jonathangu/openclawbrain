# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-990a8677-d8c4-4854-b56d-fb547c7ec36c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bf37fe09634fbaed69f393758385015698c30e4ffe16d85a6ab728cb7cfe25b6`
- fixture hash: `sha256-dc83b67fd93a911909b6e6a0822040e20903fda7a3d9b344617db1a16b36190b`
- score hash: `sha256-81349507f4b05d8deabb64cde242976ba41bc7a07bbe6b9ad20e0edeccc707ad`
- bundle hash: `sha256-646e5bcc728ae20e7fe6a6f9a7d6ba8a8cdbaeba6eb6cb572783dbb31d8ac017`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6a3c5859c5aa675b38ed66866de5ac4f6b502c35d08a72874cf67deb2a63be26 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0461d9f914c59e449c92edfec1ea8199486bae7d9a5485179640972d6f3e2d4f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b74cd4ce356c81969e91fb57a36a89507975bb240d81abc70688073e5007920a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-316aac22e4912f83a82bbf62a8da2270e68c1ff90709e823ff2b9c0ea7ebde39 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-25ff8add | sha256-bcc618dea6e1b7555827b4e7a550ecf0bb4a719fb45cd604b4f60563f707f85a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-25ff8add | sha256-6a9cca4482ee6b19db38b427a559614258502da76e468829e30d1e5338ca2735 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-27b39f58 | sha256-612280df4d8b2f2eb0f2edd7eb2b741c2a01ba1a90fa787c420aebd96b2c2ec2 |
