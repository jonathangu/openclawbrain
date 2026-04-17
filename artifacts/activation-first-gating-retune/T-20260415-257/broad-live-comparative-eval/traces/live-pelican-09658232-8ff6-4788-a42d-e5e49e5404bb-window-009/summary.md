# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f1078b1a70bcd22daa0ead376beedaa52bfe2cf8765ec6a491cb29b47f4429da`
- fixture hash: `sha256-48416b4518f830c212c5a38183605df066ce4a1235bd3582b824c27bcab21c53`
- score hash: `sha256-40625f2d35767f01172a58127bb271ffaa21ad2afbae244d2f424628a98dd0a3`
- bundle hash: `sha256-811c803213d0b151598cc19fb17014eab547a42612e3882cc02400c2b1c4c331`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-57e6cc1ff0fcf88903029010179cd9e85affa629951b704a6bd53f2a38e4810e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-30dbc029ce8d19c1430c94193154ca4fd95a47292045d96aa4aa3d20bd1e47a4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e5f2afc7e12076f5221afcb9e8bdcb1ca7a18e360abe0cc037b01397d20fdc96 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-963ca495f7a555a559efba807b043469eb164234e2659a6551eea25e0549b06e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a4b69353 | sha256-cefaca8c85f0a0965c3fcbdb05e5c5a14d8d75b57d70786bef6b69718d689a21 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a4b69353 | sha256-cefaca8c85f0a0965c3fcbdb05e5c5a14d8d75b57d70786bef6b69718d689a21 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d1712d3c | sha256-fa7b72f75a31ac4af369883341bc6f51dddc066428bab9c1ef18a9d700e0e961 |
