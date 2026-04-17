# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1ec62e8076ee4d1e710644be210d5ded13133f83ba7cc0a283a8ff2ec6e4b13a`
- fixture hash: `sha256-208011a3d49bd10b0f228ef3f15f5d25a591b8469fe6d29ce8deec0246fbbb48`
- score hash: `sha256-02ed5b6032920886d70b498046853dd3551251d31cf66d955601476cc6d45e21`
- bundle hash: `sha256-e73f694530564ef5a8c0f3c25b5201bfce6dd3c454a45eb70061752ef1ad430e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30241ba1cbd874d0509ab1e29b9c021ef1eb69d9f017747456f3594de63d356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c1c71d4acb7dcca7cdcea78e6f0253edf62ea2b8cda8edb379d22370b9112ce3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-613de25ebb4ea0b54de3e0617bd980012ed989049e2d57d87092e47157798d4e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-cb53fcdb0644e6dc4c25d327b957d630d1ddeaed26af32587b859ee445b57c10 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-be79da65 | sha256-0f33478c7f4a71b8b8965f521eb602abb5e605fb9973a49884373ab332b2a140 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-be79da65 | sha256-dfb5cf8287fa17befc8481f50c453c00836ab7c246f39d45f2d2531174c0ae03 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-809b2978 | sha256-7442314ca35faac1803837338bca4dc5f7139a085af08fef10b80abf88be5ed1 |
