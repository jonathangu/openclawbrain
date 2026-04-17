# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d48635c1ffd88f3a117b615a76004d5e367f3ddb12c33db64c7bc064203d9b95`
- fixture hash: `sha256-8c0e50ffbe18960ebf818512a9b376865f8811b20166fbd695c968bf02943a6f`
- score hash: `sha256-edad5498eb9f8f94e1a40f855ab97b38068847bc44ea3e8ceea216d141cb3b21`
- bundle hash: `sha256-6fd702c0120cfc1aec91c1cdb8f0537f3389d18f8254521f919b16d82cd4229e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6f313e5fabd4dcdbe02c394910f74b656656e1556ed3e55025a3581d3065dfeb |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4d502d5f0d46ccc1ab19e8de58f816b79b7dd185bccb7f786bc41b8c58c2bba |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-efe5636857d7f39f774afe5a01e3b5a43cbb6e853fc18b43d92d53e77c0346aa |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-579ba3ce31b4e7a22303df662c8f33d1e90f6d43b4d0a63a03c46751782e61a9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-bb9a3ae5 | sha256-79a2e47b2d100f2d81f0e020722c9d2917334757fcaa0d541a2d7e97722f7e97 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-bb9a3ae5 | sha256-79a2e47b2d100f2d81f0e020722c9d2917334757fcaa0d541a2d7e97722f7e97 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-30b87ebe | sha256-3c11311dabf383e50469efad625275b3be94255fe8d01401370483633907708e |
