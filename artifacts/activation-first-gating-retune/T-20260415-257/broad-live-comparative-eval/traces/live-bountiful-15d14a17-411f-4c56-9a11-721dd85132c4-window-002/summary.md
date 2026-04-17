# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-15d14a17-411f-4c56-9a11-721dd85132c4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d48635c1ffd88f3a117b615a76004d5e367f3ddb12c33db64c7bc064203d9b95`
- fixture hash: `sha256-8c0e50ffbe18960ebf818512a9b376865f8811b20166fbd695c968bf02943a6f`
- score hash: `sha256-15cf5d5481a758571645807b5b9417afa7001a5b6a6d49fc35d7bb047f0274d4`
- bundle hash: `sha256-6ab6c6580b36ffa615028aa091b2fef3c021f0bc68684793546913cba670a9bb`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6f313e5fabd4dcdbe02c394910f74b656656e1556ed3e55025a3581d3065dfeb |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4d502d5f0d46ccc1ab19e8de58f816b79b7dd185bccb7f786bc41b8c58c2bba |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-efe5636857d7f39f774afe5a01e3b5a43cbb6e853fc18b43d92d53e77c0346aa |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-00a1e29aa90dd20877bd6ea3fad9acb3288b428990bafa52e8ae0f8a511f12da |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-bb9a3ae5 | sha256-79a2e47b2d100f2d81f0e020722c9d2917334757fcaa0d541a2d7e97722f7e97 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-bb9a3ae5 | sha256-79a2e47b2d100f2d81f0e020722c9d2917334757fcaa0d541a2d7e97722f7e97 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-30b87ebe | sha256-daf61d4dbdccbfb78d1258a1f978ab2a7486c7b6ce3b4674fcd4a27c262a6bcc |
