# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e89d5a5-376f-4b67-9e88-3aea314ab451-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-11f89d0770e58c74a32e0ac08329b409440ac220cb647ec446567aadc15cbdd6`
- fixture hash: `sha256-7793c2d77fac055a1c7c47c9d026a76a01511a45ccb17bbe5db49943de3d0ea4`
- score hash: `sha256-a062c91e7ccc3ed157277d2e6022453aea63c5ba402393315a4d28cf0a2815bd`
- bundle hash: `sha256-95b69f06ceed7a32b03a884b8c2cf2a8e3b5cfc047db8c9b627ab375d8e77b08`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3297dca5d83084645cc80493377a366cf545c5142415159b972c4f8430720ab7 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b113962549accd401a2a87dca5caa10561a0ce15ba1671e050992d70ea43068e |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-19b76c85cec62683b61a6118af03067496ef9a8ff9c22719783934de228a8af1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7f200f64782e39fc34c0aac90c02b5429a759c252916f35ecd3196fdef509bd6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5c3ee43b | sha256-a738c3835fa9f378a39fd42ef132ec4a7da10bdc9fb241b49959dc389cdc6ed4 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5c3ee43b | sha256-cdebbf97e9ba9a61554ea1cd5a9c00d7eee015ca0a37ca51c975f75dae2ce691 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b7afca54 | sha256-2e3f2bb55661763262a7b6e3d76e3ad6097dd2578f8e8f7d5aa91aacc5ad48aa |
