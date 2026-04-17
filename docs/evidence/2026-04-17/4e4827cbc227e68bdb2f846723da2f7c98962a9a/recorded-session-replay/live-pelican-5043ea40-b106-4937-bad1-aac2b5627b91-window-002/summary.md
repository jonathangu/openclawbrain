# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-347a801443da9e3d23f8dc976f3d286dbcc3cafa0984aebf1f93ff8efbfd1773`
- fixture hash: `sha256-3e9f54e7049625692dd39972563612e44cc8adf4a2a27dc80d450c5621a5caf7`
- score hash: `sha256-3bfdf3e9e469c8894bf4888a732cfc018ef1c37eaf99355ef1b2767356d430d9`
- bundle hash: `sha256-b8e824a19859995041b3dbc7f432691a665523efd28c1aa07be804206e80d364`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bc058e6f191036e6bf4f3884982c6a502fc3d927441bbbd1c5d745ba4e254aee |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-aebb3f8a8fb8fd281fa0c67c66461cb192dcc301f61927ce91afc48a359a2532 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b6cb5a1ecd4782b4eb1b5ce70876438aeb22675156a29ee2e0234da015d68627 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-072c5303cfb515a8cc730ea76c05dd8412a3a192f4199892557aa0bb2d4681c8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-40c579e8 | sha256-ae7d42c800c5afc145ce13bd7e52542595fb260053383decbf5b610807e1fff2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-40c579e8 | sha256-ae7d42c800c5afc145ce13bd7e52542595fb260053383decbf5b610807e1fff2 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-fc1af715 | sha256-dcb9c0dc2f8399d443b2b0ab1c3bb96831e718342059a41f66417becb40939fe |
