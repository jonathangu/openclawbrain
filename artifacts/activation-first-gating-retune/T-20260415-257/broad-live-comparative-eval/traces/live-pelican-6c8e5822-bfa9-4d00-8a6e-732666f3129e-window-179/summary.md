# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eebf5a156737b8a1b13583833520fd34225ae0f30b4afb05ce10671f54ea2108`
- fixture hash: `sha256-63d1e3dd69e143127b58a78b17f85b8f588fdddd25950ce30e59877032c4d44a`
- score hash: `sha256-81671f47ebaf2dc4a13f1b43fefb26cf5e09d7bf27e08a9b95f95ae0d48fb7d1`
- bundle hash: `sha256-802fd0cea9347018f45b2896d1a9873ae1b380e0f3057d52c48a4615d3f26cb5`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-798f7833fec1b062f8a2789c97b9a979ee4a90e5e78bf32289929e17459a82fb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d50d7cea1a90cbf23d49d4c68990ef06de956ee21ae54d214ada37c795482925 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c35bfc2054a4fb7613ac147ef453062712d1c8438699590973997e96bffb51fc |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-b137c52adccda421b1c37e5c5e3e6f8e07f680b64da02a9a3c1c89f03ae76da1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2afc3f67 | sha256-ee3ea6da4628ba580de1c762f5c118dc6fc22dcfde4fc3b96566ecdf498522c0 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2afc3f67 | sha256-2419847a7d64ecb1a22117d9fcf9aabbc742e7e1c0b1707854bf869b9b7485e8 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d12aed9c | sha256-fd0e31e6678917eae1d624bbb1cf66355748ff309223d36430e4809aa5edf06e |
