# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-858b0f43ef470e5eca2afe1da13983b8601f538afa969d1fec1c6c995e06b43b`
- fixture hash: `sha256-5c10bbce6c643d206da3406e24da637074a598036c8915b8615377d8cba78cd2`
- score hash: `sha256-5e9a8cbdd4b95aa97e5d86b56a350ed52632ad357ce7d46538a2c365037922b1`
- bundle hash: `sha256-261c0d4e7beede52b0c980b5752e062528867d8f4ceb48965bf09898dba631ed`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17f51076699069dddd3d95709feed654148e93f103c223926ea5c4a4e2da537b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3abc96b71c7dc9a27740434e18e1912df5d06b80850c2f42aa8fdbab35e5cf44 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e45ddd5dc12734f216033edbf86ab268f33a490a50047164da6c43c06f99cfbf |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-6121d05d9789edbf138e4ec3aedc5f57b37b554aa1ba9e045a0fe5d02e29fb25 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0eae340c | sha256-7b8c8fb50e8caa897bb5e54c45f89731a89bd700ce7d01aec882de9164743a84 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0eae340c | sha256-49d4074a660e23551787f69353133b7ad378ca52f1e95093e02306c340fb81af |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cf4ad0e1 | sha256-0bd4ca3108f96b6aaf2137d2a16816ac7a2345c2d884cca97c50a0bdbf802707 |
