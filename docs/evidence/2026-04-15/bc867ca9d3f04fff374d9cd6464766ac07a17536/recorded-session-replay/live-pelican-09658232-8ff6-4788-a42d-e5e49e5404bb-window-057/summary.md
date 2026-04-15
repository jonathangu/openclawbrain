# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-057`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2f2e67ba6e9f3ee34d9a729b960d4347b90c5776b36c8bb01215597777ac63b8`
- fixture hash: `sha256-31116913aa40fd67b6f1a05c1b62a0f72f8a386379a84cc5c256525c2b570370`
- score hash: `sha256-73e8fa1c45db83bdd2c58c62570dddb487da63cd963d464b8949863ccb5899a5`
- bundle hash: `sha256-a804c3efbebbd494618bb8e97ad5fbbc1be31b5bc4a622d1236b92dac4b9bb3d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b13aa42d069a6fbba4caba9f912ef9cadf19ea12093ab266f931b4282b9e22bf |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-44de8ebf066e42fc960eecf20873e5b8062ffb567af28205829e7d62d19b7561 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-44036dfb472ea2337ec481648c20483018819bc8096bd4a6484d1a71fa2ff982 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-8bb7614f969ee8ee7dd7035248ba4de57883e17a218daa2f3ce4417e67913e8a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c214fc4f | sha256-1908b291e9666b38e9920b5c02caf1e3876e43a0a56dd8e50bb2ef9d1765c53d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c214fc4f | sha256-ae54324745dd625392dcdd9ad169c077aca9ca514165b0d44514eaeafd76bb3b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-c214fc4f | sha256-1908b291e9666b38e9920b5c02caf1e3876e43a0a56dd8e50bb2ef9d1765c53d |
