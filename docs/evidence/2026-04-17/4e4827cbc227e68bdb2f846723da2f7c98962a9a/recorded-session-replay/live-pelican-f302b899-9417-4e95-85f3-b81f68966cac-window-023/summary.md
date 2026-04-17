# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-858b0f43ef470e5eca2afe1da13983b8601f538afa969d1fec1c6c995e06b43b`
- fixture hash: `sha256-5c10bbce6c643d206da3406e24da637074a598036c8915b8615377d8cba78cd2`
- score hash: `sha256-cb04c7e3ff5932c02a7554b6255e79c76fa5037b205e87ccc112bfe3a7172722`
- bundle hash: `sha256-b3990064848a0afcf209c1f939b3732a8dc3ba5f436b1197ea14e92e72a1a199`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-17f51076699069dddd3d95709feed654148e93f103c223926ea5c4a4e2da537b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-49d4883f006b171a5c8602ce621f48638844fe15550da1d8ce4af5c4699c8736 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e46217b6eed00778401b3093b335acad00a2abc2a38a4b48881aa2561f9d3c6f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-5e8a4e6e8fd5482bb66feb2b3aabaf277378913fe5fae5fc457d80eb44a15847 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ead98aa | sha256-0688ec8157efa38b6a3949c7c4c05d9b1978bb332613016762381bb0030b0a70 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ead98aa | sha256-1ea784e737ad491df9ba179be05b8930c89c28fd8041bf34c5c75d6961b2f57c |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-ef4a357f | sha256-512a1453c4955ac2f8db405ecd042ba0c60561424b02c02532c3c11d2ad256e8 |
