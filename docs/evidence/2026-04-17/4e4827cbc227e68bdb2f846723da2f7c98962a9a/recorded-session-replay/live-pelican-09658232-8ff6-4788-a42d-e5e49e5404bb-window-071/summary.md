# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-071`
- winner mode: `graph_prior_only`
- trace hash: `sha256-aa6afe07711fbc8a13484cd14e70ac82c78cc503ee5449452a36b775fa63c3d1`
- fixture hash: `sha256-bacff39860081979b6852dc7223e7e30d3e6e8700496899a8864e78cf3c36fa0`
- score hash: `sha256-23d7344fe5e7075624694f2bceb27963bc72728422e3f7512ea815ff63cdd4fe`
- bundle hash: `sha256-758dd38f2a14f0f157d651244b464bbd3cc65c86d95c30363620f3377fc0e044`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-080c0791d3c8d4b27935c18a06ca48413df84ee848ffe0bfd6099d007a81a298 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b3dc82031f0bbdb1c0b4ff9680b340bae7e62374279d3b8fc100e19ea8b96916 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c065d1b90d70ae3400a21d2ab03206be0f3a98d2817a92f3d370f5d044cd24fd |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-36fff21085abe27b28b604978dbcd54d797f523f4da7bca91a746766b42b0fd5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-773bed98 | sha256-72b145e63205fe48a045f6277a941a2c6b0731e4cdd4ad0743f6b6a5ae1dfa48 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-773bed98 | sha256-15d7c2157a708add3033c6b7079e141b0e6fed8ce120dc7fed3b8e265fcb44b8 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-a6c983b5 | sha256-20b97019356f85d709ef85addc99ab4413a80f2467a5a9211d0a6d8ab85aade1 |
