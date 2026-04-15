# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60359ed78d5b78e9d115bf8cb9e9ba270e0f90bac409bf6884d4a443b2440f94`
- fixture hash: `sha256-0da91a494c8a34b6c27eb293958b781dbe6bc334337372f9fbd368fd3d0ee08d`
- score hash: `sha256-1f039e943c384a25e8efb462f08ee7ddce689a9e104827f04955ae40048c84d9`
- bundle hash: `sha256-3ce9e9efb19bfc3745d546284c5e1aa60aeddbb4d93fa77475db4cb7cef7c1b8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c77331726a9326f500ec3f7c3dbbaeae387d368e17255232ecaec7597f897fed |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-afb6c74af9e259b498a3acc814914da5fbcd0a6a7fe5d3b983d5b8bfb7fbfa51 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cc939ece90810fef7c98dd712eb41ca462ebcb8546a297e29b94cc74f5f71fe6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8c1d1eaaa9ca6b8f2dd3a08c6dbd41cf1bc3c2cb57402054f19904c91dfbc228 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cf33edf0 | sha256-64fc3dd8ff9129c0cc891722d994037859de7ab25b482dca261742183cce2cb1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cf33edf0 | sha256-faeba4e4c80c22cdb8a11f36c039414ef35ced555395bc27d68817672b2bd580 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-cf33edf0 | sha256-64fc3dd8ff9129c0cc891722d994037859de7ab25b482dca261742183cce2cb1 |
