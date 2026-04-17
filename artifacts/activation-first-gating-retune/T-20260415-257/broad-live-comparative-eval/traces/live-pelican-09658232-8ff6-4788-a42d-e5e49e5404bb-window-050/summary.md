# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-050`
- winner mode: `graph_prior_only`
- trace hash: `sha256-edca99fed5bfa02f8196eb002c7ca5449e3ad77a55a7ede4db613e0216b9a288`
- fixture hash: `sha256-075e6ae40abca95623d0eaab9386a47facdde038c6af2c88f5255ae9a6184b2a`
- score hash: `sha256-dae7c5f8f2b5529db62f231545172243752843ba5a4f658ba6be74581272bb57`
- bundle hash: `sha256-fb2088bfd5a9e5512e2f3d1b7323b92492933a4326f7d139411160c8d6cefb32`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6022525cea9be81df298416f8768122250fedda0d93e17bc4857c9bee2c2bbc7 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6b89ade587c85f19ef6cb0d3efc21cb5d3347c0598e074bde1fc329bc8712d26 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d515e4324e839c1dc786b60bbaf0b1e88a170530bfdfef4f56af488d78aaad02 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-76e9af6a9fc90b0dd69c73696e80c87dc3e4b7f2935c18eb56cfd471af0b84be |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f2b689fc | sha256-556fc84cc4dc4b6753fa59f5a20caf0bfc3fa1ab9775e2773d8fa70b9a2a0be7 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f2b689fc | sha256-556fc84cc4dc4b6753fa59f5a20caf0bfc3fa1ab9775e2773d8fa70b9a2a0be7 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-1a717ee3 | sha256-67ed824e13893fcd4299484c665ae084067e228148d8f465f02fe65c8ff4eb56 |
