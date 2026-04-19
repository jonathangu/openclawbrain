# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048`
- winner mode: `learned_route`
- trace hash: `sha256-9c32a87b231e4d5848a772d9d1cb8d355e8b17c5c883fc0f1ca8776ef042ba2c`
- fixture hash: `sha256-66d4441e9cd89d5df06e129fcf70accf27e8123573950bf81a6f813e2979adc4`
- score hash: `sha256-c890449f4b78e4ea792d7b1b8072b866637c3d06a6985c44c42680dff04a4bd2`
- bundle hash: `sha256-b63717f8b424c7df7a31d7b4d504d9f1c98a2c08933d7190bee9b87afbce4d9b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b9d9843197c6ea9cf1bbaf94c65647f4ecfa1e2224f8678711a552cc896cd7e |
| vector_only | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 1 | sha256-1417609ca27144951bc2a9f8f1a0bedd4625edd8f34ccce2caa2b3c39acebc66 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-daaeee99d413067cfeddb59ef17eb952f9868aac30b18fee7b7771770d5b5208 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-9c60e7299cbdfc4e135e8531b672c4cf279ce5c789f69e8571b4876181e35bbc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | yes | no | pack-bd2941f5 | sha256-61fa56475f16cefc1004afc1be7666f90e788cff912c265e696a38cbb07fbc2f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-bd2941f5 | sha256-e30a59fee2028c0e52bededd72461476e5e539937e62649df47f73558c413846 |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-bd2941f5 | sha256-fc3aaa80290244a367d12442bf82d88636ea88f61ff80d8df9a0239264a095c4 |
