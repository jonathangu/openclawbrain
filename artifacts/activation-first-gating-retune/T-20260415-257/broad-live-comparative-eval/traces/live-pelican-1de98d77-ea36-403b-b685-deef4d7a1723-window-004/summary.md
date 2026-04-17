# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-570958465e5589f279bb28e850af48f0de0e358122b512d402db3214c7541c3c`
- fixture hash: `sha256-06808b26154de9486de3e390d83e02d5c54e1e0ca160f5f4c88501af04825dc3`
- score hash: `sha256-a6760ce45bb7ae160ccc6c420931c8426ce35c3d07c902f6262230cded5a7052`
- bundle hash: `sha256-533d98998ceabb245e94c41095f86a9dbe104f00be075e3c3f74b4b8677d8157`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd27477bdd5733b8ef83edfac9b06aafa0bfaf3753550669b2a8358e4c2d729f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-26a93a976b8a8e485190de2c4e76ce706afa0cf96de5d1b60e3adec5a518dcd4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3f8cfc759eda1044541aa1a7144c21a3db40ae3a28fda1e3ad0df0896f4d6e88 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-29baa651ecc142f5580ff8fd0d0194c3afa19fe029b9a6bf9216976e2d80be1a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cb2ecbdb | sha256-c1d7a04b2287b0c3085365e58db57ddb7cf443597216f5c398a01a40182af48a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cb2ecbdb | sha256-815e33ac4f69b11c99bcda1800ce3f256684a4e2e0f0cee9374793f6ec5893bb |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e448ce44 | sha256-0929425f9c9fd95919106c45d7691a7d0ccb564519f3b415668b050c2a243c71 |
