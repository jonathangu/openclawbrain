# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a9f3db5c5e9f18aad5ca8aa8c8134dfac254479399202badc35306faa348393`
- fixture hash: `sha256-af03997f06ab50c99afcf76923b04c21e1338d145564c582674e59eb816853de`
- score hash: `sha256-f91b5c96767d42658a3d4dc80c044f6da37510c0911873d775ee1482ab2606bd`
- bundle hash: `sha256-0a0d68ad8fb850f87b55ec3f8b013900438598e6d71676fd0d23207bead34927`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-048408a5bc1e1d56a6cc83e227b9a2958b83cb861b21925fd209ce4b8456f636 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eafcd47754ce29491b7bc795e723ec5c8170a6400a5aa2e7ba6f95cff05070d4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-90ed428444efadb594f75826d2b860821eaf9e87a3fb1cfa15dc1c4fc7a6202a |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ff0493f528864795ce01fa90f73dae91fadef5bd03db36b44f6144b4a208bfad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-232332d5 | sha256-2e343318a45fa30a0db2f2efb29a841588dd23b30caa6e60adf6bc29318e8bde |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-232332d5 | sha256-1853540ca199bd3d0534e224a9b4cb5d1c1b190fa4896b191b3724bc6434e8b5 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-56da85ea | sha256-7fb5e8bcc5eb0d88828e2dae0d68fb375692bd0100adf5c06b33b92d7932652e |
