# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c817aa28a6ea88ab750b90d075966003c4144ca68cee4de31510afc8940af725`
- fixture hash: `sha256-12c8924300be23df2d629cf06b8bf4e9466d47a9b90ef4b0770c780fb827282c`
- score hash: `sha256-09a1ede3b4f215eaffb573ddb05de1c6e2fd984171a028ce53a23de7ba9c16fb`
- bundle hash: `sha256-d302791fd04cd9f099103210dcd484711659386d3ced75b99226393b43dfe133`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a755e36be001d38e08b764d65e8f6dd1b01494428975ffb22d7f3f721a73e79b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bcee01bc8504254d01c997aece59c6d4ed046c7429b208bfb63b2e9de5f41dd8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33f413fec6d9680d8b9753608f57024a25fd589520bcf3fc622e168020dd1672 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-c3671178dcfecba4da95ffdae118e93d604508407491f3d5d2ffdee3365a33ab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e65b55b | sha256-d2262b7c1dce171088a830bb8966b44eaddfb7b15819b60dd5714b4fb37b3d60 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e65b55b | sha256-4a98be288853165113f378010e7e9081941815e8d8fb841a6fb26671b1d9593d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f3ea5852 | sha256-6eb60f76ecc3b9410153d92ae970b6fc75557a5eaf6e8fa00a113fdb43f262a9 |
