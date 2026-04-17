# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-037`
- winner mode: `graph_prior_only`
- trace hash: `sha256-67dcb8532f54fc5f6268aaf2cd959dca249a5c5832d88990a647435a45026ce8`
- fixture hash: `sha256-53a4c9afb3f28aa87aa1b17aad9db78e9f58b7b80cd2cde3904a19a0bb713c36`
- score hash: `sha256-a83a0b0c9afd7ee40c80ced40f3d2ed94a999ceb22572c4ed9eb2184db898d0e`
- bundle hash: `sha256-4cca9febbc02f834f7c3ac62f42f42af58e6c65f2c2ab931c08a11e4697e05a3`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e721a4b8ae2bb9ec3999c909e1329c35bf2b76bcb692645b1624780e9c7c3c31 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ec8b2434b07123971dd3c14ac37001ed0feee33b6c48d921549e152529c1573c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f6c218ebaff1a6adca7da89a6795b1490fe1c530962ee5a1518e525dbd28ff2 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-2dd028abc07ea3314a612469a421cc07e541749f22cb0f0282d384ccee834141 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fd13efc2 | sha256-a63b44e24fa6b844dd52aeed9f784db956f4779115ef5056f22e77dc444f4e61 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-fd13efc2 | sha256-4af086e814cde9246aff7481ce5c89572d45fbd7fd95082af3ac01b838fc1a5e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-d88670cd | sha256-1493cdbd5dda0eb0b8375bfabe708edc293feb46018ce5bd795c2b6893c2db46 |
