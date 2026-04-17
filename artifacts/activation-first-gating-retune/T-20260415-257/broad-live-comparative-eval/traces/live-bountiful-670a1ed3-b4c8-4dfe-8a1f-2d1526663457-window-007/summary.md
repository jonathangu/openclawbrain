# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-829093e68c8b369222680a9ce88928380b9f7729d760e59ece8cf8d1e776b82a`
- fixture hash: `sha256-31ba926396eebecc30aa75781e7d614cd75f3d45744f5fc68d2426d0829db138`
- score hash: `sha256-2764709e96c3ee7f90b0484d7ed08405b1c811457fee0440e237e5feaa762367`
- bundle hash: `sha256-927aafb2719ee74d8eaef9575c71815a66730b305b627b0be72901aad250d1e1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8d38776b0b580dc292e4970ed98776136ff0d2acc01ecbb7a8d527a0c51a84c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9e8a8b9997b40437b05ef9eed2a6bc9fca373d850d04d5153e50af7441a358f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-381cfcca0fbd10d6ac03405334fcea5941729d4fde32363ad2bb0a0e6c5b2d8f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-db579ef2e6e09812c989672ae8198d8e14354398bebe9affcc3da087d360a98b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e8c1fde5 | sha256-cb2e7c11bd9fc3d0583c3762913c80c622c97b5d8c6cff25a989b2a3514acddd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e8c1fde5 | sha256-bbc3f0ea45a35395caf1a7d6a9267b3038328fd7576f870a7fff37d70baa2c31 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-8af119d8 | sha256-98cfa6efd94374d7d58bb7516e8dc71013bfa9e4ab696caba8c640551c68e0b7 |
