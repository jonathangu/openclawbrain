# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-088`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e0cde3a25a3b093a3328111ec29f970fc82068378dbcdb19446f77be2e4c1e`
- fixture hash: `sha256-a657cfcc3c13a64972df27ba9b34b582252db2226ee691420ef45e3b6a2bad38`
- score hash: `sha256-0f2b8b8f4a15a5aec346c4fd7fdff619d67379b3f44c6d26c420620013b6b2ea`
- bundle hash: `sha256-44233c1800a0e07d4fb2b1900601fdb0996bfd96b6e73d641338e64b4955a7b9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f0b3855bdbfdfdd31f2cf7aedce5b8a8e42a2e757ba398c67ba975aa86dd21ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-89fe127e2bf0690c2512059ecd6b5de44cd61de1e4c09c8498cd46d0cb847447 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b42e27c3ffa5d9ea901f9be898965574d0b1f710f309d5b3bc40ba96173417e6 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-5f45b0dec486b8149bac269fd9ae666f26140191bafed6179b90e790a9617c22 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-85d53dc1 | sha256-d6ec27d860fe4b7460fbad63cb358cdd0618f192e3ac9338c0f60cf76d611755 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-85d53dc1 | sha256-d82f3dee64644df2cad80676010ca0bc57026e297c18896291d430f85b8c081a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-02c63b24 | sha256-35bbf5e739dca5809153531f28365a748756f0c721ab5263ad8ae1ab57302736 |
