# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-058`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70e107aa90463a0c77bc30d344eca5153707641920ed24320747bbd52e05a0e6`
- fixture hash: `sha256-2a5cd5afc4b09fa9beced059043152cd23fab3958640aae8275a1e91138ba120`
- score hash: `sha256-16bb8454fef39df99c8a2d87aaaed49492e2ec1ad81270f061b7588ad5b5409b`
- bundle hash: `sha256-6a93478dbd580482f8f907fbdf3d55216afa7b7b68562dcf83021f6cb07a96d1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1c146c3106aa3a476acb28a8b075ba9caa0dc741d245d11ea00bbf3c4bbed6c9 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64ce78bae7dd91679595e38b78ab1ed780ab92bb0a3c7e448da93a71b0026611 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8c93481d0031293a773e10ab5ca1388baabb1dfe1a5ee03690c94cbe1e4852cf |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-b11824bbea7b13da68d2c469022d2ae88d4ff354ea2c4568055a59ac2b006f46 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cf362584 | sha256-d631516d76bd3c354ef92d1bbac6fd3153c0a57801613b0213fa6e5a4e1970c1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cf362584 | sha256-3af7ef2db8cc11d24ab9a5bf19d4c575a0b57d2afb5899f7477947c77c053dfb |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-95025bc7 | sha256-d51bafe3e2852e628d51f3125943500e11c2b7722b58a6f895a6216551dfd7ee |
