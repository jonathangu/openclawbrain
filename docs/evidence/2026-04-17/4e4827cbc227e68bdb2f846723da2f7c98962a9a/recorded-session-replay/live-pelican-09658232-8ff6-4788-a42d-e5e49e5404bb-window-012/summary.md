# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-50bf6e94091f3556fa81577b5a708e4425c4e417b6705bd10df603b7966e593f`
- fixture hash: `sha256-18f79e8bc777ea9555de29a01a8501d21c8ddf1c9ea32bbf589d49b4f4a3aaeb`
- score hash: `sha256-74d043d22d4e35721f2a320fbeb544f96e4e615877e379348417aab496e47903`
- bundle hash: `sha256-6e3bb73e3393ff9e2a52e0084b40d3212e292e7f6d791fa425e6ac9d564ff6f8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-12a31c86174cd8cb94081745ed9b01e9e8efd75760d60dfa60b0b81778821ef5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6eb4c7f5fe97c067839e58e1395cf68f1e59481b91b68f04dbf6fcb7494e5556 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5f4e16e819c13e7d9b3644852f5f0b9b7ad3e99d4d87d4f1b76dd49f361db018 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-878232d1c2009199032095191cc0c86e964e7d268bb045805a2a57d5dde5a362 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-99fb9f21 | sha256-50b012a660b1d33c20b7ea38315780b1c90fa6f803b574a6d12f9106d40373bf |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-99fb9f21 | sha256-b02d6ee5468d45e0de3deacdd939a1a7b5686d21524526f31a0506f0ea027b4b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-68968c66 | sha256-5582f9bdd5564e70c7aafe921096317b987c646e501962e09ad95f387e2f0d5f |
