# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-040`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7095a4d9ce26969c4dde9c329e749be730ceb1f708c47df4f4c59a5abea7434f`
- fixture hash: `sha256-107b047d2badf45fec45fded8a1234ee55c336b1a2803fdeba6955f2f30cad1f`
- score hash: `sha256-c4f42018f3111884e8dc870bfca0312835c01fc5c27ab2601822f8b51358ea5b`
- bundle hash: `sha256-d8496e906792e9a3764ab4a30b0e7696657fb8cce3008aa998b5ca35318a9346`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8413d38761902f8b7b6bde87782ba48c8aa416069cad02d85c57f922d6bd4f24 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-98d7981c976af74fcc82882a9f3ef67e01303c02c0b71ae597d8bb32e3447cd4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5af858655310e6fdcba2e205a64c3f68adaaf17b5df236ec4e1aff52949d26b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c3ee3f60767f86387632daaf6132356b4954b1ac7e6e72ccd940da084544908e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6c07f258 | sha256-5f0fe7e7437b50446ca81af99e137bd64bf9f48167ef3d15b071f8f5f0546332 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6c07f258 | sha256-0097f814d9b63503a03e2f1a1d7999f3c29c61928988966acdef3a8687bb0a2e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-1d755d33 | sha256-36ea985f86b11cc31af57798440329658ed4e9092d6dea42afcddc392329bba1 |
