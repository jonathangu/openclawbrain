# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9ea17f3aebef4af75f0a93c521d6e776070c7076063ed113dca780cff0b9684`
- fixture hash: `sha256-e43f09daa5c7f1f8012274d4f09baa27758aaa51c3e914baa4ee6b5329b895af`
- score hash: `sha256-086be815d3856530bf9408efa11d8724737cc55ad575f11c0a8d828d5f93db1b`
- bundle hash: `sha256-16a5e5b1460dc5c417db408ab8e8a91fba81fba3ee7562c278bf488f247159e2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-174fe02cb9d576a687ddb560851b02ab0e12cb6737fa301408229ad552fa41d4 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6ef768e2cb0570a9effa103084fdcb8df7b6ec583050551265da23e164bad80e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8445f93ea856bfca7283fe62111118f1044ba9a573ece1a87bdbab9dc8c78478 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f63232b1ad6fd4d1c7f3923968d84eb5a4eb068cd28bdd2929242064bc3f6747 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c47a83d8 | sha256-1a4d43972cb4df9143d659af92f0b11ace9193ab880f033b63721d84c6bd7148 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c47a83d8 | sha256-5e1afe566aca2391b26de470e18ef6ba2c957e42c2a38c37f62e7938930caf79 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-fcd71553 | sha256-270657a69104681915ea7e2178bf42539f38a418328b0ad6d1597ff817e1fff2 |
