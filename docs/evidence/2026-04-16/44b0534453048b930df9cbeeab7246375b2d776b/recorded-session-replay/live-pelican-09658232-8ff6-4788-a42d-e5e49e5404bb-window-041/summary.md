# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-041`
- winner mode: `graph_prior_only`
- trace hash: `sha256-76273bf572d2d6df7f5708306c7a325e0d7fc022256a8c48664d2d2c99f93d6b`
- fixture hash: `sha256-5c64fa2fb4319875db5a6403e087c56d2c1a468e1ff9a819a4e71ac1b0668ff8`
- score hash: `sha256-5abbfce5405d1be6d8fb71768ad27597f46a2b5aabe424a35cc847b24c233881`
- bundle hash: `sha256-8d145bb0054cc481840d86aee0712df11b43b0b2508be3b5c85c4c4d21351b43`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd39b973453e541375d3824b8f2b46f3993347e9ee385b937ee54648b5838113 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a9d5067bc88c30f162821f8ad8f339eb40359ddfcc773599b31ae7126a1c8dfe |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e11bd5f3f7d10ea270fe912e0051c12879289a1b6be32e50c41633f1306f1187 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ad28aa8972246e04d0ca3d986d5665d0686bc671a61a00a9cf1f52e2f42819d1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-78acd8ae | sha256-90bf277e8bfc3370e48d8a936550932831aee9a9c092bb4669bcd1e4349fd5e4 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-78acd8ae | sha256-90bf277e8bfc3370e48d8a936550932831aee9a9c092bb4669bcd1e4349fd5e4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0ed86eeb | sha256-d69ac86848ce3286408fc56d1639e57a1c9376e692aa87306b4523297894c53f |
