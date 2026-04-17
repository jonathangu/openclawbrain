# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bfcdb554e6c3bfe187f4c905f92a9b282d7821367cef535897c2815e123fe75d`
- fixture hash: `sha256-3907274214cdd60210f9dcb9d9b0e865d090d5365a59db918b98e4ad4849f4e5`
- score hash: `sha256-cd02e661da95e973a5b7358ee76d8ccc746cb9e7c0c6c6757fcd783b75b3da1f`
- bundle hash: `sha256-fb3c37961bd81b78bd02cffe84c784ebc3e047ad036950b69efb1a2666bffc23`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-18ae191771eabc01fba0eef9c0e7f277194aa1ae188e2e94481f667ee00cc41c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3eda2fcbc46986b025c81d13b975880f1c2316efac55b75b8a81ede71099dc2b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0eab84d4d2830ab3ee2a8b77762e012267f7f18255c7a4c851c18098cab9b631 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-2cad1ae038738d27c5f3026c67b3a1e4a60a14bd95cb5d15979c5a0cbf20388c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a3e90159 | sha256-7d33f93d6032006fd35208ad9d39fc3000b6568f961976001bb539a4978a7691 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a3e90159 | sha256-497fd1bf59088c9b33f11c645fab12b64e5ccf25ff299190157c098422c84752 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-71d018e2 | sha256-e5b96202af3a4f25e38340a6be3b3c81a5c3d7e9ba16e9fb394c026125197e68 |
