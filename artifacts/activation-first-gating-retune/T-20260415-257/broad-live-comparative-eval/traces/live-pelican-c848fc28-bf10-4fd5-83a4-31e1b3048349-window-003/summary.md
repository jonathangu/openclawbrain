# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c65984dab810fcd56a73ba24f7e48a3de3329e9e72c9abc055205970cf393432`
- fixture hash: `sha256-6edadb4cb34df6bab57971cb77cafbb8b923e3e92f73e144950ce412708011f4`
- score hash: `sha256-bad98d89ed201859eda806b1c7ac11b455fff279747d0e0197ced4b0291c6091`
- bundle hash: `sha256-3e7e351c7d5090336bf65fbc4e658da8b8d3eeffaa76c11f61feee9050f8a6a2`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b98041498153f3fab8845179ecda7c5ad292ef71a993f916db2031745eb7d0a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-53cb9af2032c7c0c907b74bc1df8d202c23dc7467c7f3fd7d4f7caccb9cea8c3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e1e4468fd8bdc33c2500b1cf1b0dca9a13c51f2b63b7144aa76cbd352e8f4e74 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f15543d72994eb561491c7bd8033411f421efa5b85c300390dbcdb3aee5821ad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-193d2d6e | sha256-743e13e96c64813b5b8159765ffbd240b43d1e1b6bb9cf57769ffa713dd7867d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-193d2d6e | sha256-d91e4b10ebae4d903b7fd87842354da2a671130281d27b57908573f25e2e5ab6 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-c8fdcd81 | sha256-df694c0a6cfa8967a3241f9fad1abafbec57bdf3926521aff077bbbef558aa61 |
