# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-367ba0e9c1765adfcb55faa49a77e3f08a37eaf77c4964ca4eb0f5d706e75deb`
- fixture hash: `sha256-c755dcbf454eec2e6cb44da638da71dca0e7b64e802782c096094c2870f2abfe`
- score hash: `sha256-01f102e027d00d2698e207987d53daee2abdb78e8bc2cad8743607ce0c99fb22`
- bundle hash: `sha256-81ed081dce218c0f4eca2793666b10f6e86b8f13af7316fe911e1d41de6eea9c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df30ae86da7dcb946f187b86df35238c1caa6176c275bd81d1099e4de3972842 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aa67937bcf01303368b7fdcec8f65112acf32e8d4e86dcf28fe76d7c7ee02b6c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-46cc2d50641bbb36edfc1e8235535ec6b2b03c0d323293c2bc9f15558ef28dc3 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-12198dd4ef37ec3ff1bfc08cf41b8f090761cda034db4e7dc36c39e283c34b2a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-073fdb54 | sha256-f14b3a6e27dca422ffb9b4e363902a331cde6b5f9fb959f906de40bc7c42a9cd |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-073fdb54 | sha256-04585433d414e5db0e9eec19909e34ab2ad3569189cf66dfab9f5f2a9eea98f6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2325769d | sha256-8b1dce0f70a0be721364cb288ec7ce8535fbce5d52ad37f5defaa23ec10091bc |
