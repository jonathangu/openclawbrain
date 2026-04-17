# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-367ba0e9c1765adfcb55faa49a77e3f08a37eaf77c4964ca4eb0f5d706e75deb`
- fixture hash: `sha256-c755dcbf454eec2e6cb44da638da71dca0e7b64e802782c096094c2870f2abfe`
- score hash: `sha256-cfa32cf73c27903a9d451aa6b2e4aab72e45d199fbee8d32167329f996e739c7`
- bundle hash: `sha256-d8a733bd0964dac5dad9f9f976c6514091c852aadfba5bffa5890063aaf9a8b7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-df30ae86da7dcb946f187b86df35238c1caa6176c275bd81d1099e4de3972842 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9d408f26f2becd50a0eb77bde400a3939a9c22840e157b937b8de89e51bf2200 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-27c6b88a4c2290fcdbeb0cfff8cb91933151133c676d2fc92e913353b1c5d956 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3b53add42bb74e5b5e92cbe7dbbffc0d87a94f528a1bc16551bb80e5414a1c93 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-395495ff | sha256-65e59aa8719f143a30dfca22be94a5c10a1d4222eb7e343650818fbc38a10aaa |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-395495ff | sha256-69a77ca4abba5e1a40dbed7b7723421d06984a3c81c1ff475d550e257221d673 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-553a3148 | sha256-4297043430a4d2f78218a6fb14625809500e6b9e0b9d1f74eaebd84b4d436092 |
