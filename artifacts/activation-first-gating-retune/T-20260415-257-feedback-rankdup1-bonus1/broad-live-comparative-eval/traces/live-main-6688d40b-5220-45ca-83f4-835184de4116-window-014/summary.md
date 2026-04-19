# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ed82473cf8a44c6f378cb688937e33b3ea351a6801142726a4915ee5fb6d88a`
- fixture hash: `sha256-4909ff1896b085966400449aca0e9ac319b4cd9d22c11198c9e8e1d61fedcf2c`
- score hash: `sha256-c27341d67eec5a98e6da0aeef6db217b65378fdf7c3d35de6d38c97d4d415622`
- bundle hash: `sha256-2e1dea3d3d975dfb5b5aacea0f2c76f1df9f1044654b22929903fe166d409541`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46b89f268ea78eb5e49f9755003f2aa744b81e1b854e2ac1c9e8f1a95cc59955 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-20c1446f451336ecece608800edd87aba5b7e890c9fe5ca65381a17819a166bf |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-7babe2e5f1c9dce1b03de181455c2785110911af5282029bfa4ad0576167ecfe |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ee05f1f9e37b26b78de4131df9ed60ad448c7f6d5cb11fafff88d350c160fcac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2273de52 | sha256-f22812c4e81d0fc801d692785894497f868b50ffef58773aef5b4707c5952cd6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-2273de52 | sha256-3e04c991f0be0e1e61d9734b9314f2e505e94c52db81d2a0923b037eb4e9ae44 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2273de52 | sha256-f22812c4e81d0fc801d692785894497f868b50ffef58773aef5b4707c5952cd6 |
