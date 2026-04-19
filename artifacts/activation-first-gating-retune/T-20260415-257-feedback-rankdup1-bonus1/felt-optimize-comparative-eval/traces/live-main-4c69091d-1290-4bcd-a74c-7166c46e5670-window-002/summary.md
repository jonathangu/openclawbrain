# Recorded Session Replay Proof Bundle

- trace id: `live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-44fbcb6576f6f004911009e236161c3fb072626b9bf71fadcefa3c9dfc1347dc`
- fixture hash: `sha256-178059882cfa4f40ce27919272b11654587c109af203796638882d20de0899c6`
- score hash: `sha256-8d80e55fe49a7b4131e2e4538c87784cfa7c457561e5a224ee5ed9a9103bcbfe`
- bundle hash: `sha256-511208afaf01742288574c88a20dca9b0fcaac7c55f9360fdd99e636793f0ad6`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc575aa7f7247c73c93f72af53d1bbeba87c049e551196e9ed2534df2a742d2 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-edbf55d87b6ea42210665f7b83d4ffc5847bb135c90dc10fe3ea274b603a5a73 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-bd41852f2ee80e9e962e8d40d0f9fcb0229a45e62f1c27b11b9b85e2a3baa35f |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-2c4b16066543933f567ae8955d71200d5f51e6fb4edb4acb2c76c743d024ff9e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-98152994 | sha256-c9e5796f69157f80c841c0dd914a4bfd0c17f4b8ca243988bdcc11d9d002020b |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-98152994 | sha256-448c6c3f46f1a7dd30870fd443f0451bfb642334c9e64e8244db261c48a83851 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-98152994 | sha256-2e9001c137c609c30588a91fcbef4964b0d246fc05a13bb315783d7082dddd57 |
