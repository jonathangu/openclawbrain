# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0a0997478c28e207b3d7bd7d15ba48de1f89922b141d32d3371eb242cceef4ae`
- fixture hash: `sha256-9703f088aed39f3dc293adab170d3ce2900e0f982693a357e7c4d414d8997e11`
- score hash: `sha256-1e0b9277137c413b5b990fcdd1fa57afa45edd1be997e847ab9e808a01702c08`
- bundle hash: `sha256-02ca7a8da6013d4e8599f50ad8dac65766bbc348d73135bef845f1394849ccb2`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | vector_only | 100 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-27bd47690a786c5153f1de6a47c4efb1e5b3279455c8d667f6627f41a8eb28f0 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-d828a1a4bc7ed496b4964bfeedd765ece3c063ed786ed8db1726f2087686b9ba |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-a8d616cd9d777c029f8a37aa5cebb4e3ede2304cc04b0a320547ebfdf9a64d11 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-02f9d15c6366d2030e131071f1574ba904d3dbe1796d61d77a6e0deb078f8e4e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-410c7aae | sha256-a4b9c81babb09f8b8b3342d7734882c961bc3e786ce8560461ca7c874993dc07 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-410c7aae | sha256-8cdded98483cce617980357d1117bd036c08ccc468f9d668296ead460645175a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-2d8a0be9 | sha256-43206f3101940bdba73a9b81aafe6086f00b5b9d94e4009b3e7d709c09ca47fb |
