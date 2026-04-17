# Recorded Session Replay Proof Bundle

- trace id: `live-main-4c69091d-1290-4bcd-a74c-7166c46e5670-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-44fbcb6576f6f004911009e236161c3fb072626b9bf71fadcefa3c9dfc1347dc`
- fixture hash: `sha256-178059882cfa4f40ce27919272b11654587c109af203796638882d20de0899c6`
- score hash: `sha256-721b2af3de7f18326933dccb357fe3cab540e9bad1d62ca0fabcfb7a61eec5bb`
- bundle hash: `sha256-80605b0b36607229e56f4c2b006bcc0e85b46f19982664b69be92fb4854664b6`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fc575aa7f7247c73c93f72af53d1bbeba87c049e551196e9ed2534df2a742d2 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9654c56ab1a195138e5b93fb272503df006166ea9c1626e722c9bd4530e0024f |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-ae7327afb2a149f4346afb3234cca525f067ed2f8184f1b21c17f0ab703ac007 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-20cd2ee6187643596bce238b7ed0e49b3da95feb99eb8708a950a72bac12a0da |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b41efd45 | sha256-13b55d16c8d98e4f14c3af72693381e575ba3d6ed5c990ebc2918180b8eb0e82 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b41efd45 | sha256-dfc45faf47ba4dc3a8da68859f370ecf12e9f7f88a2077dd19e4637fd48c95c6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a9c8b3ec | sha256-51dab051c15b86558896ed89e017a006a0c9eca3c37a03477f3770c1699e9793 |
