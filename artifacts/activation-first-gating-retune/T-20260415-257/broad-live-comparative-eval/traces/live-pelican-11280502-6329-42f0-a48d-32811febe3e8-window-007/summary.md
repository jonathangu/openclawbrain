# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dcacef92b60f17d22d3d52374e44f238a440853875f12c7d1e583783b59ae36c`
- fixture hash: `sha256-0309e3e5061db0b2681b554a0a0a11c1546e71f80786bcf376c32c5e8bf3ada4`
- score hash: `sha256-3c60e3e4123ee812d33df45ea986d7882f8cc3f7d976e5cc6c1ea44b6d0e0dae`
- bundle hash: `sha256-106983959421e7e87fe269de06b537fee174bdd832140186905294150c0f2002`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-23b94bf0f175f15338e42b4d068d31347ced04aa9e1b9081298f39e373d2dc34 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-9800d94e45ae1a3a036afb8a6c28ad8f7881665ce2ff3db05a535c29ba0f0f4e |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-6d1d87f88a989a6b9a27db4d82c3b0e74450b3a7ecdd229e829955a01a4cd932 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-a0b7a05a8df8739a39a724972daec7d06cb1af33a3159e9176d01978871d8aae |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-f05a677a | sha256-94369e84a086c305c6ace5ae40d69d0547978fccbace2989fba2477569ab149d |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-f05a677a | sha256-5055c6fbf34d7759be1c7a011cac8d21e6b3ed3a6df425c534408f6ac4c4fc39 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-f05a677a | sha256-94369e84a086c305c6ace5ae40d69d0547978fccbace2989fba2477569ab149d |
