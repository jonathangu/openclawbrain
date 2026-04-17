# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f9684c38d91e55a983d42052df21e03bec407bc3f34393946fcda8e1b2d39f4`
- fixture hash: `sha256-5b6e1bbde60f4bcca2052f19249d943d07695521da1e7e8b46846e97b143bb5b`
- score hash: `sha256-762a692eb8cdcb0ecf6cd9936592bc98ac6acf98a0439fdd27b576cd5a81abd2`
- bundle hash: `sha256-a5f24d6a47dac3e554dc9b7be18e6224c5fcf2a9b9b158d3e79ee85270677b5a`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-74a0ec494cc9b5ec66bff70ca0bc3e9262d5754f8a93ce7d222367da206ee232 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d0bdf3f5dc44f106dd5a04e90ff7331c00744737eb1448fcca6fd30d129ffbc4 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bb97aa628377ced8bb7d3c7dc4ba084a5d7d6ee2e86197b9fa13e6373f15fcbe |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-c75929d1463fd1625ff521fe2e8c60a0415912952df07dedf23701e77e411dba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3a224fe3 | sha256-6b4e8627fec25a1f6c27442458face0e68748810eb8eab32302ee2d5bf7af6c9 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3a224fe3 | sha256-df0ec52e692fcdc22c50eb83873b86de5ae806f7516a9345fd6949fd88fa966b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-7eb447ae | sha256-5c1c9b5f1f765e73aa63b340e7295d841d5185b95c065288a499375cbdde4382 |
