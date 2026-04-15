# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37b9967646ced8e1a7e53e66d95e96c0d5cf9872e9f6cf5f223ff75c45212fe4`
- fixture hash: `sha256-e5562aca0bd9165edb9d4f0591f9dae6981c5299e9b8cff4453286d3a3e6c950`
- score hash: `sha256-869d2659e584360f3632cc4f5443e44552425f8dbf19295a0d5b61072fd8a0ef`
- bundle hash: `sha256-d7b23babf247c512c0fe012c4a6cc08aa65be7bc5b26c033ac6347ebebbf2280`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e59e0368806e0012160cf4b2dfced7c5e08071a2c01bb62268694e031a82feac |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ad3a21e766c08e3b449db1627b5f222d6dbf30c77edec8115dc4bf766f670ae2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3228166b398ed607564a2af57e4007e2701f6f410e61bdb1f1f52d3b04f9a3a4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7d03fabb7ff1301c67318515e42bc8a9cec26d10c518dc770ffe379620822735 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4c42e380 | sha256-2f686aabe55d8a61215da83adcccf874e62bc29584a6f247d6780bdb673059e1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4c42e380 | sha256-2dbac910b31f4b0d584b8242dc7c4befd0f9c34e5c0b7d340cee67199585d263 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-4c42e380 | sha256-2f686aabe55d8a61215da83adcccf874e62bc29584a6f247d6780bdb673059e1 |
