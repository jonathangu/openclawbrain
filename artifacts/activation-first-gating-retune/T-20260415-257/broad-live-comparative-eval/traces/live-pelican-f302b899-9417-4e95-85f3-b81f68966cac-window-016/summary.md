# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-60359ed78d5b78e9d115bf8cb9e9ba270e0f90bac409bf6884d4a443b2440f94`
- fixture hash: `sha256-0da91a494c8a34b6c27eb293958b781dbe6bc334337372f9fbd368fd3d0ee08d`
- score hash: `sha256-1e53e60bff2d4e347f9a9f8cffb658b2b36a1ff2eb11cd4eb44c06c3a865e2ac`
- bundle hash: `sha256-79ba651e7e68bf9db8e39df15e4ff57dfce7150a25d93bff331c9b659c328760`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c77331726a9326f500ec3f7c3dbbaeae387d368e17255232ecaec7597f897fed |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aa7b74e2e78db3a17a62ad8b90036db93de9f1fa711141079a1b4e85760df9c5 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-46bd7f20b9c14629cd9dc0f56def7cd02ffb16b702d12c5127dfa9b31538a159 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9629444102c57efe16a1aa943696e5d08b245a7aeb741d0694d49d6fbc95f471 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4684337a | sha256-5f4f0607a283e13ccb8443a07932d197345c824c4f72f528b8b2a948c5dfaafc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4684337a | sha256-20c40b2017bbfb06d64a8d23d7d37d03d61bb8ac7026cbb8711a2d8b48feef0b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5b5a5393 | sha256-710a8626b6f15d272229080779f7d3dadc2f16caddb1143043155787660c87df |
