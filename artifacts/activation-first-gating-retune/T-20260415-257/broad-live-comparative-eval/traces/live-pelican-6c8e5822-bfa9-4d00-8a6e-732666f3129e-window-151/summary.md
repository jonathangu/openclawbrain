# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-151`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c1b296177f077b6c8091fca65eb450be8d5f631873f87466a2fe9011d8b7c085`
- fixture hash: `sha256-56e21e2f0877b996d5170fefdce01e8f6c2815e782b17ac6f82fa56c1dd0500c`
- score hash: `sha256-85816e6201117d092bfa4994aaccbb039646c50d56e64a94c19573b78319ca1e`
- bundle hash: `sha256-d3384b4f4630b040df4b00389b1069c79eef0f6d01e409d9a20416b4c282363c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-29c5109446f744c540ec8fb2d0eb8a2d5f87ccaaa85851914bbf19fee8f8ade5 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a504198303656b2e3636f9750464c0c045b0d4d3cffe231f1f8782d1c1fdd761 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fc64ed8236f67ea03c544c8cd0edeaa9d62378bc52ca9c7d2eb55bdebae6607 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c9f67d7eb784fdcb77192d6a8c71e0ed44852e313603740739b3a2dde7d8ac60 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b273780b | sha256-bb2568f661465a09b150c5565e53156b8af9530f88c0111f7fe7aac606f5defb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b273780b | sha256-e4944d87ec81e53cfb49f50ebcfdae5ecef067d6375f7228f75f6224a68bb872 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-79733504 | sha256-1d12d3c7f58e42fd2f5f2862621b7d0065addda2c9ea18b5ef96c84d274c9043 |
