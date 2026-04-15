# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9d936165695614f904d36571a8a48065c182dddc8afd06f7b5a7de26e3d1a3da`
- fixture hash: `sha256-6ad09120c53334c8df0b9f19b852f07c2aa8ca071680e8461d1d0fad693137b2`
- score hash: `sha256-c0d4ae4059b07ab57079d0c175255876019e7aa64d189262b64bb0900768eade`
- bundle hash: `sha256-50c0c15a77b3fcae949ee2b566c96bd6969c8751e17518d88b5dd698537775d8`

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
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3329ac350a9048e47f1760a5c97b317667c0cdc04bb3d7fb2085cb6158792e13 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-68380af9122f0d255b549e96fae51ddc558347fd64b16c17f21207c39ea9db07 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-cf84a90a785b1c0996a67250e406ae69174438a4e855beea67b05c34db8a3eca |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-8f65a66ca34e1796d809104856a2da668350d7f6eb0feeebe649986a224e2502 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-11829959 | sha256-b1d998116eef2d1d9862e543cba9258db7bc0e5953f596e058e3bbf58f06c0b1 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-11829959 | sha256-4c7e9cc01f0bde71bd269f779b5a60bdb2c4bb68322138176f6fa9188514ceea |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-11829959 | sha256-b1d998116eef2d1d9862e543cba9258db7bc0e5953f596e058e3bbf58f06c0b1 |
