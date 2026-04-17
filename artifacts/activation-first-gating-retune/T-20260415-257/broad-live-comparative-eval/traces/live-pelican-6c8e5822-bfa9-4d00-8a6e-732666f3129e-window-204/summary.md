# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c07d86727cebfe369bee33466c114948863aec275c3915842adcc6210ff9f00`
- fixture hash: `sha256-8b304aedcdacbc80bf121116c28b99b2494a777738f36a03524f91c39297ceda`
- score hash: `sha256-ad87e8b35123c9b3495440ff3575794a07d6cdcc59af9ecbcb2015ccc8b326d4`
- bundle hash: `sha256-70873112a322e0aabfb22a81e29badd41196ab3c16d71e1fd9548c6074104009`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-61200abce9d599bb6b2839cc09f35d3da44db5dcfc15754c19ad25f67b630577 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3ffddca8c133de2eaf5b91b37b7e30c6439c2dcfd9e1a1ab56a94398b0ead9c6 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-55aad517a5c52a4b150665046bd1a6d7a11a95ca99e0854932c85cbd41e4ccf4 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-789ec1e51633f2e98686df2e286decddac9f1544975f739fc8b46279b18d0f58 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-adfdfc41 | sha256-e90b14e6a9ee0c5a9787edbaae57a2b4e206684f9770025a029e2f5928f791ed |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-adfdfc41 | sha256-95642c53157f57ade4b6cad9da49bee5339886d88bb82a0b05440270c93ad1f5 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-4bc47162 | sha256-ba08185bdf676a99d22b61754344cce5d14137787844ccad6d17d39425f1bd48 |
