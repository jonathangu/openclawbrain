# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2ed84d12aa80219c71f67ac4b4dc49c9c31220d644ee2203f557cfbb2718f653`
- fixture hash: `sha256-a5d98f20c022a45dcdc79196fa677af12fe3ae7a1d81ee01512e8a79553eb0a0`
- score hash: `sha256-3d35b73892d445888458211010bb3a6fd3dbe2bdae79d7cab6b6ffabd78c6b1a`
- bundle hash: `sha256-2e3cf8bddb0b10d2e852986974c483377724e2393bed4ea478c8f8b5f342f1d8`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2d0401b09442b39c74248dfb10f1b77d9b52939def1349d2c685ecac4f520b39 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5348de45e7f587c0851a2a0628ea884c4cfa5bf9d35da8178c25d91d9030f05c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7a3a5c7398f96087c05c277b2239fae0a22bef743213234f715cb4d736626b9e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-33dffb9da3a1bce445ae7239cdb935b9712f9dbabf36bf539f3a1a0d98bea31b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1126e138 | sha256-e437bd62fa2af88b9a9b36d67c3cd21f5ba44f9e96f684a85e16bb60369d0408 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1126e138 | sha256-3067187259238a709ede4f281d0d35d9c1c54bec3cde91b07a2984ca3d5147a6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-1126e138 | sha256-e437bd62fa2af88b9a9b36d67c3cd21f5ba44f9e96f684a85e16bb60369d0408 |
