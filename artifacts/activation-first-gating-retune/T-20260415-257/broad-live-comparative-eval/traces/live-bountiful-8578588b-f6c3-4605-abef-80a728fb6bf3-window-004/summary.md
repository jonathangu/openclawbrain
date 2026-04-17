# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e204eec37724aacda490110857d04f610417247b024d310ad6302080aea397ff`
- fixture hash: `sha256-6f87b65c79b51cbc945548d32dcf271be29d7e50d05ff0c454ef7979a8b75cf2`
- score hash: `sha256-f80d7020b4abc1cc7a4a1c2fc6db140276983a420fc6ad2716f8298f7f72dbe3`
- bundle hash: `sha256-f02a17a2f510200ca3fe523cb105bdd554351ebe69ca1ea974df1833f28d6c9a`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c76acc1a03aa06534fe63599a6439df3e7c5ba77b6ef580f7d358b50380fb3e8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ff6aa5d5cc74b60f0f4266bcb6b7874ce9cacea6d227c41c7662824645a68d91 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4dd584aea2c355c9be5952f425ea9d96a004cce62a93e07e01aad441c0446a18 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-4e89e7237d552d36eca44d2b137c9b00b1c6864f65004b5bdd6639007a05c270 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-68fda9d2 | sha256-265697f071253c7a0c8e04b0dc1156d4e530a23272d12ef783c880bb826e1c53 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-68fda9d2 | sha256-b28448cbda645e1feebfa6951dc89c5a42655f3417ea31f76e3fbbb3be366f79 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-a4e80e43 | sha256-38ffc1e4f7c6bf39679a2b0b6b3a8eae5eb8e93716c9f34686bdb07522b8546e |
