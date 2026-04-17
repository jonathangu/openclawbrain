# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e204eec37724aacda490110857d04f610417247b024d310ad6302080aea397ff`
- fixture hash: `sha256-6f87b65c79b51cbc945548d32dcf271be29d7e50d05ff0c454ef7979a8b75cf2`
- score hash: `sha256-a0a3152dba6ab2300e9395530f8864ff9697ba0174d486d15d45a93bf62140ad`
- bundle hash: `sha256-f917064b227ebd455a65ee45647953077fe6bfbd3fb4a13a0111eb2274a15048`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-6488311c4303cdd19c15e2b6e89692bfddaea43bcd0fc8c54a623f75e2bb84fc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-68fda9d2 | sha256-265697f071253c7a0c8e04b0dc1156d4e530a23272d12ef783c880bb826e1c53 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-68fda9d2 | sha256-b28448cbda645e1feebfa6951dc89c5a42655f3417ea31f76e3fbbb3be366f79 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-a4e80e43 | sha256-9a748addd659a57afd2b43d4e9e37cbf2fc0343a17bb536ba7519c0d3ed51677 |
