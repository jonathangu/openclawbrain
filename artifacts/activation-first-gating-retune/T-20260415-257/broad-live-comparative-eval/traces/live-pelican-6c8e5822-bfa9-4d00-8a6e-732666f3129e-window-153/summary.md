# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-153`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ffcf94e58297f053bce53168278403d4ee13aef69fa248575deb3926c6117a0c`
- fixture hash: `sha256-69a203d1bba5e9efdb04c3d2b5eac78a0fd9782e268e61f935bcf93878b096ff`
- score hash: `sha256-5ff012bd09bbebab82cf1fa1751df63462a72dca3ca67cfe8a712ccafcd87042`
- bundle hash: `sha256-1107e1946d5a812f876f1b0fafce3b77489040974e26c415bc83876b4f2280c0`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d38606df0a6b5cc6fe27f296186c09efc80579f0832811cf6184d8073ca5500a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-65b21103076828d20af85306809fb601c8c433951c24b9146754fc6190c61b5d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b476d9f1f87a5264db33836b1ff2a8d34c997bf6a1ff48c3bb94f92e82e4ceb6 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-1ac8e73516a9bdd50e8b2a7e921fca933b21caa3547f125c06d7ec82c82287ab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3c2c0df3 | sha256-4c96db6ee295345bf65cef1a7abd223e107067d670bd92d8c71b697dc669b53e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3c2c0df3 | sha256-3ab174c1941b715881ab2ba21b31e4ea0fbdf832cdd2b60683657e81e733ea46 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-44e2ee54 | sha256-d5c8529d981cb5d5ad15ac445e4a090bffed8df67ed34784f855d9a5816740eb |
