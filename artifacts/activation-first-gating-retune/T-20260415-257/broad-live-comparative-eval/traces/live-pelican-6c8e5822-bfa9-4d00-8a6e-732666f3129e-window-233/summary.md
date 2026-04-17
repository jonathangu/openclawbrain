# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9d936165695614f904d36571a8a48065c182dddc8afd06f7b5a7de26e3d1a3da`
- fixture hash: `sha256-6ad09120c53334c8df0b9f19b852f07c2aa8ca071680e8461d1d0fad693137b2`
- score hash: `sha256-51f0b0a47ec1037863a0a68f41cd94a3aac87b9d3f631718a5ec507d04bd2463`
- bundle hash: `sha256-aabd9fd4da523f0abd66df73715d26007f944dd00d16f4b6783c9893f5b618f5`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3329ac350a9048e47f1760a5c97b317667c0cdc04bb3d7fb2085cb6158792e13 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-295701b103917fa010da4943ef2bbc7f27b11327179c970c5c5f3cb033ffb628 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-f7288b9cfbdf163ea6c6b6b0a91a9f8ec4b0fc5a1102fb31b672d40051b9f786 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ea808c5cd43ad3095aaecfaa14f1ba6bfc6b2894b789e89eb6c84538cf52ac49 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-88d2dee3 | sha256-74176e5d791dd529f221e07950834d14324106b76a19200877b4e9857b8568df |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-88d2dee3 | sha256-27a9a229bc5677ea94d4e08793d0f86b91efc4da7b2aede9f09128ee926f6cbb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fb1eb00e | sha256-eecd192260db443dec4fabf43d158a1b902c46ea4af773eecf83ef5889eae6b9 |
