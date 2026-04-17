# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-175`
- winner mode: `graph_prior_only`
- trace hash: `sha256-653e1762b7192e93df1dc01ad3fa2126f6513bad2e3d5a89891f193ded446910`
- fixture hash: `sha256-4262eb1c667bd83d27b33dceb3d4d1a1c6a1b57d1ba763770502ff6e7c8a4239`
- score hash: `sha256-f6f985f745027f773a5968b3112edacccb42c3e15051ba5eac6b0b28e7e1e5cd`
- bundle hash: `sha256-e9db83adcdce06b43bbb0b170d01338713a000a1c4d4b6043f0ef251174c62ea`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7f90cb77dd3ff507938d3ef155b0e74e6914215ea5bf7fbd610cc02d8404add3 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ee3adc4bcb25134a99d3d820de69ba2970c348124a253e732957228702597757 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a55ad6e5fb7db7474dd988444c7f8dda379845bca2a6b487ede1b07b17462d68 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7f0af50cb9a7747fc1bc724fef1a4baf6214a1add08e85f376fe283e9f890873 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-230c6dd4 | sha256-c90293e510cfc72fb6758afe9ca2ca96e66dd09c7b31451aadbc45a7477842d0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-230c6dd4 | sha256-dd921ec4d31835821eec811f7afcca33d23a3238be16932699fb0c174180e13b |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-fd35ec89 | sha256-c9c42ed616c66ef0807e52ed7bd072ac9142b259071f666239c5cf3afc434b2d |
