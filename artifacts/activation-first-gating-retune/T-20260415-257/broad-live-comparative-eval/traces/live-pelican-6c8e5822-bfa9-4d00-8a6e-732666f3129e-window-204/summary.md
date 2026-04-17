# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-204`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0c07d86727cebfe369bee33466c114948863aec275c3915842adcc6210ff9f00`
- fixture hash: `sha256-8b304aedcdacbc80bf121116c28b99b2494a777738f36a03524f91c39297ceda`
- score hash: `sha256-18eb98829f03709de0e6199ab320a9b8f4df09e76130f8c0f26dec798d239ca0`
- bundle hash: `sha256-1f13151314b0548a9aa9e01eef8a6ca5c523859c54912a1fcfb3377da30f39d6`

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
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-fe04b23af19679c2e6c9279567d3e844fb86e40da2ea805ccff0ac7ef1adf141 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-04e51d43d24644e79ef02e3b13347815240be52f9bb2a1949cfbb47a937acc45 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-bb1ec9d74f3e046bff7ea2507e45eeda21858e1570d1b6ca237defc4e9f7a30f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-00121b8a | sha256-a16f5e4a16de138ca5319d9e35170d8550af8315000a129094f4facfbc875c2b |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-00121b8a | sha256-3d5b7584099e48b7fdc99d65277d5f2e2812bd23a96c62655a5bd2df20baabf9 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-9dd890ab | sha256-c0e4b71c86e273a05e2efe559e012735f01612fef41f4c7baf6f123dbc7940b4 |
