# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-009636da02f7f67100c4558c66177e3052d69c0fced9d7e92f816d385fb5757e`
- fixture hash: `sha256-e98441576455ea28abee01372cb0b00d04c2271a6e52b08c0f8f71e05b4805fa`
- score hash: `sha256-9b60ad0334da47956203ca9590337eb5ee08a562e429a2c1c83dfd00f93aad7f`
- bundle hash: `sha256-ecf4b5b239b1436bf0564dca807bbf15bb1c7023184f9c86bc2a802032055ca8`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cec99f5e1ad091e90131ca937eb9886122311480b894b2b01cfc694105b3de60 |
| vector_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-3bb3137d19cec5e79266e4b783525f36d37e5471211135a3001ca4da0cf63277 |
| graph_prior_only | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 1 | sha256-75f6ffdce7cf91896a637882cc5a81cb7916d4b40dfef14bed6f0196858a77d7 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-d1e28e42c109905b87903f02ffdc07282edc914c18ac4aa595ead8a8a290f712 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-a02ba616 | sha256-7614e4b098578b3d72ad867dc63601f833a4a7010b9ea3a307caf14f6757a2ee |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | yes | no | pack-a02ba616 | sha256-eecd73cbfd716c04461d00972e5637162b53f2cca864c8f24e4136e5b65ca230 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-a02ba616 | sha256-7614e4b098578b3d72ad867dc63601f833a4a7010b9ea3a307caf14f6757a2ee |
