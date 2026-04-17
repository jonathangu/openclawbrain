# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70911ec1e7805ccec970087d6c2246db12da18117e08ef4135c17a78ab963e90`
- fixture hash: `sha256-5f3ce437bc5a34220be72a905054c7058ccdfb9aee9afb407a944b39db8e43dd`
- score hash: `sha256-ca4505dcec8bbe784c9fd97bd2d36e2ac8661950a40142faa814efca4c2a08fd`
- bundle hash: `sha256-196f3e03d4fe145191023f80d54b222fa57ab6408dba6d9f1dede55ae57ddb13`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c5564d3b460f1097de136a88547e9b2bb9e15503e1a0ceed301551bb8e7b5353 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ea600060f25b29041e592417d5844c5a9559d892829476758d6a656fe04ab92d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8767458f8d42b1444d12bedee837980895c18bc321cffc3e4d4bc30c3385b145 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f5419cdc33b46db1999c9a4366c610613306951043b9d8fe9e9418abdbc876a2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5102e7dd | sha256-5c2816a96a0f6deac0985102a9986ef2f67e35ae52bbfee6c9956a1946e2873a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5102e7dd | sha256-da3e8c5e28217f04673dda84f5ba006abdc85ca87e9a1e8295922782cb2c7e0c |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-80c66d46 | sha256-05f0d5abb615d9d5f281a3de1ffdca75b622012ea8c33249898bd17886c9f0a6 |
