# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-512c430e649faf76044870db1348b61987384f6c4b42eb2624038c368ab6a4bd`
- fixture hash: `sha256-6f2c5641408f7a03798669e19a288492bcf8f6f0b8043e459e2c72b4bc2ef9f6`
- score hash: `sha256-8f916521fdc352eeabc1db41e2476714e004d7b99eff2e8f319816d405feb012`
- bundle hash: `sha256-6fcbb3f1a4137600883feb0e09369d584387cbe65692804403ae1351aecbefa2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5fc8b4083586f10fbbdda0686c1eb4cc964fe1c89c35a3824fb52431cfb03e36 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-385e3246b728766b06607dbdfa75fc4d2a9623625b3790c10d746a9e09472dbc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-886c3c08288034c228d603135006981acd2a1e4457a200c35e140d7e396a52d1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bb3ce356eb95a91eed73a1bb8121ce1983fdd37876f991ea7520862ea2844391 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bb6b86ce | sha256-46527c5e66733fce773b41c51a3851b5da4a4e029ee6c43e5c4392961d9320ba |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-bb6b86ce | sha256-a5c446bd8aee1a21f519a014c614b10b7ca7bb435e45298117aa12dff2cf711b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-bb6b86ce | sha256-46527c5e66733fce773b41c51a3851b5da4a4e029ee6c43e5c4392961d9320ba |
