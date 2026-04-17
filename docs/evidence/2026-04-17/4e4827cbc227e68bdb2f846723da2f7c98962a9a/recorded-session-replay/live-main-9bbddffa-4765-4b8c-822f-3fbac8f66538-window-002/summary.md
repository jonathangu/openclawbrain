# Recorded Session Replay Proof Bundle

- trace id: `live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d82a353da8dcdcb26be266a0d33a583d117e1f2075d582930b7ead32e3d715fb`
- fixture hash: `sha256-3e1329cac030395635745dbbbafec0f460454aaccbb63b26f155aed0ae65e7c6`
- score hash: `sha256-479f502820fff309de946c36f8e2cbab9a51dc5b58d82eec7b260097082eaadf`
- bundle hash: `sha256-f8dd6ad0a6dcfdfab527c56b8ccd9275c2be3b3acf06a73e78374d46a20065f9`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38cc6030cda1cec211011cfcfcb3fe3c0763917e1cb1cee36ef6155b409ff4f0 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-df29343d14b4a94a2f0fae99c6399debe75a9f9468c02a9bf2c0ceb860934a2b |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a23be78031752bea72407d49c8c01d75459b91e7e195cf1a1c0d7ec50a91bdf8 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-4256fdef2e92d3927a91469cdbcb226c2720fa15f547c669db1f06e7adedb8d4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4a5113ae | sha256-da775a2994fbd6715afa7a3982cebf83aabb923124148909e9668ec09d1fd82f |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-4a5113ae | sha256-1376dd1b9ce652a373d44ee6827d185f47c028913329b6882727e7a775a95221 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-f77fc859 | sha256-206a906a8b0f32c0e4eeeba1350512da2a6e42d5bf8809baeb083569f7faf7a9 |
