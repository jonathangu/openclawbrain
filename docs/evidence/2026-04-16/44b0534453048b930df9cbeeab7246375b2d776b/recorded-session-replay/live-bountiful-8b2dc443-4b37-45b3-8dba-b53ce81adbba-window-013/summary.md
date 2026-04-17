# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-07e10f6820bf810e1999011ea58316d9d53ec99aa0ac7473d30a2c9a79d153ae`
- fixture hash: `sha256-54c4d68b5e528e2dc7ad50c599fd75e1b659a972d8d4c97376e292a3ef62dcc8`
- score hash: `sha256-1eab2cfb3e459f0b2543ccd8e3deb04323fb9818b2eceaaa1c557373c7acedca`
- bundle hash: `sha256-ac343034dc41a717c92219c3a60eb473f849ee51f930b8beccbc9d5d239d7630`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-430dcaca40205cc8d42bfba95521d8acee2a6e6c074b542cd0b9a2d9f1547939 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-22e3ecb0ba495858e9884c205d49e3597d9706c873043cb4b8950728a25ff562 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5a85ba1e26a4617bdcfdcf4d8e93072e26233a6dc3f46c74aef2d08501289747 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-52a08ce9275e81c78a6b418c7a1139b2c172eb8db0017a5e062a6dc0c7461d18 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-199d49a4 | sha256-67b867996708ebf9d62180f7a54ac1f03e989162be4c2aa159a26b6ed7e9f04e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-199d49a4 | sha256-6701f797758962ed26f308063cca2f224d32acaee506efdcbc4d2434707c9064 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-dec86d7f | sha256-654a5f48daf0e184b9190388b9af6116df1e933eaaf3a095fd453bbfd1436dec |
