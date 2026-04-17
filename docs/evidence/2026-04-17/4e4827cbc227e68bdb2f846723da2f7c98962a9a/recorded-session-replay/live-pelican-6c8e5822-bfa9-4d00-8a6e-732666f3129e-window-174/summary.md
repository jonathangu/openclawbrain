# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-174`
- winner mode: `graph_prior_only`
- trace hash: `sha256-681abae2fa5f82c72e9292e394b227021bc61d412148159906a6b997f617cca5`
- fixture hash: `sha256-c46f6cfeb0331761c1d2bb543d4b028a9a876c69162435d955a285bd82156828`
- score hash: `sha256-807587f0e7912ca0854c9600cf90975f620e114fd66facbbc7ff55088db4bf7b`
- bundle hash: `sha256-00c0374509d5e6bfd1606f790d7fb9ef1b052eb7bbd701cdafa27ca11fd5e712`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-461e5f784d6e942b4cdd1338a01f757f830996458c9f4abe17a0effaceafc63b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-250b997544a4c1ed451436abd1ad449a8f500890e1ffb032d0161050af14cd6b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8c3d9078940bbeca48aad9e13e384208db04db96f4683bee24700f595d3592e7 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-753b0185c150b61350741cfbef50f4c07fc385b0b65b4268fe8e80e6ef19d1f6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2c8a587 | sha256-9552aaf5591c2c78e1de927c45ca88093f036c73c3fb21d3a90ed847fa526c6e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c2c8a587 | sha256-8b677b5a150276a7f900cf3edf6c9db1b57713a2778f6194b62de652af58f72e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-808b7504 | sha256-910a5731ba59c6f22937396d169276524320ae15d0decaea69ca9d3869f068d0 |
