# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31afb2df9c1a17ca25197bd8dab4006e37b9c5d5cee2757703f3aa5a6af3cc63`
- fixture hash: `sha256-998f6618e36f06829cb18a9eae15dbb334b923e47c420cfa28a2642db4d68155`
- score hash: `sha256-11baaa1bb3a6838066e86bd1a2c5ed4878ca83aa9c34f142ac1f102facd93304`
- bundle hash: `sha256-6af29aac8fe52be65608e3d3a27ce309ed8c4c80a304fcabca4f9e10488152ac`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c0657d7b41d16e0a76bed7b5e5dcdadf4310444b0556eb5e7411f6141dac5dd0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31f0bb538b7380a60cb87c0043655a401f8812aa41b63ae686a2469729b3480e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0112406f90d4832c415e1415c67de697a3f10a15fa6ccedd1bf7637e3501ee3 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-1ac0ecd1f48125804518e5c8fdaa8aad437b158d151f1a4de7e2d4e445a22801 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7cbcb17e | sha256-afded897d168b2b3c6ee5b351d6415a45fa5087038e7fbfa9b23fa7002db2b8e |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7cbcb17e | sha256-0918f9f9b172912065b51b5660cb4d506d816797fdd41369e4c7b73cc0b95c11 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-0e5bce25 | sha256-1e9830a66b288ffa167b6c3ce69198c21b57aa8ee4ed4eca2e34b88ef438f04f |
