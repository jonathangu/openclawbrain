# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3f86fd026217c7d6458e87e96268ca58f7633ecf498ef1f8793a6a7617c13f8`
- fixture hash: `sha256-c25bf3a6bec00b35ab13366d1787d21cc5e0fb28011aa90689176fbd43238498`
- score hash: `sha256-30e26bbfba9d2ac3c1152a7b74f7808242af69077179ec3779cce00bf6355e87`
- bundle hash: `sha256-897b0a035db57a90a008f5de668fd069ee9dde3408f76d8a7853d61ac26bb180`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1e36e95d3b902dbb1cba84b7196a751790c689dc2e631e7340724bc6d85c3a59 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b5e4c25cad0940bb332523cb5346a1ebffa56b075888597d6df3913203c83d83 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7d1879f3133f1f4b627970949a51a2d04bffbea2ea3d061b92cb76e87bc76f8d |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-d6d59449fa06c3ae6fae73f36af054e375d606885d210cb792fc92bf3fa736a4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7d52afce | sha256-bfaf913570b9d702a7508947df3783cd1e7694b5c14bfa445df9d61a3e8cf989 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7d52afce | sha256-f89beaa2d3abeb1e984a40963249d1ce945e2b936a6a4ee936f52ea4eda6380b |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-fa02d373 | sha256-ce5ee5f5349647b4f261fd1d3932220c40e0eff0b859ba557deae392c5076a43 |
