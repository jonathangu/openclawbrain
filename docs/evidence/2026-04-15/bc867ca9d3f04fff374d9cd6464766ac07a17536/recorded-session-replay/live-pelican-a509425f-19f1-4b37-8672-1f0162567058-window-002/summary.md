# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-a509425f-19f1-4b37-8672-1f0162567058-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-34a774cb3f6c8a06b7737a6a2929058386a540d4a4f6fa06d56dab519cbae33c`
- fixture hash: `sha256-38158baa488957f4efebe2494068936f86320ad50d0d4566b804a6468d20bab5`
- score hash: `sha256-c8dafb277ccfb743d89988261fea2bde7302167a394b7df66e1ef5f29cc6db9b`
- bundle hash: `sha256-1c3f76ab74b566e640ffb881511df0b097c05f6dc647f5b0135583ee67ba8cc1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 60 |
| 2 | vector_only | 60 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2a941461c5687ced5f6be63f00e8602b946e4d86dfa5dfb8e215a577d1b9170 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-db18fbfe59c50650ffaf9e47aa046a0dc6adb1480590dbcee16d264b97e05592 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9bf9b56206827f4e9836f36770de90a2abb5cc1f62e021253a800d888b90c23f |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-8957c00e7d8686c5f13f3ee657a0fa376aea495a233f29fe748cc5ace34a011a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b4029aee | sha256-ab8fed97a782161e15ce0c3a5d2a1326cbfbf726d282ca220b4a6002eaf63cd2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b4029aee | sha256-e15cfd5245da18dfa4373fb15bb900c41d3881215ce0c69fac8e2d54287007d1 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-b4029aee | sha256-ab8fed97a782161e15ce0c3a5d2a1326cbfbf726d282ca220b4a6002eaf63cd2 |
