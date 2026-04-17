# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7ce9d77f8d5b34f5d4a2ff238035837b9a17936c8718ddcd44e0135af5ed67b2`
- fixture hash: `sha256-b278e1b6b555771ff403bacda1c9f56aa4593110af14f3b45502af98316b55cf`
- score hash: `sha256-6f7b3132acb9aad9b3b89b31c40bdaace5209a11fcd435eab70331e1358a5a1b`
- bundle hash: `sha256-729e91c96e2fe5c505a164cfb19e169b7eb33fa7d9de1ed3a9ce60a4663af65f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1fde92dc9f75c3ed7b5cdbc92af57a8fdea90f988cee9df5a6592eb109fc517c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72c68215e1b7e8fe10d4327b0fae103653ec331f19bcdb3e4269719130243956 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d648894c585d91e126e6e78a49b3f48438b215b400325f48a8f71d0f5cd82b3a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c9bf1bf24ac1c28da47a4b2d40c6d4cf4cd3be16b2bdf2ccc619c77fd4bcfa44 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-03209a16 | sha256-855825c9f9562bb5d7512f278ae016e7419db742ec6007f3dd164d8693d2c730 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-03209a16 | sha256-8783beee105a87e021578475e949eda7ad4024ed168e099ca3751991cfd44477 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ca16e4e3 | sha256-7cbdbeb7eb982eeb5ddb07ede878e3b3ca938bbc8d9024df1023e385ea34e7c8 |
