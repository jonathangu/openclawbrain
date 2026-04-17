# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11280502-6329-42f0-a48d-32811febe3e8-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-56cbb762872ec9be931577e0e0cff6303eeba2f34a75a56a7b3caac6fcb77a1f`
- fixture hash: `sha256-94342c3c881533866c5dca496c9a26c188cb0d64c2968ed52c2a79ce1e516ec2`
- score hash: `sha256-51c8a1b23612dffce633187bf0432a97898018a3f0aab3389e4670bf602923b4`
- bundle hash: `sha256-f9c0571e0da0613a23f538a195f1bb6b9bc28a698042f249b9c993abd4ef766d`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b47b12d5a1c7efa8df449fc327f742bc9ea2e78e636eae25d8d3c474db1900a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c4b74fc6022413150d5a40e8b1228228de1e032f59085a3f83cf2f27cbb48c8e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-901204c83715a12f9b91e245925998af1b9f5e2bf1ab086d6f6c7d2b23790433 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-33fb48467426b1d05b3cfcdf2a0f1e8e3d5b7bf2c896918dc3d5c9755b69cdac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-242d4aee | sha256-27a58a2f353716eed61769efdd8dca0b910eb2ebe09dec059d913fe6c31a2d34 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-242d4aee | sha256-a6113eba117c2fd311edfd49d8436f14ad5f374ae779a63ed48577a8bd4b893d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-bc17c331 | sha256-6c1c52ec0e9c8431e46caf6f81b4374c3415af043ea4d70b96c8dec3c19ccdf0 |
