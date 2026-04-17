# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-5043ea40-b106-4937-bad1-aac2b5627b91-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-347a801443da9e3d23f8dc976f3d286dbcc3cafa0984aebf1f93ff8efbfd1773`
- fixture hash: `sha256-3e9f54e7049625692dd39972563612e44cc8adf4a2a27dc80d450c5621a5caf7`
- score hash: `sha256-57ad1d0a8969db3c5934c4f830027787a04b457de81d2c2ad30fef72e777ec7d`
- bundle hash: `sha256-03fbc9e61f9b077fb69f48019ac316b95f5faffd390615a69d89713c6203042a`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-bc058e6f191036e6bf4f3884982c6a502fc3d927441bbbd1c5d745ba4e254aee |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-711a706a1156974c2a6fccee2ca88c877371fded377ecbd1026f57deddbeb39e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f1361652522201533b15f01730ce399864d2afaeaf11ad487e11d3714f8f001a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-29294e6b52505709f19269d63f7e4a865f437752d89e41f21d4de09fd4215706 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-040ae271 | sha256-71a7342a6968fcb31543b62bc6b28e79652fa3e461ffe2674913ba97e24e066d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-040ae271 | sha256-71a7342a6968fcb31543b62bc6b28e79652fa3e461ffe2674913ba97e24e066d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-bf605f9e | sha256-97419cb01a10815a72898c95e18157e357dbe95c4f07fd564d136d66a092ea7b |
