# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c591f5c65a17f8b728581b0a64e58f54d550808c4d6d87e9681919456c4e7956`
- fixture hash: `sha256-7d10d338cf842d955d253c17c711f61a919941b05a2192e292201851e3214a2a`
- score hash: `sha256-83c5d21a2a6aff104afc29e952dd5e423c613983e67d79b48f6b1dff1bbcc124`
- bundle hash: `sha256-b295547520ef631170f60b20a58f906a185971deb54aafa062ec9d4d05da76f0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dc716d2fe029608c9fe52ecb8defed0c2de7ebf60cb8d8503f70a55a165b4d33 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e90d471201aa7051c71587be92dba11bc34a0b9535dd2b81f6e32f329f8ccc57 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-35a100e21f45359a87a109ce91b479d8140f228464f047f300bf4a00d3648b80 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7972164762824962be872e0a8e66b65696eee70270d3aa595a504a912d3a32f7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7d8f2ae4 | sha256-33b38bc4a8d7d5c4fe708d3fa6f9a6e99a28f56b7ebd6e896882214afa04ed38 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-7d8f2ae4 | sha256-9566b1ea39622fa8d4cb2c463742a27fbdc2a1e853dec23c8efc921620dfb6a7 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-e6e4bda3 | sha256-dd351ca50c93a4ace108c5559506ba808344ec8baaea7ea6d3a454d21ec58f9a |
