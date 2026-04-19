# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a725d997b61c014cfa09b0556fbe7457ad1066c9e82be586f6e632957134b68e`
- fixture hash: `sha256-0175601a40ce3b110cb977baef20750f4bd146a6b251d6933e92137c6f93984a`
- score hash: `sha256-8b12b6eb46e9f9554a845cf31645405802ce3bacffd9c3363d88290be80ed4f2`
- bundle hash: `sha256-488097ff960fd9434fcde66cd66aeb50bf49673b9d0cdf9deb5b57fedb642712`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5b1c6095a0bb639e8d10640852b232aba3f9944783712101783106fcf8add6d4 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-8e1d200c3aa6b573d3a6a5f687497d3e1e3ea5f0e83ad077c276c9dbb4d09670 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-38db7e54d9b8fe6f0fa49092277bfa2a3d4a51ffac577fd5dce277cd577ef0ca |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c34e6b2f9aa803724b540b137359e121b4cfbe0d52aeac56ff6a683b7cd6dcca |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-25f278a5 | sha256-2a68edd48f683cba20a4bf00e894111071e6a515dd86bc22c5c28a96826e7495 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-25f278a5 | sha256-f6f78f73a9024c2cbb4988fc4297db29a26d399c3930b9aeb46b4d0a027df2a1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-25f278a5 | sha256-2a68edd48f683cba20a4bf00e894111071e6a515dd86bc22c5c28a96826e7495 |
