# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-012`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb3f29e706c8408e5460da5ae181547f400604bd45efe4b812bde36a617f82f5`
- fixture hash: `sha256-3bf32dcbf845b428f375103144110fdafde5982202bc1871fff67136d9720e81`
- score hash: `sha256-babf0669c9a8facf0fdfda9e5f3bf4cf4f6f97e87996c772f1590a588eae456a`
- bundle hash: `sha256-6ab0271824eb569ad50d7470c95c89e3ff3082349a374a93a44a8fb913c746c1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-865ec1deeaa418471d6bb216a38e6bca377292e05c38cf14fb63e270894197b5 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ee551029994f4a1cfa4c46f95c41fffd9c47eaa6f894bc7f3542f9834b853aca |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f3e17c501940465197177e36c092873a3f226c9f1aa30018e0f72c02c0f8e105 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9dacba315995f31bf93eaf72c20fcfe5456691321808701d9c868685b91c3dcc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-01e7e1a2 | sha256-408a8d408a18987d40d0616d709708243fa5977867766e452dab6ed2e3b3131d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-01e7e1a2 | sha256-9460779aafd93930996f2830a58c3af755b2216dbc98e236467a1911a3f775d6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-a7ef34f9 | sha256-9bdfe609a4ee3abb00c05d5ec88dab4ee3eac6a9e5382cd4cfcab6498789995b |
