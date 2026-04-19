# Recorded Session Replay Proof Bundle

- trace id: `live-main-6fc9b209-69a7-4584-9093-cbfb2cfb69af-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1238bb817085e52d5386a747baa6ea8bf61e3a37516af898c3b116b0246d9843`
- fixture hash: `sha256-edd92cf0e628f6e0582722d507204fe8af0abb5e8a70f6ed2001e47aa93a6a45`
- score hash: `sha256-c62161ca5c8326b0526463872712e63ca09742b900c0383c1d394a93e86fc19d`
- bundle hash: `sha256-5ed7b998d20df8648d1088c480614204cd0e0ea8f1c56084b68d9a190f35b120`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d2f9bcbabb6e41c0be690a68df09ebb71d4f854521659c85e60ae6817b1b9042 |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-7df12f2bba1e97dfaacd27ee1c72c200cafa8b5bcba942dacc87c37e4c1535fe |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-a4c6fd7df003f8a5c723a629cf4b94f74470ba322b2cb88e9a95c4ec2962e3b1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-64061844c723e7c76dfa124fe894460d1079220167aea68632945c0648a9f2f0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-a61af84a | sha256-6f7fad58d093fa621648664a96ecfab6905ee802bfd7065054a562ab4aea59e9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-a61af84a | sha256-85f29765613e7282b57cebc0548feda5b836487ab11e47da55e2fd96486c3c07 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a61af84a | sha256-0d7fd21085e0aab7415d4fb93170522dcd284790c3ce2bd2693bcebe293dd707 |
