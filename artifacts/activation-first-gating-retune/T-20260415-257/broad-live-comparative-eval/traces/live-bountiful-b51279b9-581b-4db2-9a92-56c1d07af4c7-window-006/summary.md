# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c6d060740a9f2e795426f379a0219a66e62455dff3645ae4e75b99e7bcfae0d8`
- fixture hash: `sha256-385f14a07cdfe3ee9f8100f14af1f7e22cea10fdb4fdaa52492b8887adcb0a61`
- score hash: `sha256-cbec8ac58cfdf67da702cf37792097eed90247ee7b4e762013228a92491c25d9`
- bundle hash: `sha256-b1b2a4962d7e2a3c979dcc36b69f6781687aa52578750a95bc39a164d6f33294`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c4c247761877c56e63a7b06c8698a61fcba3b5b8c21d71224fd41fba84da376 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-1b8a77a3909be43d246a616db8865f374b30ced3fead62aa1996dbc9d453a235 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-6d2b8d71d8de438b9ceb15b85aafb5e9dedb6d67a40723c7bf1acfdb268f8551 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d8acaf71b6ad0734f9ee3a1103664749012d21bd75ac851e47119647f6bafbe0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-27b40106 | sha256-a60929b6f5bff97c1e60353605ee99fcb797ed1dbacaf476d5a06bb620c86d66 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-27b40106 | sha256-094d700cb78bf2c9fb8b30bf8a429138ad3de70169c11b430c1a67b04f5dfa57 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-27b40106 | sha256-a60929b6f5bff97c1e60353605ee99fcb797ed1dbacaf476d5a06bb620c86d66 |
