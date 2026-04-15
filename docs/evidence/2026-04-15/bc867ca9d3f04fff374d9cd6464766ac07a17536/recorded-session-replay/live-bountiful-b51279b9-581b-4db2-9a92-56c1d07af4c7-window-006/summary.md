# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-b51279b9-581b-4db2-9a92-56c1d07af4c7-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c6d060740a9f2e795426f379a0219a66e62455dff3645ae4e75b99e7bcfae0d8`
- fixture hash: `sha256-385f14a07cdfe3ee9f8100f14af1f7e22cea10fdb4fdaa52492b8887adcb0a61`
- score hash: `sha256-4f5ee28dfdea334bf4b2295ec8f6d14aead853149757b7575e5ef8ef0ec103f9`
- bundle hash: `sha256-3f26d1b2e775eaccc30bdc310fc65c0ae70401d8fb1563ba2ed64fb33e4d444c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6c4c247761877c56e63a7b06c8698a61fcba3b5b8c21d71224fd41fba84da376 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-25826e68d4ae060c987c1927e414c55b07369d7c26b603770323210ed4e06086 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-afd6659e61f7799dabb41706eaeb3f4bc3cc7313d6a763e6d05b74559ed7681a |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-fbeb5456cec7c9861877cd0e612f4c67d93e7fbc387d94f313242d267dd82aec |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5269769f | sha256-eec0c4e4ffd548cbc26a465c1b16bd794413b20d949d394ea826903bc15577fa |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-5269769f | sha256-0094630976d1731d3dbf76fa6376b19c852c5990163c23ff777d43c0d37beb62 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-5269769f | sha256-eec0c4e4ffd548cbc26a465c1b16bd794413b20d949d394ea826903bc15577fa |
