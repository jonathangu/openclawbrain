# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7d50a8bfbe12d6ec52d00a65d5c5309711fc92d4bd65677275533c95c1fbb9f9`
- fixture hash: `sha256-486866769a6220eac0c25d8477d823ddd1d78a29159bb789869bb12cfb7c0a16`
- score hash: `sha256-b4227ec674c26d49c3402f78fff7a5a6c94e30ff6e21a5bf186d450cccde268c`
- bundle hash: `sha256-94f1a7a837fcbce39eec3dc148f7d6badcd0ce3092e5371573cfd3e523b45d80`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-030b3c11ef3b6ff56c24da96c3a7b6b56306fdfbd30d56345e3f6aeb18dc6984 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-64fc685f305d2cbc9a851124e9ec1f0b533e7486b8826c33da38a055f5ccdf80 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff91af6154264b46468292d6dbc948aaab8f17acfcd7243d10f47b1f11625d88 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c2fee270b46e3b0766fc8ef1c6abf805ce5b833a534bfcc6ed26b74f1238995e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0060b1b6 | sha256-8051802b007cb00d665d2ce04128f87658717e78c70909f326189cb42e29fda4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0060b1b6 | sha256-09ba9e360dad397a976f250de6dc653f729634e4240c2a7167f3b00846060573 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0060b1b6 | sha256-8051802b007cb00d665d2ce04128f87658717e78c70909f326189cb42e29fda4 |
