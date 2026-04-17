# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7d50a8bfbe12d6ec52d00a65d5c5309711fc92d4bd65677275533c95c1fbb9f9`
- fixture hash: `sha256-486866769a6220eac0c25d8477d823ddd1d78a29159bb789869bb12cfb7c0a16`
- score hash: `sha256-87612f1370a6444f0f8579441822ac8748fa92ad29dfd6ae62eb40f2c1367fc8`
- bundle hash: `sha256-6e0f42a847af0c17b850d2104910d94708b8bec568c5c6ca2e21e9871c53aee4`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8a23fc8214fe328824610f86abfb25f9fa3882725deb7ce9e621d258b2aea631 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b7a9ae461186f30b114411efb5060f203a0f3fd57536cec3e7a5880c6dbfa520 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c5a8bf898c5addb389a5f5124db572b8c0acfa53eb167c565509f46c05fad0de |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-49981215 | sha256-a51938b9f1124cc5110424cd78251f54bbb033b8654e891cfd82abd09148d72c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-49981215 | sha256-45e8fcc3636bfd3f44622b2ab02884f4ad90c092b7734a17745886c778e75045 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b2048266 | sha256-c1333b3fa98bdfd30b9196ddf5a04ed4c45b9d737b92062c215e67c7dcddbf89 |
