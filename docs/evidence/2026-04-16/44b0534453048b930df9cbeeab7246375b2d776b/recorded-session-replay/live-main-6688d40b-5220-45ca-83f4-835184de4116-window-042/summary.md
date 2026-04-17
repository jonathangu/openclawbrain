# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-042`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7d50a8bfbe12d6ec52d00a65d5c5309711fc92d4bd65677275533c95c1fbb9f9`
- fixture hash: `sha256-486866769a6220eac0c25d8477d823ddd1d78a29159bb789869bb12cfb7c0a16`
- score hash: `sha256-d59aa5afa8e9b888f5a021350ad510c784b59ad43a9c0d2e5bebdf78ebddd376`
- bundle hash: `sha256-cf412f3200cc19dd7564253f7cea52971ec6b6df8f9ad224fbb333ca2ac4c508`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-030b3c11ef3b6ff56c24da96c3a7b6b56306fdfbd30d56345e3f6aeb18dc6984 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d1ce65a1327d81aaa85b030ae6987e3d04d45dea97f831c631e8409029ed5b2b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb01ce654b0c28e71e1e83b7c430cfccc787df02ac2bd92343027ad5f001596b |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6954f565b8c5a2f790e386842bf713af5bc38a021f16685add93e5ea57375398 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5baf1639 | sha256-83c1cb02263ed2f0f11057bc94714e3c68ad66896508a09eb0bd47f4922c7745 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5baf1639 | sha256-a1eb8114bbc1aa5ccffb5e49a998c776df8d2a94d102fc4864b4f9558c36ea72 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c41b868a | sha256-707d842d64f9eae98f472fc9a564d0ea923c32fa3937a53de1b7d11c9dec5a6f |
