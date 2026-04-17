# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e2a057c58ceb7779d689dcd4238dfbc3207e352fc341de03ac7a06d504301da`
- fixture hash: `sha256-737f6561e785d3bc05d3981f983d5cf16785ca63d2f46199fbc1baaeee1f2b69`
- score hash: `sha256-117f95900ce501eea68a77a7895460708e954bccd40cb77d739d78b0d7651078`
- bundle hash: `sha256-5567c3306422fd8d4ec9058c0a7e148d6b6f6fb37248f833f6d3f9f3ed765267`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1e7a79c157dc055e3ad83a213c22e42badb5ac82b3ed30aa50ada887959b805f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-97d8b9e8f8eb8dd875d526bcd7abfa411ab5fc417a3a0929d23e561c0e50312e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a5b4134f865b1b750847a0ca7b05ad6f52583824ac93927c12a6ba1ba9c87fc9 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-7af32326d8ad2eab62e4a488de30ef11873618537a4d8f0811307c0ad27d440a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-97d62558 | sha256-84a66a88233d6f71405a0996f28459c94b93f431307c9c756477ab9760907a50 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-97d62558 | sha256-e0d12b1ec3830acee4a40a96f775bc6393b387637336899b30e57aa8ebf915c2 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-b944552b | sha256-e48c12e3b4de51f7d61d35164d17816997ba3c3b07bb02d380b638496096cab9 |
