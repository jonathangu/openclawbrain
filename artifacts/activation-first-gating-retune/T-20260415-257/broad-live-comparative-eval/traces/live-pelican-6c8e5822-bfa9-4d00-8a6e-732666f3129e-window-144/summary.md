# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0149e3caf93b3375cf02c24f74af73ff26b7bc10ea672fea0331d56ac334a82f`
- fixture hash: `sha256-8eef5aa851168050667187c6a1f16965243d4107da455697233fb94b6cd8be15`
- score hash: `sha256-e2e83f327033bb4e4363a66ad871fc84c86d41ebdfde027e6cba8e3d0ca5f260`
- bundle hash: `sha256-44e3d4ecf1f674f275fd77b91990da4bcca7c0185d52d64bc8f2a2db070fcbd5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-beac0c48f82ed7e8a11f136719a9c12038db11daf2070f49f0ee8d4c618e927a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8374d91f8396c59b183833ea0eab62e720c60660cf33b6a520f6d7cb12a1be6a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b6f142aa1440aa97d43a5748239b4add25f1e459b0d7b86d48357a2463a5f459 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-69bcfacc63add4f86b36234ecada5730a7f96020d73fb8b9827bc6ba2dae7b12 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-85ce8209 | sha256-9ec02458cd8585990a21a1bf3cdba511a4cabc7225787d1a068d7e3e59c97807 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-85ce8209 | sha256-082a73e807da4487a0e27412af6d7c965f198c992f4815177943ea1e86160cb4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-bff0e596 | sha256-44a5aa092a455180c2ef853d8b9276326bc773993ec1ef47a1146d9df4cf45ce |
