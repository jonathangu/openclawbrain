# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-016`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3356e25999580c1815d0ae49bf40bb5a370485dd768a3aa1572de8bdcc8cba97`
- fixture hash: `sha256-0729bfe4197c5261cfcc1c8ec0f8202300a63367aac183bee3a483ca417a77a1`
- score hash: `sha256-4c7d23091f85403e4aac67b164871389d906899568e2f2f0a43753482cef80e0`
- bundle hash: `sha256-8e0b65b3c94ca55a8132a88a1ed4a33cafcd98c575ccab515fdd7e5b1a87650e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a4b42315dd1b1a176628a7fe8f13cc2dabd4068509144906d5e0b01e2bdfcba3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c4a1642fc9b7ea7e1300e0e1f33394ffba9ac7c2c6e6a83e179fe58a5003c65b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-98d0359b7aedeaddb08014754d22f3597a1738ea4aa1bd240ca55047fe0be121 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-435a8474c046ca52dbab83f98e8174450243f710731eb8eb2c19e919b6fc6705 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9fd84abd | sha256-4e49e03c13004386063cdbc319e5a14b71672054435ef9fb5dd308e01e282a01 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9fd84abd | sha256-7577d678ddb2b89a9f1c05095ab4f71e15090bde129dd6fb5e24d5c56f5a8df6 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-9fd84abd | sha256-4e49e03c13004386063cdbc319e5a14b71672054435ef9fb5dd308e01e282a01 |
