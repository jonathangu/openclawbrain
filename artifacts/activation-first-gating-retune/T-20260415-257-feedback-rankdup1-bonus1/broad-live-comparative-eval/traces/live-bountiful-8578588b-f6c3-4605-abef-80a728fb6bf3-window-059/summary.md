# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-059`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f67214506d27201aa84ea9f3606a2dd530633ac1dadc31f05ed3e5af4326650`
- fixture hash: `sha256-a8505bcc6501d6f6856db49c4b5c901ea18f4f407cc8520314fe72513fbff478`
- score hash: `sha256-967b635c9dc49cd5ae34db40f16023e91655f2ea04ae7f38aaa03f35be7b7e3a`
- bundle hash: `sha256-e77744be36cbdf6e1dd5b92dc1551e89646a63a6dd13498c2d2ffd23462e1b5c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-44d92ff0a09445bae66b58b53a8f96e059e7981c4a7d5440523a3b87ed99e3f4 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-a35349257828e5155789e023897b987aad83477c1e8a85522ddbcbd2d7930f8a |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f957f1aedaad99f3ae695e94b837ec224519f0469f1bb59c62963f3b0394c52d |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-fca14d6b3acad85b3047cce1e975255aea8d121194f2f15ae3513cbaae40af69 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-48ddd351 | sha256-974f8272d4c2e1d66cb5b3d18e2a0a5837db651916683125adfa17365da9fbda |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-48ddd351 | sha256-463bb831d4184c0abf2af85d8de5254bd04161682128295de7f3b6245b26c080 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-48ddd351 | sha256-eff4ad20f599b48dfd401e40c842d92934c2c75489925d04a703551fb5f176cf |
