# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98f498b917834ee9c0a78d5b62a338d5c94ab2df87cb501ae8615cf42d07619a`
- fixture hash: `sha256-54eb8df766feda2c6211a7171b884e66a0008ed710f7d28bcb6341bc861e92a9`
- score hash: `sha256-7f6ebb5c7fb3c4faecb027e993cfa24a3b0bbea7e0d8f7976cbacfed8ee0ea27`
- bundle hash: `sha256-0fa9efe6bb78daa52ae6a07c739a20807d7fdb6dc4a688da563c075051779a4f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6670af2aca1f1e71cbf3c0f145ce7f96dddb89bb0330719aa7609642a8108f9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0e81ceee547b1b0cd70db5f7659b58fdc14c2eb56d53f69c9e2a99acd6be8932 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-993dad0379855617e1263a97c7ec849cedf4aa1707560a374f3d3a24908dc172 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-da23c6fddf173d94b21da439a24d2fb4e9ea87c933a4089963f3544ccf8c996a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e73aa162 | sha256-a57c24bee00cef2456573029015109e3d9cb5d56fc30859562f3f02e8c9e8b3d |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e73aa162 | sha256-4324ccb5ca75ae92826a7c2ab9acc5412cbe76cb78afc8ad22f023b2b155f812 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a5553117 | sha256-efe861de1b1288693116123ebf597be77eb80c58fecb9a7db4a4ee8aa7d06762 |
