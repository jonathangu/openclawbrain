# Recorded Session Replay Proof Bundle

- trace id: `trace-plan-proof-artifact-triage`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0cf8cb60e9ada33f76aa46557e599749091e81d486af58269182793f5706eb5b`
- fixture hash: `sha256-d2de95f47483d717328b536c37387d67416a65c6ef4706bb5e261a3f2a08590e`
- score hash: `sha256-b89f26fc3b64b2d1cee92202900e7dee747e06f7f885a5fe2c5e5909586496dc`
- bundle hash: `sha256-d60516727af97e02bb7c720d9436a600a75bb7b7a702c1c81c9c415822c2e077`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 18/24
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/6 | 0 | 0 | 2 | 1 | 0 | sha256-91cb4e3cbbf02d6e28f4e5844a1cdc601a411cf0f2238175efe87f3839231475 |
| vector_only | 2 | 2 | 6/6 | 0 | 0 | 2 | 1 | 0 | sha256-38feec3d38ef80a68853b1484162ec84e7cd4efd0e13d8ae1000b1e77df0dee4 |
| graph_prior_only | 2 | 2 | 6/6 | 0 | 0 | 2 | 1 | 0 | sha256-d25d46fd2b1ab3c2f7acfff190b2eef2aecd1534c15939a683e019448cf2d7f0 |
| learned_route | 2 | 2 | 6/6 | 1 | 1 | 2 | 1 | 0 | sha256-b0a2ae22ad9d8714f71cf04447c6e3d87ed7cdeb77cbf94777dfd0b769760bb1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | artifact-triage-turn-1 | 0 | no | 0/3 | no | no | none | none |
| no_brain | artifact-triage-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | artifact-triage-turn-1 | 100 | yes | 3/3 | no | no | pack-416f2acf | sha256-19092869424e95edff9b83d9bdb4e66b5212e64466045ef9aa4c0e6ceb46e6ef |
| vector_only | artifact-triage-turn-2 | 100 | yes | 3/3 | no | no | pack-416f2acf | sha256-9ac8b8d75d56c0ed6d4f26217867da1079e6e2ec580a29f294977328791840b7 |
| graph_prior_only | artifact-triage-turn-1 | 100 | yes | 3/3 | no | no | pack-416f2acf | sha256-19092869424e95edff9b83d9bdb4e66b5212e64466045ef9aa4c0e6ceb46e6ef |
| graph_prior_only | artifact-triage-turn-2 | 100 | yes | 3/3 | no | no | pack-416f2acf | sha256-9ac8b8d75d56c0ed6d4f26217867da1079e6e2ec580a29f294977328791840b7 |
| learned_route | artifact-triage-turn-1 | 100 | yes | 3/3 | no | yes | pack-416f2acf | sha256-19092869424e95edff9b83d9bdb4e66b5212e64466045ef9aa4c0e6ceb46e6ef |
| learned_route | artifact-triage-turn-2 | 100 | yes | 3/3 | yes | no | pack-84a5668a | sha256-2fd6b56f7b138588dd2bbf8057741d4dc3832275cee1c0ec56061cdacea2cce6 |
