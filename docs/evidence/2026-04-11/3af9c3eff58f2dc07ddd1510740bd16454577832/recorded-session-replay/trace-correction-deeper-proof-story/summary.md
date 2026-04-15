# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-deeper-proof-story`
- winner mode: `learned_route`
- trace hash: `sha256-34e695185a4af79eba4d4526f41b23f1e694980113c7bb70bca61b4149f2d707`
- fixture hash: `sha256-51944764426fd1dd7985c8105e13534bf617e449a4d2c80a2255c1e2a25cfd0b`
- score hash: `sha256-485e6b66de6f63ddd40d0a216eb227d7deb80a7733be90100e326d586ad3090e`
- bundle hash: `sha256-911b4ed41f123d4f8179f270de4c80b93294e8dcae8aeff50f44cb1c622dfb87`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | graph_prior_only | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 9/12
- compile ok rate: 0.75
- phrase hits: 8/16
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 3 | 0 | 0 | 0 | 1 |
| vector_only | 3 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 3 | 1 | 0.5 | 0 | 1 |
| learned_route | 3 | 1 | 1 | 0.666667 | 1 |

## Hardening Snapshot
- compile failures: 3/12
- compile failure rate: 0.25
- warnings: 0
- promotions: 2

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 3 | 0 | 3 | 3 |
| vector_only | 0 | 0 | 0 | 3 | 3 |
| graph_prior_only | 0 | 0 | 0 | 3 | 3 |
| learned_route | 0 | 0 | 2 | 3 | 3 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 3 | 0 | 0/4 | 0 | 0 | 3 | 2 | 0 | sha256-f751ee87184d52e0299f5567201a7f14964ce5b8db49fc1bdf449f68b5a6c219 |
| vector_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-8c73b4c9e50663aee7b2f8e5af9e9c2097b0e07e94d144be927d3e3efe6865e8 |
| graph_prior_only | 3 | 3 | 2/4 | 0 | 0 | 3 | 2 | 0 | sha256-93c64192c4665e937d54294d137f425e12ab9815a56b9f827b8c5fa807069d8f |
| learned_route | 3 | 3 | 4/4 | 2 | 2 | 3 | 2 | 0 | sha256-20c6e7979e73965b99903b115dcab0e9ac4910038ecbdf7007687139ef168d62 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | deeper-story-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | deeper-story-turn-2 | 0 | no | 0/1 | no | no | none | none |
| no_brain | deeper-story-turn-3 | 0 | no | 0/2 | no | no | none | none |
| vector_only | deeper-story-turn-1 | 100 | yes | 1/1 | no | no | pack-8f60d0f9 | sha256-97b58cb30ef18250a3980036a651e6f3cceba7dae6b3b1c51026cf4eb2c45a38 |
| vector_only | deeper-story-turn-2 | 100 | yes | 1/1 | no | no | pack-8f60d0f9 | sha256-9bde2154771e1f09ce7314f16090678910471dec4713f49f11005139350d9d45 |
| vector_only | deeper-story-turn-3 | 40 | yes | 0/2 | no | no | pack-8f60d0f9 | sha256-532c3a9438dcf14d715f03a5c540e2964092a009c6ac3c24fc7b40a34bf8cfc5 |
| graph_prior_only | deeper-story-turn-1 | 100 | yes | 1/1 | no | no | pack-8f60d0f9 | sha256-532c3a9438dcf14d715f03a5c540e2964092a009c6ac3c24fc7b40a34bf8cfc5 |
| graph_prior_only | deeper-story-turn-2 | 100 | yes | 1/1 | no | no | pack-8f60d0f9 | sha256-967523806bf5a021392a511d4a9663f7f784d0c63a864b7ddeb0d53000135f83 |
| graph_prior_only | deeper-story-turn-3 | 40 | yes | 0/2 | no | no | pack-8f60d0f9 | sha256-532c3a9438dcf14d715f03a5c540e2964092a009c6ac3c24fc7b40a34bf8cfc5 |
| learned_route | deeper-story-turn-1 | 100 | yes | 1/1 | no | yes | pack-8f60d0f9 | sha256-97b58cb30ef18250a3980036a651e6f3cceba7dae6b3b1c51026cf4eb2c45a38 |
| learned_route | deeper-story-turn-2 | 100 | yes | 1/1 | yes | yes | pack-9133d6c1 | sha256-cfa695d632306c879ea4cc48d5653e2ef0079f7904dcf9bf57bff02bd2673c67 |
| learned_route | deeper-story-turn-3 | 100 | yes | 2/2 | yes | no | pack-8c3cc526 | sha256-4b4fcfb5bafe73a5668aeb685a073e72b7d7bbe477866ad1ddd403b805efb4e2 |
