# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-954e7370fea342c102d12700848d801f9cb4f766c26a152991092af096940dab`
- fixture hash: `sha256-71ae1b688e319cb6bd41b60a17bd7289e838d68bd59d5831fb46d3db379a64f9`
- score hash: `sha256-1e5d87c4cb08578aa5968cf052187f27e56cb476cb11811abf972c46c4b3c1e7`
- bundle hash: `sha256-13488ceb3699296445e819c6b96bbd3daf549399dc008236e7314434250a9d4a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c508d0d953d2e20d8464c0689a6d8d9a5c0442d5d485367bc8ffc01689888e09 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-99f7425d2e05ea283910feb657dcab7fa932f60a439c93d2d5ef8caa01d7b91f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b225d0e780831a0af90a2a39d0c66753b4f2dbdeb1f44c2b3d8db401c9920a1 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3338b490e9dff093ff8067bc18cdde8c12493d3d79833f8c96cdbf1fe4e90a61 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a136eae2 | sha256-b5cd499736404c9581c20ad7360cb7d35ebbadb8a56f1eb61d86a007e3e91223 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a136eae2 | sha256-ce4090b5539c7ab2387b0a14cdb62435efde196d310d3337dddf2375d749b5f5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0113a46f | sha256-03bf029bffae49023bd6ecaa4968580309cd120e9b3bc424ee199a615fa2dba4 |
