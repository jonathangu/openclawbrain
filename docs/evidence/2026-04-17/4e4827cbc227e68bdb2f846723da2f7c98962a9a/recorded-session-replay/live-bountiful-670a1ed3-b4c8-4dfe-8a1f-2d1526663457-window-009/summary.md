# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c58ed04d44aeb04071688c4a26c4c689e25ea007697f349c3e4c8fcbe3bda533`
- fixture hash: `sha256-ad70501e856aff4a57d924d7225c4dc64463e70da2f3e42777305ef85fb46a26`
- score hash: `sha256-a77febc2a23f0983f5a0a1d50f43ef9a0da4943ad4bd3e4c21d23a7db7d1833f`
- bundle hash: `sha256-24d91b8be748409816959c83bb4ccf4dce80f4a5d3e42ec33bde68adc9ce8563`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d770aca06ab90e2e0a0ead714079ce642ffbbb18580e6acfdf4fde922a74f5a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b3dd1d72f78b236e8427c103ab69186ed095b9ef27a39ae3cb8d5fb9e48d1998 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f2e807f11c486fbbf4e3eaa6a6968827c55a40713622cb2adeaa26f431f923e3 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-d96a4376e4895e4d70f8fba49604f62da0680663142706ecde3874174240657d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa5bf047 | sha256-8a64e8d814808eb371e57b65cc7449eaf9290758e86c8e0a4b17997369f79716 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa5bf047 | sha256-83854db433bed891cd97bbfa94fe0358d23415f5c530b67f0cbbef7c071e644a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3b65f424 | sha256-ccf158815081a9a0a75cd2808eb3945a71bdb64f01108b1b1e16090848feda10 |
