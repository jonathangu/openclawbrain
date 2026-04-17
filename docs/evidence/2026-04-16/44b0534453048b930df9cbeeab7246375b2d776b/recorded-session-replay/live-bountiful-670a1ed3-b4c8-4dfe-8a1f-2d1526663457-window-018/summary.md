# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8731aca670fb1adc2a11de661b208e90de02229e43a59b819be0c26634995543`
- fixture hash: `sha256-b091c6d75f126cd4fa41e0e62e2c1bde2a5cadf897b977dd808714e16a9eb7f9`
- score hash: `sha256-34b43e398d99c5af2b7453c404008eb38c6edfe3c8c04d4407133202dc41a2eb`
- bundle hash: `sha256-c12cd146c24f5dd6a3e6efe8c49e9729b99324548b67cbba16985b645ed5bfe1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dd25e120884595a4500dd8027a1e5e49f93c256e2e2739aa127521c9309576c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb016b459748ba9fae4294f8700fe30620f6b7cb973352a8fefa7d0522f80eda |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4fd843a3937aea199d3b3ae64eaea3dd30be72687ba9f67e37e596c077374475 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-aa47d6f54282ade4b61902698e124186d42977f5fe45c56c070030b1637875d7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ad10cfc1 | sha256-f190210fe63e229b5d740b55dea90c207e4a05116e0a3ff66e69ef31c2f5d254 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ad10cfc1 | sha256-a89c8657d2d58bea8530f7db92f006e6823c6e07c4ec90d74b6118f07d494b2b |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-f536e3d4 | sha256-bea7a433eec000088f4da46ab0148c472b1254ce43162bb4593019071c00c42d |
