# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-086`
- winner mode: `graph_prior_only`
- trace hash: `sha256-30e71eada7154db771ab0903504bad5b650bf53c07abaff3dbef886b8f9ed0b8`
- fixture hash: `sha256-64c4e7be68886fd98adef198061f1396410e569f1ac383e3a1d8328f35e849ab`
- score hash: `sha256-391db499a839ca8c25105d3c09c362c82b206dcb12e5152fbc7bb2077cf4a58e`
- bundle hash: `sha256-cf845472235e5b8df95fca77aede09075382623b99c5bc2a04b55f815515aed4`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-11abf4e14f8e2df5f3f8d1c731716ff1b91d254865900f32b5488290ffdb74a7 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-de1377cd734540cb741c40999e4a724f82641e395f9348d1bfd8bd98f662dbe5 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-cdd3bef5ec10c4464690d281ccabe2b5d032c5c1ceb6b14dbc2d08e580f8fec8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-c702184d2e7fdfced5f5d2494be55fef130b91af9630a3c788760c7b2e55ebab |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-77dd073c | sha256-ef6fb794126626206c2beacad73f6100a3d87a537ca19d2eb4a47dcc1b7a0328 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-77dd073c | sha256-5c411e7d66573750bf5fd9bb421816ea8a850b0cbd470435fa8c86a6e18dd151 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-77dd073c | sha256-ef6fb794126626206c2beacad73f6100a3d87a537ca19d2eb4a47dcc1b7a0328 |
