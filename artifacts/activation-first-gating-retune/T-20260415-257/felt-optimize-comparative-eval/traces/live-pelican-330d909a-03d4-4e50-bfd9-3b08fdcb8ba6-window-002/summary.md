# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-330d909a-03d4-4e50-bfd9-3b08fdcb8ba6-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3188dea0835fbe3a5c0a4bde0dceb823e483b3aad858e66d490cbabd38ee5d72`
- fixture hash: `sha256-c551d2cb8b10201e8079d270837622ef96e1675624dead00decec0e3fb02a4b9`
- score hash: `sha256-af82d62e903075604c3bcebcf8e9f18e5318c3037ed4911a89d79bbd4d0e064e`
- bundle hash: `sha256-774afcd5b6e09d0420aa5268d7d2b8dd2790d312dcaba3519754dc7a78db978e`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9c7e263c626ee395c7a005cdac6d8c14b4d8e92d0d3065cdc0b98a11e431231d |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-54f3a19b33211c448ef5728a09f5ca649e492a58ee5d1a487c390ff1c9d16cad |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-d2d34ce7210e7fad7727b358fefa8caafec90dab584ac773a215a0586e5751d7 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-85b196127b62dc96d8752d39f96726a82176bda463feead25f48bf5c5286031e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-33adbef1 | sha256-05cb97739208c0a98a70a4825ee5fc5f805c7c15c2f279f45c5f097a92407af3 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-33adbef1 | sha256-7f5c45faab34ac06811ce2878efbe98dcbb43fabb106b42cf8fdb3ecd54822c7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-33adbef1 | sha256-a0049da33a780da4b861291d64edfded35713baa3eb319ceb394f62860e7b6e3 |
