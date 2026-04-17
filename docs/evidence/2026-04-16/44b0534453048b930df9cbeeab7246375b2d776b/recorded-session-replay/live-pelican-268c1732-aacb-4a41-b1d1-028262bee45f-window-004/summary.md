# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-380e12e9dd757771937f4748557c11b50a1f9a231591dd724ca65839af3ce6a8`
- fixture hash: `sha256-86ffcadee00971f5c46315d2afa19ae2e85e45bae4dad0e458c42f57f711f9d0`
- score hash: `sha256-7213c6e53327a829ff132936486e8ed58da70e6759990e367b45b4d49d3f5704`
- bundle hash: `sha256-9201e4dc715b3bf55e634af8a4607a511dbcbd1b68b5733b2b4f255e8ec13d92`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0928afcbb7b85c41b2a1d624e920cbdedd75575cc8baa6c3ef5218e9d291b99a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f71e1083fe523f913965a2d98b878c5b7d715d7da6d504369a279a97231d2e90 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-139a942cf460b1a2b525f6c6d43ba06a9bc7f245502347fec57eb65ac4f71f23 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e3ab539c77b0dee9cfe3252cf9a4f5d7ab10f8fc059c5e2eaafde476b8c2a607 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e526baa | sha256-5bc3d25e872966f57f26fcff782269cae13bc2641d648527d3f29643f08c6d1c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8e526baa | sha256-fb49ccb7ea0cac0c864caf83259aec88e07f7ebe68c5a79263dcf36304670ba9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c7c73145 | sha256-56eb42000a64ffbe8effc325e85ae8092364da71ea8ac34ce67b2783da8ac071 |
