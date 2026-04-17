# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003`
- winner mode: `learned_route`
- trace hash: `sha256-871a8d0a2f1d4e43acf8de9d8e6956ae4d1ca9dd0a419c5265c96970bba52722`
- fixture hash: `sha256-219199343b7c6d3ad1312b7304ed4e0c3741109cf5c94240ae657c56e05e2f48`
- score hash: `sha256-158a24a0fe30409e51880fc2818a8ef50ada76d57f8d9756968b71c86f832b79`
- bundle hash: `sha256-1c8b0a3f19a8d0559cc23b88d15940342beaed963b589820780e852bb753f2c9`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 70 |
| 2 | vector_only | 70 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b917e276e14b121cc269645e14a5fdafe3dcdf3d48a758ee09ff6c7e3bf5cdd4 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-509fd712b7625814d27342eccb744f0927b3e63f0d372751d94e47f101ca2730 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-38749b45b11b8d6cd60c674657c648f41c6fe04cdd4f801dcacd5d0320f2fe4a |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-377715f933f56d06022bf1234e000f8c3f21d4c6cb63207e2561138796d4a379 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-c3b8518b | sha256-6ad8675aea56bdd3696268c651a4757d760922301cd9f5e016befe4bbaed4a0f |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-c3b8518b | sha256-851f4c0d058068515ac5f301ba58ab9763dfb3dcb6d3f9c4ae4a6c1cee1000e0 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-b8854b00 | sha256-66d89e7135d30b4dd0f51785e3efcf9f04d89731edbd595a53ddf6c5010177a8 |
