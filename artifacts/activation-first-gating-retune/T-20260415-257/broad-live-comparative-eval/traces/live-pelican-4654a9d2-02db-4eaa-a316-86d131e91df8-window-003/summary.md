# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-871a8d0a2f1d4e43acf8de9d8e6956ae4d1ca9dd0a419c5265c96970bba52722`
- fixture hash: `sha256-219199343b7c6d3ad1312b7304ed4e0c3741109cf5c94240ae657c56e05e2f48`
- score hash: `sha256-4b46fcebe136bd83086c0b584ab32ff649f550ea15819cbb2b3bbd8e56a198ce`
- bundle hash: `sha256-c7a145d599a1876151eb3e1e3444d527317f7890280208b3f0e88bfcbc26852c`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 1 | 1 |
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
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-c34e44257f6b5c4793085ff181cae894ba80938d1b527663f16e9c96f9fdeff9 |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-cc1d9d137c24a5c951555b8bc077a6e12c041a733b9b7be22ba859adece168f2 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-eecaedd9ed5e580bd45a06b68e43e18509acd602bb2d5ff05a4ed3f779fa9fda |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-53e75187 | sha256-745220f7d8ce7258725d82853d1d7a2abbc29668bc30f12fdadaee5515b348c7 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-53e75187 | sha256-ef511efb4b49b433c93acfb57e5b3a92fb77b66f37a512c0371583d93c286f7f |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-53e75187 | sha256-745220f7d8ce7258725d82853d1d7a2abbc29668bc30f12fdadaee5515b348c7 |
