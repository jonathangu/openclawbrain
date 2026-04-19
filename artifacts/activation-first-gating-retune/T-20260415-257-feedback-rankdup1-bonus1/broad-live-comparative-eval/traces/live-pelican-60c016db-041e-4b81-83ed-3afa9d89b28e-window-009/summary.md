# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6c49ca2fe441629d93695e938073dd41facc650cc9fd301e1fa807efab482f72`
- fixture hash: `sha256-a51ac08634e3c4803c3d3973ce1a7c858ffb1429844452de0ed6e3279b36730b`
- score hash: `sha256-35bf2a25aa5cf832604433f3f49d720a5cfaabd6157986aa61ad553438cfce23`
- bundle hash: `sha256-e2b21418b377e2b03672c0b4ee3ebe520129225febfdc12e761de7e4b841a6f5`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c2e3cc9fd6ee1604d2c526310b1ddca47a11a50e7f9573ba696d4001f01dac |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-261749fc095619802c92816bbf8548ce89fe37429d1920d0b1906bc16ce86ac2 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-166f5fb909c080a6b4d223cb583ca191ac3fc6416fe9a3b4dc6562ae6094390f |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-43895ad94a134b9df1cf74c1e38f08ddd63f58e7c9e0c9c786f3dd6d3e437505 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-3e9f1f89 | sha256-38334b1316cd6e285cbba6f547af7fffec722151b67b6a87f126b4af21121ce6 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-3e9f1f89 | sha256-647d66d563718a2dd41500563cea5e6786e88fd706623fbb6e7cbacd2eb279a3 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-3e9f1f89 | sha256-1eb19b5b8df1ed9c6938a78c24f68755b332a1659421f4580328eac4ce97a68c |
