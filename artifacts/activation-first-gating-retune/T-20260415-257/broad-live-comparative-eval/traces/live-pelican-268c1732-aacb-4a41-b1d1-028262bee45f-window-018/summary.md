# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-97d7d39c8ed80340fd41820d6d636bdacbec2fc0c19c6596d376217775b20481`
- fixture hash: `sha256-cee22d0c8692c9c54ea684f49e1d3ac5076518c4157aff7a2d52bb3e3278c63c`
- score hash: `sha256-9ceb3a4b44ffd28c7fa5fbcb2e68690cf6049506470ca0be113901202aeb385e`
- bundle hash: `sha256-097cade7ab99073cc1af02fb8cc3e42baad88a36556900f906cfe3e48107385f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-37de16724e3f909b52770a9de834272378dcc6d8dc93db3d2e32057318f060c6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-656dc6d65ffd348cab8432e3b3b533db884ad2679fc8723cc531e56b8c7053f4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a8ba94abf984bff3d206f4791691bedf4e758c74504edffddeca3a1d9bb5d620 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-3977cf4e2befc6985b44d4893896358b04eca8ee0cbfc828d039a270b00934b1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-310600f3 | sha256-dcee99083ca668374b8887e348accf6f2a2d5b1d12e679d02933e65c8b145148 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-310600f3 | sha256-ee6bb71e3c47c2bfe51c51c9da36f2bdfc8ecd08e080c03ef41e6eb022943f06 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-454d4c38 | sha256-92e7439abc207674b09ca944b17913a1db2a257a0082f8fe2852b6d92c01ec49 |
