# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-42dd5ae1fbc52ab37ab26b7eff707ccc814072dbaeb4cf80246f57beb5474c7c`
- fixture hash: `sha256-dae1773b38ede59c62f735546227926063dcc22433a680794834acb15197b82c`
- score hash: `sha256-d2598e2a753de0d5e1dec0a5e8ae4e60ee76ab5dbb4605482d61e7beced0232f`
- bundle hash: `sha256-0353ada2db6ef7e84aa9a27c4aa84112ee0e8fc7aca685ac8fe8cc021f1abf5e`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97339b57eb3bff564bd492b91102981f9054b332cee78d9338b804ad8b646434 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ade7960865156625ff89b899268f72e559a1d91e6a613389843fbaa9ac1ed93a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2b70195c256097bddb2de08bfc6781eb86334bba75c77faaac6afb9943a75a65 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-9e499810308ad8b226aacefcc7eb70f057d75ee68bdb0963487b0c7c6af65714 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-641c3005 | sha256-1993902ffcd3d227e4ea4d45911cecf55cd064d61910b3a86e58847a1e501a70 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-641c3005 | sha256-533b989b9570d3a128e475c5c44736c2c3968d4301dc40f005b650695906ba3d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-9e57c86e | sha256-3f512903f7ad3b047d1b253071c47dba3a3ce64f5ba1eb21269f5a56cd906886 |
