# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c1be6f8d-22af-4aac-9f32-234846a9ef57-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b9882f49bf30cb6d948087b310dd1f1c8c43cb51ebde7842866360d6db046b12`
- fixture hash: `sha256-371ddc3cfed0332b92f92e9c2b214fd34bd05f438837cc6562acfdd4c1e2c749`
- score hash: `sha256-30594ccb880d049c8af682981374fa276cf67ad726b6d83f9f54cbd2132ccbe6`
- bundle hash: `sha256-0a9209768475127af6ad1f52f248fcf9344a9240bccd424391d5cc67865ec117`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d3958fc805f3776c38e6e687c85563bf09e68cc8dca03392a973d72cef995c7a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-93e3d7de48edb5c306df91263db4bc6296f541ee95aada11c02034644473dde5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0fc8359ee7f813872835f15ec8ea4af0ea48f43e2d4f8c66f0ade3c03b42de99 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c22533e011a155e6db5b39e8c821e68186e67c7e5fe9475f2d17af0343b07ce9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-776a3cbb | sha256-92b71ae5abf8c94b386fa70e4b3695fb1354c9ead549762b661df66165cb93ba |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-776a3cbb | sha256-5ef04c25473bcd4fb3fa2432f4ecd30563e074804f2ee9663a4e3ab706a8e78f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-776a3cbb | sha256-92b71ae5abf8c94b386fa70e4b3695fb1354c9ead549762b661df66165cb93ba |
