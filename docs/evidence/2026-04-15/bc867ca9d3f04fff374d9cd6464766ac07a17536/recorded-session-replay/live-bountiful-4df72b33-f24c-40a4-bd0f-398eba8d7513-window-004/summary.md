# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-4df72b33-f24c-40a4-bd0f-398eba8d7513-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4b9d58238866be4c30cb67001ed41c476bb074abc457ce91f27bdf2a95087dda`
- fixture hash: `sha256-93a191f41c9134f7fb1b39f4120c598d79722f0fdf720a1c60726eeea45f85a7`
- score hash: `sha256-6554f455edb479bf0c7fd8457799e00c81ad724b0ac0e76e88b92529f25e846c`
- bundle hash: `sha256-9de6b2b247eb9e80cc3b7f2aa4639f30d3ed7511e7d3eee0be0a660d8d631c6e`

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
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-53ac170bbdfe31610a82a7fea6a20f739ad327e9856e23aa713b46f86601ea52 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4f75f859f0434819fd17e53e1f7e38d790af31fdd478221c06121be27d16075f |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-34e5686a9af98105a610cfa9050489b5cfd623ff728b75b9b381eedb9a979d5a |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-f28be31f230f5ba08b6034deb4bbfe13a8cdfc5fbfbc51fc7aca3497d8554e26 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-8804911b | sha256-09d0423547d979971268242cdb22818b8b9b725d216a6cad87f50eb6407a86fe |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-8804911b | sha256-0919474b993117410b945d24ead256446a31a843802dd4759380f7e40a308df5 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-8804911b | sha256-09d0423547d979971268242cdb22818b8b9b725d216a6cad87f50eb6407a86fe |
