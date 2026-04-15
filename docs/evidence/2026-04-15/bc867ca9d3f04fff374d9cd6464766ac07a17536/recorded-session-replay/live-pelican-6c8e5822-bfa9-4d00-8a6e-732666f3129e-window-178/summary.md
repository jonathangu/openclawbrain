# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-178`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc837ac64ce4a5cb1d121e2bea7830254f5b1cd1faf9dd8be0505cf94fe18342`
- fixture hash: `sha256-555eb18092c7a3b48bf36359187522f84e12b063bd73ce65d859cb8f468c2af9`
- score hash: `sha256-a1a3fb00afbe8e0b0fd6581e0f8a0a41fe0c6f7637cd3c1c6d091dc3036236eb`
- bundle hash: `sha256-2ab418515628c69e60c83e48617a51bcd882681b0fc592255ea6220507263fc7`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-94018db213a88670c23984311d9a8431beabced6aba3b25434ee10a70b79887e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-92f8624873bfa86f4331d26ab26694716b4670a9be8913622749559611448ca8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-20e8aef2c5b023b6ea9217d7f3a5027348d5d373d136a763e6583b7c03d8f30c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-81255fd4c80bc09467d4deb5c7948ad761f0f3e17f1e161ed95a61222e0f07d9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d844634 | sha256-85a3f5a5a8a48e06e96bbba7b4052fc87b0d6b233f4c39b8e56366cf1dd78e01 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d844634 | sha256-0b60566c0538ede7cb4769b022e7d15701b1392c244b45904c5e6a48fce6b0af |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0d844634 | sha256-85a3f5a5a8a48e06e96bbba7b4052fc87b0d6b233f4c39b8e56366cf1dd78e01 |
