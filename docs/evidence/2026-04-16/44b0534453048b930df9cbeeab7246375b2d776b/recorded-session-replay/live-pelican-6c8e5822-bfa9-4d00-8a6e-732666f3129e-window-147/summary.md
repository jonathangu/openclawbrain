# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-147`
- winner mode: `graph_prior_only`
- trace hash: `sha256-12b53203712e88b756dee356041b3ddb0e18e328e1c8f8ade691064553599eca`
- fixture hash: `sha256-8ac6a4fe3950f0ed5cfb2e1b9bd9c7ad4d79faf9e22bb913250d8fa59920cf2e`
- score hash: `sha256-4c29d02026b8ea0d8deaca9db1b9c6724e2dc2a4ff11e5318a3fb14f9ffec732`
- bundle hash: `sha256-8b6d8547b9142ec54f7fae33c760ee6e3f4f74f2cf6c1e6da774053c0cb4c39f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-55766afad53c9e202670418bdf755c0f71228a26fa5f954c36b74006ec3fe092 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cb2f7974ca5a8ff586b5335644360c40a92344fa4393f63d2a60c619bd85f578 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6ec583d4d1495bd4cb331d8d1772ddbcda159dbafaef9bcb97f66eb96f4c764 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-3b43979ac07e7a8004bcb09102614e9c4b167d1176bc0214ac183d122b14717c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-24ac099a | sha256-ec2825aeee8c141def4b2d5dfa37f7844093d8274cae419009052492c2884fdc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-24ac099a | sha256-e1dce52fdcf3602fda362ebeecc9bdeb41a82f0326ee73fbd762359d3840dd67 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2858f1d7 | sha256-44e88bfb7093e4c86bf4fbeee2e11586979628113661de328cfbe6b9834e78b0 |
