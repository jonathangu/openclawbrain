# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-072a9d79-0a6d-4d33-aa9c-b4474dc2a3b3-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1df02420998a45ff18b6fa7592e1d6cd553e69e00670f629819f48f156232f3b`
- fixture hash: `sha256-d26e41f8ffbf777f72220318ad80ec7f532c81cc4e8c86beb0f89befd769d272`
- score hash: `sha256-bf1ab524101d65ce6531a6e1dc34dae7d69e6273d3073269efc4f62f8c34239d`
- bundle hash: `sha256-121ad3ef91466e47d80d13dff7fbdae921c6371c09e320d17e46a98ce62b4f06`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 1 | 1 |
| graph_prior_only | 1 | 1 | 1 | 1 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-caffae64068969fa7e1d950417642498125ccd7a52b99fe5538a0a0e555ac8a8 |
| vector_only | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 1 | sha256-0ce8d3c8950f97c3e8529cf9e4e3406c736d14e13b20360994c160b35414d69e |
| graph_prior_only | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 1 | sha256-c1f7d256a17141fc3a26d2bc41e8ff577214dfcc5244dcfdac9d0bc34730babe |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-042f1281ef504d69a39e85f278dad587b851d0b5151efd15240e35aa888ce4f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | yes | no | pack-e4880b43 | sha256-501a5ef2feac0665870f91996d92594f1c67f69ee2fb427202412953d5f7c8b7 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | yes | no | pack-e4880b43 | sha256-11f7e920a56d1101b878e959a632f693e395dd893ec24b394514758037a451b1 |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-e4880b43 | sha256-501a5ef2feac0665870f91996d92594f1c67f69ee2fb427202412953d5f7c8b7 |
