# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-081`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bed00a4e07440963fcb335df85735aa4cdf299e8eeeee26d6072510f9d967592`
- fixture hash: `sha256-7b9b3de84eea4b8f6489862313ae3b9c5b0de1ba49de793c86ffbf0e24eac4d6`
- score hash: `sha256-6ce7b16a4e2a21b4bb1a9bc5528bc379422976d641a87c0be1d29da561fc944d`
- bundle hash: `sha256-9ce26ef2896184e7074a05f0503fdde81b8419e803b3d42efa82469ce1066f0c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7244838abdb700b7974c4fb03ae1270d3910510dcc5c175db289b9a82a5df872 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-84de6765a9e5b7882bf51c4c8c43807acd6d7957a631327584d09953d4cb70de |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-181527789a9c44e6fed1db4e8a43b9053c018dcc17f0cc2b08aa9bb414e7f5af |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-734a4fa455c06b0eb5a83f9d6ae61a0fc8c5db3ce4bd14cf708442c3e9d94497 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7ef37b2d | sha256-a243430c5b8189524aea66d6dd0c075f01478c481891453eb4654c643f4b2d32 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7ef37b2d | sha256-8d88a08d353cb78f7e770e9bab7368ce395569a504d88b696edafa0cc86d8b31 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-3d8a5aca | sha256-02409a242099339d6014cff1af2b5a55854ae26bcb9f85e2531f0d6ce759a19c |
