# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-029`
- winner mode: `graph_prior_only`
- trace hash: `sha256-766d9b6ce430d9d07fe2ff3297e9849fe05332d7539d3d62db1cee2a9f89081d`
- fixture hash: `sha256-21e8a90c2dad8ab78ca636bf0f382e5b550e2af76a7681917f1773769c731648`
- score hash: `sha256-7b4f057e6b85f57ef2acb76f4c56e4aa722e720fff8b23a35cdb3d92df1e87b1`
- bundle hash: `sha256-2d54619b616767050e107cd9294741fde3c25b52d8e067f4f4d0182bf217ac13`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d8021a8424a98c9c0ae913d23bd911fe66b4179fa226e5ae4873cee34e53cd89 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3f7a249684e6de94043e84ee62d193ee73cbac9bdeeacee927036b8cbf23c5ba |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f98dea3211d612f0853fa22d5b5420bbd144141d828831ae0e44bab3dc9ab29 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8287bc08a54725ed6f83504f74d03ee51ff61437aa1d2c1790dee75e2b905e3b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4316535d | sha256-a1d87fabf585e104a324a5b85e4d5978fd970b39d30ce29ac75014ad1564bf56 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-4316535d | sha256-94c4f3353742aa6f05a8c2c9a6d476604a12970c11c266aa227119b3dc4e05c9 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-994ea7d0 | sha256-946ab9a3770fb112f2de078ed9c4f5c9de26ee8385414f8013cbb2eb9e501772 |
