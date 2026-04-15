# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8b146779-6fd1-4e35-b861-2d0ad85401e4-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7c9bbe1bc32703bdc0ba57cd7c2e5ba0147d232db874dca18f8d1c93a644936d`
- fixture hash: `sha256-1632c273e7fcb25c5de9fdb5adf5c07fcc4c43677737f0e63cd97217f3d6d9e5`
- score hash: `sha256-90f5cfdea39eebac0ff8d34e39db1d2428b1795b9febe363dd0c888a47eb694b`
- bundle hash: `sha256-e429e5239297a772573fa082c926e3237a6730a5e957ac92c284c6f11ede460e`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-80e9af933d3a18c2836442131236b812d1fdf8db3bb96c2fc77c951fce5a2ed4 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b5913230037134156d299ef6b8583104c3e0903882db228076736949692f9726 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b89c1432290d10e38a55dffdc67fc8df4c6eb3e59fd7ad0008acf2fc547c77e3 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-94c817466948f305d4ad0e60659154ea925a220dee5dea5c0453a937c983787a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-83d04819 | sha256-d8971db23ab6d542a0f3113088aa4b05c8865dc9aac69afa20dc57b2d714bed2 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-83d04819 | sha256-47554152e3c719f762bea785231a06de5b361214656ff3a4e13be097abb86c2b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-83d04819 | sha256-d8971db23ab6d542a0f3113088aa4b05c8865dc9aac69afa20dc57b2d714bed2 |
