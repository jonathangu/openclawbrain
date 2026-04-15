# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-053`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ebe19ca9bc459ccc52f05cb3ef8e24277b70f060cd7939f534aee99c63488ae5`
- fixture hash: `sha256-83e33e2dc3d5736fb8b475959b3f799a1522431a9ceb8bc4c7fc74edb18967c0`
- score hash: `sha256-2e39fc7d8726eb67601fe7b45de10ecf5543a417b8e479d85f99f1816a1a1bd0`
- bundle hash: `sha256-1822b619e545e9130e008d06e01370b88f96392c36369c6b35df288909740b1f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38dbfef9dbd8d3a664a1f95db7f92b5a52579781d0eb6bee8aa47758b54b5ce0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bdfc382df7ab20a303d209fa6d8b1c22bbb8a4d06d2683d8c1810853e596a82c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-5f6e913cb90658b55e7f0cee9d215b46047d6fe87a7eb04cd1849550ddb3cd7b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-9b56d8d3cff2aea90278972fa34626ba6148820e96e0c15072cae05adb81c85c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-46583fbd | sha256-f8bf7072d9d735a70f05039a8250e65b076995efa7e51635ebdf0e0a4e87779b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-46583fbd | sha256-885590a0b669803913557a08fbf412d842592be59d5e68df07db79234df49c62 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-46583fbd | sha256-f8bf7072d9d735a70f05039a8250e65b076995efa7e51635ebdf0e0a4e87779b |
