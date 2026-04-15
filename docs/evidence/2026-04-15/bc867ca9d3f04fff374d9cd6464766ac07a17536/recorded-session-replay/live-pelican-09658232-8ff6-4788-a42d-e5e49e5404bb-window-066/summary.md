# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-066`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6ac1c0fab25c05525176cdfff2149e8d15cf9da0d9dd3e3ff8d1e6b40aadd074`
- fixture hash: `sha256-99a9dda4d1e27d20e5b5802fe99ae2cd9ee98cd875422b1ef45282c42f60a797`
- score hash: `sha256-f99990c1862663ab9623128ec6f7a31f0146efc98b83c470329e76201475312d`
- bundle hash: `sha256-ee3e65733fcd1792c349678c3c2ae20052fe158f555379e5558049c4f71359ab`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-c3402e859f2552a40a7f253ef60215bf90d6f117858139b3ed26992a03a4545a |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-40df57e8e90e75c880ad1a921ca34e7072d8610d00fa97a1f8f4a77e917df10a |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5a0d4a7e22127c99a54c87bad6d788862e3deda89092c46668fc01718ab31318 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-baffc6450dc878cb959d4b0f06399272d272688b1c384db00ac49ff5506244c4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3985bb47 | sha256-1e1697075098b133f3b55596f7423fff78b7c65e6e9063c4da83fc4c993ace00 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3985bb47 | sha256-e8220aac3e6a0c4ebdf64e7e2dc9ee356f73e791a3d1d2fea9058effe0b9caff |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-3985bb47 | sha256-1e1697075098b133f3b55596f7423fff78b7c65e6e9063c4da83fc4c993ace00 |
