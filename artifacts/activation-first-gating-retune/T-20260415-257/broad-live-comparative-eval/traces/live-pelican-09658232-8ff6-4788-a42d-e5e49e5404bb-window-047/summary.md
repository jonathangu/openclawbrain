# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-047`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1d8011faf3d69f32bcf0e92bcef735c94f96aebd8322b667cbe52a25917f1a6e`
- fixture hash: `sha256-f28ec0241ac4efd4c1f97733d381efba161e2d4c7cd778ddce2f415ed4529529`
- score hash: `sha256-fe69d8ae37df7c8c94d8dd98996ff01fbbcb8c96ebdd28bf153edec732245990`
- bundle hash: `sha256-c280fce66f01e416806d7fc5047c4ec14c52d017aaf79c58d4a5526a2ee5b16c`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-64f932981a8a1428e017d3b3bf8eed9c04a8f1b43e3be668df16d36de77d3b6f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e457429b950734c5c5c77a349ee20b9efa54749b06a7e633567a242cd52274a4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c34769132932ad1fed0041cbe58aa08cdd8026a35f4ea1cf436d734a735ff754 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-1f2c6fd0d2402c5bc0526edbb28cf070f5943df491ee8b9a30d21a47f78d6485 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-911cc502 | sha256-2705506a201b9d1d703602d7fd9ed8d7765487eec3416090f614a8dcf45b7e5f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-911cc502 | sha256-59d33ff03161cdca6cc5879b36e387d11320f1da239221f33f72f6a9db8dd867 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f952e30d | sha256-22a7805d299513b158b239c2f7b7269bc85a03fb6368384e2262563b0e0649fd |
