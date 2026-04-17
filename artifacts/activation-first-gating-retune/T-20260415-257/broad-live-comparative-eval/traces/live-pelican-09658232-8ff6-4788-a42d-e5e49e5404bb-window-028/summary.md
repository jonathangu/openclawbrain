# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b96b2a97bdbabf2a2491696460b8da77dc242a25c47533759e1ca69d544c781`
- fixture hash: `sha256-32449d86eb6b142eb11e1d76d43e4c37d62e87233bae5b870977e6a064fa97e1`
- score hash: `sha256-95b400e80ced52f355ecc26ef279c8680acad624f5c01d872bb41770e4942c21`
- bundle hash: `sha256-ec072d091b8fcc744a4183f066a8abd4a244d97a75b1e2945780da18917560b9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b14dd3575bcaf16e76897e36504d083be01ba320a2077714c9a7749ba84f112 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-982007fd9e88e515fd0de73d6a7bb6914b55ea09da66744c86f1ef8b1b857e82 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7e447eaa2986a0d8cc7151268f669a7db938882b7ca6582c7e6b96d0e3a0a462 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-016f6f39e885dcc660ae5d2e893e022fad1e49940fb6370ea16febfcd0b8f709 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e0d0c437 | sha256-70aaf5dd0c8b0f2b0c0f7ab58d472a2d977eb6219e7a09d33841220c56ea110c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e0d0c437 | sha256-3067e316495821616dfc8a5beee56b740ffd465589bbb906c5d89a1bd0f081bf |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-81bb159c | sha256-16993785bafa6826d47e0c16fedefbcc37b02f2d47e9c68e458227c5afd5ecf9 |
