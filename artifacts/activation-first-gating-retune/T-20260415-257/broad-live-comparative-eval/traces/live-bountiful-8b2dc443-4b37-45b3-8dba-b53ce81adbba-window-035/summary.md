# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9ff7777be0b897266208f103a2bd1fea9aaefd91febf0ea117187545ef2d2014`
- fixture hash: `sha256-7993becb690144ea7d947bba5815a89834c7be01fb7391679807d26712c8efec`
- score hash: `sha256-479c9b6bd04f51667780b6379514d27d7b911a61079bb1fef6079bf4fb885787`
- bundle hash: `sha256-e56a7b20161daa929a8fef9ac3fa89faac3b3afa62884aa9419c13c5e74a71f2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d3fa2d46cbebd404032c589d360c14fd0cefe70dcd15d65b4e1f8159657f983c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-219272c9020fabfdffcdde5b389cfa65fdcc1dcfc7e5ad8921a7aee7b5b8d412 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-90ac1f37d0d0cbe3f9ad2bcb099f6fe399f4478b8b53ac8c666ab8f82c1a0279 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-0e3c9b71d0d8b74c8c8be8f16b7b3754851969eba672b1016272d017bb978c8d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3d138266 | sha256-04c7952b7ef7d5856108da0fb7090ccfcbcf90100ea336c1b4a666a0f75ce260 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3d138266 | sha256-ef186d7e5b53f59713b538912d79561b4acb2f29ddf48a3c50ffe5a5f9f88d22 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-88709451 | sha256-30228f87370502a52ab53decc3cfa694c3f0b5d6ab9014dbadbda076e91bfae8 |
