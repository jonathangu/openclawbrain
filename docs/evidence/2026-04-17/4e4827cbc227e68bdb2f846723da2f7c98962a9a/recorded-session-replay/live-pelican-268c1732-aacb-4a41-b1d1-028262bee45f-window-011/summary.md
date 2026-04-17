# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-011`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c761e47d47331bf575b7d002c131402195e8ac49d688ce355a015b14825d3acc`
- fixture hash: `sha256-128e53eb7404fc5c5e08cc33f7657166db8766e76b0fe254b4c32e80c9220dde`
- score hash: `sha256-1ce064ffb7afe7bd104e35c1b1d7320cc52a9d154afc0ab0e8e1f98381234fb8`
- bundle hash: `sha256-b1b0a3766556994d31413ed61008027e907810c5d6d61032b600a13010cb64de`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b1631644a30db1cf3d02f7c72d9e973f8085c5ed6318e74e1e83701e3e901455 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-aec16a2db93188b3af89716b862e1cbef125c1650a205206c001b4753559a92d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3d3a9b7839b173ec0c578a08946693ee224e5fc3771ada53b9db47e63734003b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-153a5f3326978113b827c2ccfb168834cf427b12510fca76b2d853bd8a016d9e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-da44a1ec | sha256-4b9d54dae6879dd96b9a306ba2d9da2dafdb987eac6400500b01f9935ce80b23 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-da44a1ec | sha256-9749985d07a788193821ae6b3e7eaa275beb9c331460957e88c47f473713653a |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-790bdadf | sha256-dec79a0313d5b59c059d3983ba86206894560a263f3d2836a89331c40521c734 |
