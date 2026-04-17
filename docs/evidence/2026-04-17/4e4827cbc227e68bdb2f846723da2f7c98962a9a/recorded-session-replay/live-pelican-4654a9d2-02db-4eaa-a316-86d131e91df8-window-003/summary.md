# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003`
- winner mode: `learned_route`
- trace hash: `sha256-871a8d0a2f1d4e43acf8de9d8e6956ae4d1ca9dd0a419c5265c96970bba52722`
- fixture hash: `sha256-219199343b7c6d3ad1312b7304ed4e0c3741109cf5c94240ae657c56e05e2f48`
- score hash: `sha256-0910a3ca8d3c4d1f2f62823a47fd8774176203a586bae948346850c4129f8af2`
- bundle hash: `sha256-09eba95216e34be0c0f56ad71e03f271233add03e3990d9fe4d258e2377ba405`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 70 |
| 2 | vector_only | 70 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b917e276e14b121cc269645e14a5fdafe3dcdf3d48a758ee09ff6c7e3bf5cdd4 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-2b451a3cf0a4c1400d3cf3b897d43a1841b53b519aad09ad4758eb110ae035e3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8744b87784eaed99b1b1f93d02444dbeb4869b3f38fef078f6f8b32b5802d2eb |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-55230d35c1fd98a2f17ca0bf1c9e34db4eba88241e5d3e9cc821c3621e4c91b0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-4355d36d | sha256-60e17fa8d6c95e25e3c7c611b8d0c827d78e6ecac2c635bffeafab82c1fbf3e1 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-4355d36d | sha256-6134f3ad0373c63473a7e94cfae5d2b3949261a2d1887ba08346a1238f0e1608 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-3822cce2 | sha256-48cbd821a8480d99b1e1828d805240519e3bad85b05745e4acd9bbb497264d6f |
