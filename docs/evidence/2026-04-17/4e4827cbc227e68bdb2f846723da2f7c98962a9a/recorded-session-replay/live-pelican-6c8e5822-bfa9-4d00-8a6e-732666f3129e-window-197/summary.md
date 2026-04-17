# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-197`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dd29f792fbfbc606fc0ae81485babcea7a498bf8f85b66de2333918434925117`
- fixture hash: `sha256-a84a33537b8d24e458443c5c6b1cbd9d02b490a8b56c8f49f8509184e51ddc87`
- score hash: `sha256-52710353cebbd33cc8d24921da8be01e2e188bbc587295af456ba923c3447324`
- bundle hash: `sha256-27063084fadabf5840689b5c3a2c1ec6ae1a5eba8769cab851f69a2f83a0d5ae`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d6181c2140140f5786a710721c2d0cc92976577da480a328a542e8b790bc4990 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-de4a2d001625cfce575db8fc4ffe37a5cd1cf2d661a4088c4c75ebead7688f1e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e7908e39e93b8c5b98f9b3949943456cf582b91ef742b9bd702affc818e0fc85 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-93420ed6d7342721e03487c756bc4073cb9ece150254e0f1be762f8a1927282a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b27f4b45 | sha256-b60b8b899057f571c66ea058ba3c5f93043982e80013ef4643ec4473dcd7af71 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b27f4b45 | sha256-d2b46474261961362515077a51aba8e3614591c87b8f0d72841ab29f1bcdbd02 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-af7ea272 | sha256-31809651210b4cce3f68cd04c682ef2e34fb0b282983c462b3883222d7c541e5 |
