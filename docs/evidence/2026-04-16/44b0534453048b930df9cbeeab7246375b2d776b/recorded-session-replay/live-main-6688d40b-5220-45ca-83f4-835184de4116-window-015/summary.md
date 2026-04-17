# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-04fe85c8a179be229d8c68dec97a25113ffce0ef409792233e0ccc1c65106721`
- fixture hash: `sha256-66a77cd573b5398a7b3b4867686fe20ef718501f851c3ff410c457c68968fa97`
- score hash: `sha256-12f3377744958b840023cacabcab5f93333a64ef3a819c8967a48b511f6365b1`
- bundle hash: `sha256-e837ea9b2d249c0771306945d2310d5207fefcb25ab5e4c371ffedb750202f07`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ae930e12c21c7056f67f547427d9cdedef7d7970b442aa81b3fdb75182425c80 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cda90b880b043b30fbf5c3cce477e2c6b8415471ced9979a3d87d01c1f36f915 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b7ce4dc028b2f971ceb720a6fcb6f19c34ca4584a2b41f3c4269ce804d74e17d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-3bfb12a953cb0f4f458ec57216944e65de2e8f44cd89a767196aa37e6d40efb5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6d4ec513 | sha256-7f303624e767ee8a3036f82d9d7cce1697eb99c78d253773934595d62b965a25 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-6d4ec513 | sha256-12ba591095b7200f5a2a14e08d125bffdbbddf72ca91639dbfc8ef2eb6558ece |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-cacbf4a2 | sha256-603ed53a682871a9f361a7839c2ec500a40f37e11966c149070544ba7a8e715d |
