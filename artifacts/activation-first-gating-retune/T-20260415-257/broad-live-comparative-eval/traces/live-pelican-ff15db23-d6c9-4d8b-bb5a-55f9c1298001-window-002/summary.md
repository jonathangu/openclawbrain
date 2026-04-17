# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ff15db23-d6c9-4d8b-bb5a-55f9c1298001-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d9abdc2f8435606514daaaad4927f60e901a9d2b092eb5d39df77887ebe5a304`
- fixture hash: `sha256-6e857ab9cb3ba1ec3e0f72cceabb24485f23daf6db41d61af726b2888aeb0f66`
- score hash: `sha256-3fb1e38b8fa54d58741d782b258060be9a185e92b5b03be04ed6e88e4d5122bb`
- bundle hash: `sha256-7fd80eaa87d83564ad628b1d85a39b26ba00598b18c253fbf97817e093c903f3`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-865378fe979515e6fb05b86bb93e571f4e3d4c4ed17ab843485b9830a42b2636 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1ce1901092fe939858cc36385fc97cb6a6aae541f5bd122b0756e3ba5ede4640 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b1e6afeee408290dc05c969366dd735a393621067552a2eb499c5e5ae37d3ac5 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-b4eb8e88df5f5086e23e2cbb39ba5e9721e192cef7e9b7bfe0676b386688d72a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b1ed61af | sha256-52679baff9d69676805e890287cf4ece9aefa07c32958187f226dc29fa8564eb |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b1ed61af | sha256-6c04d2af941f9f6a5ffffd0469f5e7ece3264290a5ff4360d513a12c74848098 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-4226771e | sha256-316c5979742f824c1f431747548658b053cc5ce7cdd13cd64008ec11e32d4a2f |
