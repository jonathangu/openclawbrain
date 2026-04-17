# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-98f498b917834ee9c0a78d5b62a338d5c94ab2df87cb501ae8615cf42d07619a`
- fixture hash: `sha256-54eb8df766feda2c6211a7171b884e66a0008ed710f7d28bcb6341bc861e92a9`
- score hash: `sha256-a2dc6c22213ed7c32ac902f786726d3577ebbf36ffa6e554ef26f53da32f76a0`
- bundle hash: `sha256-75932bba4a75d1bd38e35ede34bdf6f24df7ffaf3195811a66f7b8ef46ec4813`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e6670af2aca1f1e71cbf3c0f145ce7f96dddb89bb0330719aa7609642a8108f9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1b0955c1ae138ff7ebc913596a93171c683e8420519e578692d77aae42cf2190 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dbef3e91fec235f5d6db6602b2b6c3839b3c20a0b7a351df0002b7443ba0a84d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8e67d66d6b895260a7f7d82072a6d0d17a680070443293bb42a9ce40b8c4f004 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a9363df4 | sha256-5c89eb1847accdf1b32e84738bb4d3672fd0b5156dcef596b6d525711f7b5e59 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a9363df4 | sha256-74799b90e0d9cf9e75d2879553c60b4af7d404011cfb8fa18e38aa2591512e4e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6750cda9 | sha256-d6aecddf424515005eca4b7454d7542d9c6b73954368fd0835fdcac0247989b8 |
