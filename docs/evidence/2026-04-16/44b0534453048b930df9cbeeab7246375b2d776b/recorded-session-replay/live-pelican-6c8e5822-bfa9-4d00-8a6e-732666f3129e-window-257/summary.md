# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-257`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cc22db3aaa15315761f798aaec1df1acf278bfe86338b981b1d314f80e60f459`
- fixture hash: `sha256-6250124575745297903131838786e09bce6bd0b2285afd782515714f7d74a408`
- score hash: `sha256-7bc6a59d97ac6130d325cf894829982359989fe7ed1ea3dd250b2a12ce686156`
- bundle hash: `sha256-c91cd3736c9ff31bac5041450ed4e1a6e228d0e88a321738dbdf22684d110044`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-363a0ce15ddf219a167f700ba2217552de3446a99d080128478494cb795b929d |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-d6f0b34dd926dd8198156e8799754f379ed7a53ea422f51fe462ed5ef237c29c |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-8c9aa1188bb7d828899b8a37dacbdb4bd106e8827fe1b300279100af7b1c235f |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-639f4314064a340e928ee5d7a364f76e673222cfb72c1e6b0ba55eca73b05d42 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3182c760 | sha256-5473b5b99963295ffeab64d636ab4aabc17f30260787567c55d10d8e06e704cf |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-3182c760 | sha256-ca3683e8c33799317f0118078fb1828453ac474fab7c749a4acac1e2c01d834b |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-ab1cfc1f | sha256-2a6ef893ad150b11d9abcf2d6379ec216ccef85c441bc4200d7511ca23b8d399 |
