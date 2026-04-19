# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-048`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ec11702cd5c290669dcb9e538d05099a378cb83776c59f4106b510b78248b8f6`
- fixture hash: `sha256-c6a51fb2365b547d2ebf3546bcc8b17b6e2bc89686df2016f759916179857243`
- score hash: `sha256-0c6f39eda9d25a645f74a1aed53b781a723d1e8bb904aeeba186ce97f18fbf7a`
- bundle hash: `sha256-84e05c0696aec6a5fdf50e7ba336a7c2840e9e4842438ddf37a0c1bc92266d36`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-733cbd9351d42e7f14c5829edc01a658388a245c6f3afae8ff24b34e8acd9e00 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-1e189ba73c213a41729623acd4d392f4137ab44b1245f828b5f88e6d990d54da |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-f0c25e014f3b673ac99dd7d6edaf5ee1b3ade784bd5eeab602aeab2faf4d0c52 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-9ae885aa6d412acf43d67bd87bc5a5bfd45459e2c078032ebe53917995403010 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-3ebdd01c | sha256-448d0ea06d41362f2611a2d91b6c0da316d55cd35eef5ddbd3f2bef0e859974d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-3ebdd01c | sha256-15d4a5313b9814c4c0c4c74424a149c1d2d7045700337f3a5b460b1e12ce4aff |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-3ebdd01c | sha256-448d0ea06d41362f2611a2d91b6c0da316d55cd35eef5ddbd3f2bef0e859974d |
