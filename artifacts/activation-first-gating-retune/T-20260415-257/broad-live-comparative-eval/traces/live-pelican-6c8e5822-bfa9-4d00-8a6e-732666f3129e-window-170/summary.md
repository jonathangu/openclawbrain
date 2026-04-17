# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-170`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b236d1de0cb5cdb5c6745d8cf8165eea05f61fe4f12fe91030959e1a1f0a9ef6`
- fixture hash: `sha256-2918c2c3bf776980cb54310652408dcd4b80904c74dc802c02149421011a5050`
- score hash: `sha256-05fa8b1e6d3ddc3447865a05ce871d592a96237ee44eeea191b8999f40a23e69`
- bundle hash: `sha256-3439eba50f14d246cce953cca52d0c1c49f0478c6e4a13bb73e08d5f811cef49`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b752caeb990c3acabff60b3183401c5659a9fe06fb13d30bacaaff23a3d4f453 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8c1fbb450b40cefc393f4fbe1dc26fc17f5ecba51255862edb511387570b83cc |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30d889fa078d94a364d6f3661923a5239d63fd9d269a1d2c8e597fa6423db70b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d70956f77ecaa2bf3ae5c4ecc8b793d956ee68780ffa04db31a66b53439b7d89 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a05b069f | sha256-a22e36631fde6e27d4a0b28c6d4f23c1da75bd7ba8d87a059a13b63c3ab27519 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a05b069f | sha256-a3511e737effe51596d54d8be543b0229127c877975981189f006348c13eb8b5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-93114724 | sha256-19007f9b25ab4a99d160171e43ba1dae6413e7c57ccb87b9bc9afbe1a45e3e14 |
