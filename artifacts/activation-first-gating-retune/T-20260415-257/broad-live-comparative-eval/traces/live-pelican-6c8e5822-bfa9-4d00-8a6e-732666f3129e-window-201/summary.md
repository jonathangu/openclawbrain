# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-201`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0daac77494acc6aca056ecf3e9f12fd58f33e4988f5a646f0ba5dd6ef080149`
- fixture hash: `sha256-267dbae4de2075656207ffdf48cdf822d6c7cd1996c42f8535ce786c53f3660f`
- score hash: `sha256-6eff3105f00f3ef1eb0620ed1a5fc2123b36b330b63fdda4bee2353e75556023`
- bundle hash: `sha256-370f640a58a1edd6e075a28a5090fc9a1652663482d111719c8ce799024d742b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7fed91035473e3fc2f9947814563b07b47451ce7ca5cc7e497e3f1c68f58c389 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a72ef44345e8fe241f6b3ad9041838996b254562bd8655d23152e0a3becb40f |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2b09026e70feb94ba73a81e6ef546ea1c4b5f1e4457e1c5147a596355219ec8f |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-95aac63dee61830d784fe25013ac289e136d409a6c7744b2e0a26850b902456e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0e9f708f | sha256-6dc60d1edcdd7e840b157b5bec0ef9416b096c624fa410335a77aef39545965e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0e9f708f | sha256-3af42ad72c92b2b759cd31ad2155cffeb46d519a4e11e143e613d6eefe765d6d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-917c1b40 | sha256-e776a529f34dd86df2e21ff729a00370b8d81560c733bed7bab66108a6b116ab |
