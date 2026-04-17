# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-53e7f7c2a908bfa01e8a36f987e9389c06b6f1c4270256cec14da19431b1dd8e`
- fixture hash: `sha256-4dd26bce21297c56105a43961b6bacbe27d7812f2b72d27dc4b8b7698e0474b9`
- score hash: `sha256-276fad747cd934b456cd02811675057f877281a9dd92a91e5d60809a443f51cc`
- bundle hash: `sha256-ef691d7857381c98becc30e91e9c212ee2751676b8df7bf0bdb9e7ec3d4c7a76`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aeccda1f8aefa0b00a23d8464e4e2bbf0fb55e8c49bf77bf016cce252f0ffad2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fd2a4dd35e9fdc0d70877a58985f7a1193df8f3cd7f209d618e051cbf6fdfb47 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ff72d755256f6208a344711d134977a207e9e63bc05e411237cb93e43565a590 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e760df15c05ab12e75348cbc620923ee4491b40da264381aaa16e16fbdac53ce |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-780e8a29 | sha256-3ea21c3c1048cffa32cdb32329a0b60791352c2bb627146bf3754affed8be605 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-780e8a29 | sha256-469b2b71401831b5c2c61089b9822f11d7688cebf98f9b131584103b53fac33e |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d92ddd7e | sha256-45299ee743d90c19d5c99bc6a131407cc8d5f15c8c3c018dae1c608708db0f3c |
