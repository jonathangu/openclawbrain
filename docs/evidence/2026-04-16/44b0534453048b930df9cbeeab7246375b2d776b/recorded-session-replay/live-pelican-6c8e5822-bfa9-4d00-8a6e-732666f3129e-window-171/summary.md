# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-171`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f2aad77541ac9575f5e5ca17b331150d26a5ffdab9f43024542cda1cc603e5be`
- fixture hash: `sha256-bd1f8b0e0683d35bf0b6cddabbcb17bfbeff749dd6d56a3da4fa75988fc68560`
- score hash: `sha256-548a52979e3e95515d882387120ee4679ec736dd4dc9ebdddbcc33b6bfc1aa75`
- bundle hash: `sha256-cade37094601076d1829e8ef6756f26fca4e41e23fb099690488f7808add81c3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8b6dcd51a56bbf9edfb3ea54756a6521b5761e2fe2a8b04b095719a90cd986e9 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d281ff7fd9541b0788e2d7241b43107679f1d66b09d61d36d268c5adf0abb36e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1d31718d7b0ed346488a0fdbfedf7bda2d9de0cd6be4403b654915a94b4ad80d |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-6ddae59ac6c983da401ea9e7e41a7ca1119762aa1014a81baec615cb2399955a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-097fc383 | sha256-c91687c498a32da5b72f23460db0bcb753a882694297db69c7882faa789a4800 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-097fc383 | sha256-2f1eb9e38a704f2370fe288d41151c8c61196d556690cf8aa08bbd22f1e68f85 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-b0c90360 | sha256-544dea1db730ce86ab88c1d13367ed4490c5aa61fa22520f15ff82d51b97c983 |
