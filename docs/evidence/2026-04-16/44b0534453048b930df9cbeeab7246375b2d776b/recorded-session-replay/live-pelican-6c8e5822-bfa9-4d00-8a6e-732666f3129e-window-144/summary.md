# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-144`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0149e3caf93b3375cf02c24f74af73ff26b7bc10ea672fea0331d56ac334a82f`
- fixture hash: `sha256-8eef5aa851168050667187c6a1f16965243d4107da455697233fb94b6cd8be15`
- score hash: `sha256-efe559c52f95b9bd668ccb48f4b815d6c037cb51eae66c1a850d6cd89c8f4716`
- bundle hash: `sha256-cb096af8d1f1bfeffc2f7389a24aa01d8e41ab6e93a7c8d267637fd2770ca522`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-beac0c48f82ed7e8a11f136719a9c12038db11daf2070f49f0ee8d4c618e927a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-fa1d34a370800728e243715e547afda58bdf048d351164f728e72f1961f8de4e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-75e008d0003fcb4afab5ef9bcc190f92dd2376ed08634f886fe7c66ab408e8af |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-b9780f83ee7dc30f3c759157967a4731b593f53c39ae129c655456ec331f35ba |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9df50c16 | sha256-b3a25a8f5b8e2916bf6042df67c70cd3bad4a270709e2a4e640062e1be0a91b7 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9df50c16 | sha256-c6d313500c9cc7f08d7acc568a4e66f601dd4f0de9284e5c205015de1ef3f68d |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-d8176fa3 | sha256-c70d681ac5cc0ead4aa59f3495b80f118d811abf6e33560fa8de795a928d544d |
