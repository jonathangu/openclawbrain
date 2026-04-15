# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-211`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c8f3ad7fd7e03d5e6a620d917f9922d423fcf350f433bf42dd24d49c0d04613c`
- fixture hash: `sha256-d090bc75588ff2d651484afffd5d21c674237c8a0eae19ac1a18854f75e95a21`
- score hash: `sha256-769ad5f81a6d216f86ad112c087b5ad6f3a1e0969cc743366f09a8298144ba57`
- bundle hash: `sha256-41e09e18c4a560f87936fb74f77f55fe13c727d30b9d051a4ab154d091da2cff`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a44aad7aa94fbaca9460011cc6ae9061f9cd3a6c6afa137136f8bba1929488be |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-b0883efc46d92289f8f6c289cfe4801556b3b501e8b0978da07da9542385747c |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-dca3e07cb592e1eceb351aa81d2a6bae6843ceaeaa8ddd48b708e4d697f92d94 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-2c1119725faef22358a4014fe86707171117ce154755625cd930b04779f67f88 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-0195eb68 | sha256-99eb6ef2b74613e6ef04d8a70a5d0943f4f779326a6c37052a841f41e374bdda |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-0195eb68 | sha256-8a71f1876ad55a73c17ecb242c2dae10d45739c01a75eff750ed035f00cccf2e |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-0195eb68 | sha256-99eb6ef2b74613e6ef04d8a70a5d0943f4f779326a6c37052a841f41e374bdda |
