# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002`
- winner mode: `learned_route`
- trace hash: `sha256-ca1018694169cc3fe531485fc537c09a6239e84c0e4410a019dba97e2a66fe7e`
- fixture hash: `sha256-9d6b96efb0f7a7d48de55af286c816bef6a9a27fdc8a979e0eeba28c500d12da`
- score hash: `sha256-0b4cd9fda7691ca3091af8216ed387a8f7043415fa9fcf0dad3c26cad0aabadb`
- bundle hash: `sha256-ee165ccc126dbfe0a174ce7fe7cb3e2bc10c74915c5e00013a60ab297f2c2ae1`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 80 |
| 2 | vector_only | 80 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 4/12
- phrase hit rate: 0.333333

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.666667 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2225486ce356841ccd69a322b5b86cae51f3de0b57802b050f099b2bdb0a0f2e |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-a93d03a2876c002ed75c9e4dd8850dd5ba02426da8df6093a4592bb4c6fbd401 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-39a15775534d99eac43250f69c98a9cf9f4997a969372b538f3196198615462b |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-323ade407e5a45c578a8450e23d6bcb3def010907cea933eb1eb303ed9a0e021 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-a76e3841 | sha256-7206b1585b964495ace0df285e91bda34a91186a3fcdf56f530b1ddd3cefcc11 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-a76e3841 | sha256-c811ca6ef518704fbf7ce97a0c0b86e7b80704ab8a5607431122fba73d31e95a |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-a76e3841 | sha256-7206b1585b964495ace0df285e91bda34a91186a3fcdf56f530b1ddd3cefcc11 |
