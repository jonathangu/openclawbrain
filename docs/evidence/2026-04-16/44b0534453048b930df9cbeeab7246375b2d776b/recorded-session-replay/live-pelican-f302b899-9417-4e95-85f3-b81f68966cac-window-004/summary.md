# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-08b73891909d4362ec26f6fa9db500532bf1bc8c805c846177530e06134e3890`
- fixture hash: `sha256-8b9b0ca98fc7faf65751139ae1faf124a5228fc02a0f5bb6427265ff145c7a87`
- score hash: `sha256-dcb2f24b046979d938165252e208c8dbbbf1db75d13403890c0cec9e74b65764`
- bundle hash: `sha256-637addd6e43829a253bf759d805cc99ec8111a62b6981ec040273ad1b03e98ac`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f871e8b1f54b24b9e075d5f4db6f8b41f6cb53e929f6d747d42ccbb2426d8d7 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e364ade76da2027f9ee91f2c203fa4aa6ac0c289b544aea4c3515b3b94798a8a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38f45aefe4b50cf1bbf2bd5b57dbf0cc27bb46a259baa842b5f04520ac648abe |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-bee8182d5d183ae0ca1e03a9107302710fe72ab45eca0ec76469d87b09b72e76 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ad2ef173 | sha256-796340d50aef4312e2b725ee836c965b3b736e00b910e6d5867970f1b9c51434 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-ad2ef173 | sha256-a7ec5f4280d5c6a33cc3b841039098bc888e9814da58f5bbf1e9ca5d7d3bd6dd |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2b31fe72 | sha256-04a779dd3c0312a60f836b024dbdf1ce8b28ecff3c13e24bec7d2225bad4170f |
