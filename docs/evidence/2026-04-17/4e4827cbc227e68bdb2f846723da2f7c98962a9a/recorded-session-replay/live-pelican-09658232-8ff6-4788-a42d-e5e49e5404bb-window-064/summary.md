# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-064`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f39bc7349702eda18b0d056342226b1aabb41caee42927ea480ba26a62daf2f`
- fixture hash: `sha256-841eefc5eecc02fe972ed7cac8e3716da5b289fe7edcf8c461503d651db37931`
- score hash: `sha256-16d13418343926179bc9ce167ccfb3b189aaaf0f4897fd3c4c293ba58547e10c`
- bundle hash: `sha256-c34f413f41525d81df9aeb31067e8c18a73bf9e18360edf5123f5102615701bf`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f5559e04ec3c75dd16ee057dcaef2391dd2363ce8cb9ccfcfa727aea97487dcd |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9efeda857c2d635a28f52ef40ad02ca0d9f6a2673ef12f396e3468d0a29b25fc |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-57aff823a4c2cba2464a99f527aec3a3b63ae574ec9ad70d53609611c0ad9827 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ea410e9c5999db421e9b3f0f4171baf51acb0a67d939ff3573dd721bdb5e37bf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3f4fa400 | sha256-a9bfa0711340a9daa4732bc93ee7b769e2fd2b0bb93567d55c4f9b1530e9ffb4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-3f4fa400 | sha256-16148e1fc32b3a9c346b068763f02057abb55d88e3e35d427704233869ff4188 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-d8526bbb | sha256-beb35b6881945f571c323f5dac868f2e0162ab69e603ba26e69361afaffbd166 |
