# Recorded Session Replay Proof Bundle

- trace id: `live-main-aba6eef9-56f5-4609-aed4-3859e5ad2833-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-ca1018694169cc3fe531485fc537c09a6239e84c0e4410a019dba97e2a66fe7e`
- fixture hash: `sha256-9d6b96efb0f7a7d48de55af286c816bef6a9a27fdc8a979e0eeba28c500d12da`
- score hash: `sha256-0cc06d8470ee7e60029cfc1b9b14128afb345022274b124792b6df6a63b1b8bd`
- bundle hash: `sha256-a0f8ea7809c57f5752079ae6ffe2bc65c4d1cfd336d9da9bf05fd0ffa11446f9`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2225486ce356841ccd69a322b5b86cae51f3de0b57802b050f099b2bdb0a0f2e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-948c610689824a3ed0c5f01097489c848c5a9f184a05a7668b4ac47e22141fa2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0deaa832efa2901835d29b870beb179f6c95a16472bf8a91a83e060e9da4bb8a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e04f8f76e3132476fc03d09a74155fb366138c63a1087cd46ed34a09e315db94 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b7b9dc7f | sha256-8e4630f6971d8969a13722640f74b149c2274e0d9ae2ac735f5c92413c968c6e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-b7b9dc7f | sha256-5c19b3e903f1a5884364a0c8e06bf4bc990eadef03aa06ddd0a5ec8fcae21ecb |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b77824ee | sha256-278a267c0c366945af4172b8baec50aa820f36b3d010d3784b1f610e067f9c64 |
