# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-b7da9e48-bfdb-4562-a6ea-fae8b4f3e06a-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-adc21e40f3c3bdc2111e458183ef292b9fdba4cc9072a5e4575150e3a25e7599`
- fixture hash: `sha256-82594518eb539bcd92075469119fdd7049793972cdce0d3d047ffdabe9e539b7`
- score hash: `sha256-177234da11a0de3bc7fc654706a1f11ab891b493507adffe67679de29cc2b827`
- bundle hash: `sha256-3bc54440c0bd636c1a098688396afa6fdd6ab22b0aef527d36018eb14f29025b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ca48aec6e03fc6ebf10d02ee2af1729bb6ff692653b0f22ac3e3b10f844865d0 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a1ee68da91f54b6265ca56cd1df6a150ade8df19470c694eb6cc99805822c4a3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-79646e4946272bf09c124540b57e489542b1ca7bdc54cd50cdab191052765bff |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-44ac7e4cf399b4e3e9b2ab3aadb1ef0109db2299bbac08d41fb4c3d165f6b413 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0ebe30e3 | sha256-c1b44ff9fbbfbfc1a9f8e757fabe080b5887f06486ec77c9891cab90b5c4b930 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0ebe30e3 | sha256-c0c9c829b2a4714d0280f1c73e486a01336124a33eb33366519019b735f54f9e |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-39a686c6 | sha256-a3bfa40887457d40666b18395cae36a39159a4b8cdbad5c084a57f57368e5f33 |
