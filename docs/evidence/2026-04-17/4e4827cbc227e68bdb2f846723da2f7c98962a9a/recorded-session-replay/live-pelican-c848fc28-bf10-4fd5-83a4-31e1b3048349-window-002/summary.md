# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-32e0b4ec2c1ecbf5a44b66dab5340f30730d05ccd8fc6dea8e459b03d93bb729`
- fixture hash: `sha256-cd231e74dab2c7ac691e39a4ea475c769c350fe4115dc674162e2af0c0f3148d`
- score hash: `sha256-4ac4d8a10b8b7eef67521098515b34ab8e1e39e2b9ccfbc6eff803a7f08229a1`
- bundle hash: `sha256-db4dd87b11d0901317c1417170a335760e9f0177d666d85b2c3a1335df520330`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-eafd155fdfa2fbd8e1c5739855382bb4aee55ae760f037b37bc2cd66c8f2b4cb |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-608c98241e66d5bbeb249074484ca936f904d98a61ce37b8cb9b350c07220562 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-e99109cfa3e6902ef34c96c6e4b5c22f6e89b0f4cf54c184daabc94d943cfe89 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-1f90683684fb4bf40a7b687a9161df48de206ec0571caeb5fcedf6e81693bcc3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-13ff0204 | sha256-eb3bbf8cad39962b412b6791c700923065aec9bfbbb599e9bcb88384e06fedfb |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-13ff0204 | sha256-a0c0b0c8eccce00813bd733369d02ecc5201186d0cfdb22324e511b863780993 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-1afa5aff | sha256-4619038ac3f861fdcc2b3ba446fe4e4d523206677526a0a7013595a57370b45d |
