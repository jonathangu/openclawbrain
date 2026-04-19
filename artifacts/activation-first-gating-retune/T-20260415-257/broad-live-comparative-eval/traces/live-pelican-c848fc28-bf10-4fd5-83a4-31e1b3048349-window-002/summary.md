# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-c848fc28-bf10-4fd5-83a4-31e1b3048349-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-32e0b4ec2c1ecbf5a44b66dab5340f30730d05ccd8fc6dea8e459b03d93bb729`
- fixture hash: `sha256-cd231e74dab2c7ac691e39a4ea475c769c350fe4115dc674162e2af0c0f3148d`
- score hash: `sha256-0db8d8b76ee814e092b2f324d12aa2d4d34e0ecc9a9090e0c206b6f5240f5d43`
- bundle hash: `sha256-47fdfc24d90adf06476e61d0ec386ac3a55860f0a5fa239efd394fb0cbf4982f`

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
| vector_only | 1 | 1 | 0.5 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 1 | 1 |
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
| vector_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-bf1c6d84fab873c7372e9beb910ee6e8d79fd3b387f7052f30ff4d57455f1f4f |
| graph_prior_only | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 1 | sha256-9cedf8ebce5683059922b679cdb4f336f0ef212a7c4dfbaa71d1326a0887ebdf |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-81a8149212de22d7c4abb8b38ff3d899d03ff4acbc1806a413c82a15e8f4d204 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-dbba1a5e | sha256-fafc4c2be01ea043aff74307c02ea13c127a2dafd6cf8b27dddc7b49c7b9a3c7 |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | yes | no | pack-dbba1a5e | sha256-9bc4dd036e18b844275159d8e155593017a4b9f8c2f5df2324f2f14c0d0d9ecc |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-dbba1a5e | sha256-fafc4c2be01ea043aff74307c02ea13c127a2dafd6cf8b27dddc7b49c7b9a3c7 |
