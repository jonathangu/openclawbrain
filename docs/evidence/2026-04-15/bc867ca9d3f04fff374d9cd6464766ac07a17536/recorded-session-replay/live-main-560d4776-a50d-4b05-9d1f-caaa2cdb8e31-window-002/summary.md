# Recorded Session Replay Proof Bundle

- trace id: `live-main-560d4776-a50d-4b05-9d1f-caaa2cdb8e31-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-035806e6f9bcb3753f58456b70d56dc8a01f4abf60114aeaf359384806f6c24b`
- fixture hash: `sha256-74bbbcc2ba3e23b87dadc56cde438b46daa30c3743245ccd0b40d24de1249370`
- score hash: `sha256-87dc6985488845a8e0d6c6cf71fda85be5e09f02c2648d1e7516d49d9a285c61`
- bundle hash: `sha256-5dcf937c37cb68d635b04219a3e18fb5d3cd062c6a6b60e2c526104593cb5b40`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-54ce8361598e1b4080ba115badc91e906441ece2076bc91bc1f9f28df2706034 |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-0401b4b02279f420154792919f1b25c111095312fa8980a11a8068bdb758afba |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-27027b6e67a4acb5e56554944cebb0fedb76dc781c471c286c61e820aef4368e |
| learned_route | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 2 | sha256-15ec2e08abe8f48c0e73956a3986c6440b6079895369f715cf788f2f77219109 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-2d75635d | sha256-3ca2eedee229bc2a9ab592929be8cfe29ba736666102555c81ca1da44b22e12c |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-2d75635d | sha256-54699e26447e3d38a16c7d985e79fdb39f89a3ea0cf678fb962afedbc22c5281 |
| learned_route | turn-1 | 80 | yes | 2/3 | no | no | pack-2d75635d | sha256-3ca2eedee229bc2a9ab592929be8cfe29ba736666102555c81ca1da44b22e12c |
