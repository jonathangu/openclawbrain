# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cacf2324859afd8e6f3cd4cc1393b48174ec7965442a67bc34f8b6260b72a625`
- fixture hash: `sha256-ca2cd496b9308f9d13fcff6478fd7a04f824cb026dc43bd11af171fcc1a89539`
- score hash: `sha256-d59b4ebaa8a814a412cce1086630fb91dee4f0d4bcbc37e68d6e59efa198a36e`
- bundle hash: `sha256-89a46ee8c6a40e059878faf154a8ec2caacf867d2836078b0cda39832dc6b09b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-591cbecfe0bbc6c84d3223d049bac9d2eb96d473137d7ef277a661d0bb2ceee3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6380f17c7777a78ab8ce86a0394a4b039f798ee6f4d2529ae8cf5973db2aa0c1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-28f54d05b1377d8c1a8749c9f602416b69d7bd685b91a2cde453c92efd4adf66 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-0fbffdf5807e9d3d3acc9bedc64baa44c5112b70db040f991d71aa92f6630bc9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc8c78ca | sha256-f95461b7ae475654dbc5d851ad06c99377d179662255cd43650bb5a686665eed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-cc8c78ca | sha256-8542a6e07a941378f16ac366f2c3701ec82552a2504c6383f4136c7765b38c00 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-2155adcd | sha256-c5a4ee2c50949473944e99dc09348a52ef07ad25b72892c5aa1a7eb9ea4a554e |
