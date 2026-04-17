# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-021`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cacf2324859afd8e6f3cd4cc1393b48174ec7965442a67bc34f8b6260b72a625`
- fixture hash: `sha256-ca2cd496b9308f9d13fcff6478fd7a04f824cb026dc43bd11af171fcc1a89539`
- score hash: `sha256-155164053edc59256e34a61bfbb63c2e788c755f223bb254be67beea718986e9`
- bundle hash: `sha256-b557afbdb731aafc98423af04d4c18d0444b8f5dab62e2bebbb91afdba69e5bf`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-591cbecfe0bbc6c84d3223d049bac9d2eb96d473137d7ef277a661d0bb2ceee3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c8f40b0dce926fe6a7de86db2e3831dd3682dca7b6558d7d1c1cc6deece4b843 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-10524020f8739318b446a0ac73cda875f5d993c9254e4188a5603d0199ac7b1e |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-8b86587c6100a122caaf274c74f48f67ef1a0651a6f395b1a96bf1237dbe834b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-26b9e58c | sha256-3d09560e8c522bdc940d932311bb1040ee90367fdc3bdda1b32842440eb1b9f2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-26b9e58c | sha256-70fa7248858e7d8723bbc770f3965ea6cf9ad6bf77f3a78708c8195569738036 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7b831a8f | sha256-9380854fd82f70f502470aa48ea7f45a9eb3afa9ff98be77e75d6ca8fb48b15a |
