# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15f74481afa0ad3c49942a752d93fa21610759dcd0f5184c05ee667b747607b5`
- fixture hash: `sha256-27569dfe07b6cf66e357fc072347afe0c073b0dd225ff6f7f6dbd4f6b53bd5c5`
- score hash: `sha256-1e813fe37580a0d274a7c4117a8e3c71c8c4a12d80c9eab8840aba8c5e82663c`
- bundle hash: `sha256-b2df6d8a39847c7fe4716a0df3de12d599933a89dafd9b5ada9f04ba5950eb0a`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d9780bcb02b4dddac9cfba41582ad72477a9d4e9b030a1ad3ced919c347c5d08 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b44f6c99f994cd1e07f6f0a45286314f2a39eff831824904de9b3cbb7812448b |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-998da25e40ca16d34417d5b4653be19e6a7b3e2eaad1b361d07287690fa13171 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-db8f59b3a0e9db333e01557ea94dbab898dafab30f7b9ba4e8a5f3fb5e5f8871 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cc0d3bc1 | sha256-8e2d728a040543acb63133f50fe4a40372ebec89c5b78a01d2f4a5b6033839f6 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-cc0d3bc1 | sha256-5c36cb8d69dcf603080c10dd4c9d2ff0ba2ed87a4c246ec14e71677012845547 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-cc0d3bc1 | sha256-8e2d728a040543acb63133f50fe4a40372ebec89c5b78a01d2f4a5b6033839f6 |
