# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d479c91d117044f49ae49da499694fdbe9a9bce3b101e2f906d0092b46536940`
- fixture hash: `sha256-afb53fed27fe0fd6a6ad4e067cb4e140573e8cbd954bfddd658b0c3c6c424a0e`
- score hash: `sha256-54076c0db7c9ef8d068cb32c182adc5b17b680b4f0f17b1d69a8dfe8fdd0d9b2`
- bundle hash: `sha256-39904e3fb5c30064c81e6be4d9441416de9e21a11c9dd3801fce7919123f962b`

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
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2f4b660a3e5d5f4a920994b92d0eae72726c74b613f3139fcacbac22692626d |
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-a112baed97acba2af39bed6224d86bfcc0a161335d6fbafa4a66d51f3319435f |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-3acfe0d688cbe556154b63f44dc7f2e3214e76fe97e11cc6bb16c412751491f2 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-a6f7776a302bfe252acfe2c5936a6ebba9491502feea4a0cf384414c0c05c289 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-3410cd41 | sha256-475e4d1b8768980e21706d9134851b69145479e8c325b3c2cfbaf5e874c0f37e |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-3410cd41 | sha256-4a33c124e6960ceff38f9931da3011269f87b7d0e276e0bba1c596705f2832d6 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-79bba284 | sha256-23a598bb5fd3b5e9648daac9438ace4553780b846badb0f3539324e0262067b0 |
