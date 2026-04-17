# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d479c91d117044f49ae49da499694fdbe9a9bce3b101e2f906d0092b46536940`
- fixture hash: `sha256-afb53fed27fe0fd6a6ad4e067cb4e140573e8cbd954bfddd658b0c3c6c424a0e`
- score hash: `sha256-7ddcc9e7ed15841aa8383663ae6a853f0fab6ddde4bfb6f7732392dcababb690`
- bundle hash: `sha256-97bf12739c33ef734dd78e51111efb19077275019396d602bc535b5e12367727`

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
| vector_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-bd21ddc360edf00a2f7a3e4d11cc83435366ad3f5c211aa1a5492b3fa3c56f9f |
| graph_prior_only | 1 | 1 | 2/3 | 0 | 0 | 1 | 0 | 1 | sha256-d9158e7b3c8e3c25868cdc420c5337330c1ffd530d28ee83767a769c32687897 |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-63e054bfcdcaae9e79a1f97017c8926b413e915a4ccb5f223fb94e63d8ee48ac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | no | no | pack-5c297f58 | sha256-21133bb26c80138218277975e88e73e915da890d083c9f55f4f17e5c56ab33e8 |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | no | no | pack-5c297f58 | sha256-0b58c51d911e5856f25e099c480af06980c91d4f33d0689c320bb7e2f5286c0b |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-a1d4549b | sha256-d2f35914d7bbaa067d37e49fed3ce09e1ce5e49183aa9569de641a9d04b16f06 |
