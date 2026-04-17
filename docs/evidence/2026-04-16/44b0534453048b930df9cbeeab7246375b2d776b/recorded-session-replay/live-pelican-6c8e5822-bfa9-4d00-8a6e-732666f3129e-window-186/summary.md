# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-186`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e334b58e5431d3b20f7572c904faed7d64f26bc6fd3cb1bf1d055e492134e8a8`
- fixture hash: `sha256-8e788213c51f0225abe30e2600382afc50022c57de7f08753d94aa61dd287dae`
- score hash: `sha256-a1f964b757aa882a6df5d51fdbb730a885fa7cbb60884adadac368d30bba8fa0`
- bundle hash: `sha256-088c3e613e8334f63da5361b13fbec2df1118efeacb71e79c0d292605ec0f277`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f4621a0c949a3fba62d418ef21dd1d6c65fb58e546b35333db0f8e5c2c8785a |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5522b63e589bad7c289f02e7f4f66fa6b7070da7605df876b3229532f3da1bf5 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0017e85234bad6555c0ef9b3c9240874c0c7a8b93de7eaa457884a8e5616c499 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-4e53b2037966e87a8b1b12c7dc2a4c4b8a3599465aba74c608922ae301f3176c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45f202d6 | sha256-b5c5e582300b9a6eeacae36d78c1ac4f8e61cec351e9effd64d7793ded150820 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-45f202d6 | sha256-c2b6d88a279041245dd7a69112e2eaf07c87df97d7ef0d7fb71bb71798719b22 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-714b2523 | sha256-7f09f4708d5d9e23616f7ee9a6a876a70c42c91ea222b371ac30e836df9bfd8d |
