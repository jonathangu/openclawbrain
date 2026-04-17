# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-469f7b7c-7551-4939-9416-5ac673c3b285-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-53e7f7c2a908bfa01e8a36f987e9389c06b6f1c4270256cec14da19431b1dd8e`
- fixture hash: `sha256-4dd26bce21297c56105a43961b6bacbe27d7812f2b72d27dc4b8b7698e0474b9`
- score hash: `sha256-071bc22e4692bef038820eedffa51ac26e83dec627564ff94c9f5cdd90717814`
- bundle hash: `sha256-517679ddfbf37e41f15cf053f6e05efb2b6283e9b4c8c52bfff30b8550a95df5`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-aeccda1f8aefa0b00a23d8464e4e2bbf0fb55e8c49bf77bf016cce252f0ffad2 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-022e86508e11b15f3aa339a8d41a88c9e95f054a22a6ddc6d504aea54080d617 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c207dabadd5f55ece399484a0c518746ec2d51d432bb4e14308f801c06c8b7f4 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-54e8a1a179449297814309a4e0e3bd3599f53f989eba418a5fb6ec075ba5385f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-76526fd6 | sha256-38c77e4154bc5e5731a0b18b2a742145e7ed2260eeb289c7efe07dddbf14418e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-76526fd6 | sha256-f6867554442b93d9285dc4ddb20b0bd8413674adecc592ad96aa3cb7691394bd |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-7b9a8ee1 | sha256-1aa65bd5226d6fdf6fe949c83e89a5ef4e0ba02b96c6e7467b7a0666776c8bba |
