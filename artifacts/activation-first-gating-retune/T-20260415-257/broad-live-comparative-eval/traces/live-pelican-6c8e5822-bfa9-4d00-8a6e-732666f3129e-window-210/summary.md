# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-210`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9de933020cc2b0d03a4b1b4f5bf51c7bca0c4ae8de78af7a2cf6d4b86ac284d4`
- fixture hash: `sha256-2e5835aa933cf2df6faf2714837c2953d1866a5094413604d0ec3e648b5257c4`
- score hash: `sha256-15c007e352a0b34eed2b13948a6607aa5d7b0282823c5a96c0cb7c11d597e41b`
- bundle hash: `sha256-56a505c349879dad66534e20f6198fc9f9ab795e50d5d70ef26cd1d134027d82`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-13242c9a12ffb4d788c2f14891b978d17c5b819a44b8fb4dd405e1c1b50322e8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5811090755d888a5674c22c4a4c1b65461e8e83e4e72a363d53d0e9c5490a4a8 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4df163367ab98307f66cf5046f48d1b32a8c4d769c176cf45deb404c98b48fdf |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-aa056c040ed612db408af44cdfe7c9ac156aff57035b148591404ec61b6f97ad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2f3f69bf | sha256-87eb1d3be4b702faeb9a9fd4bdfda681069e578ad63f15f47d28ddb6423c4213 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2f3f69bf | sha256-f785233986e90dd1bc99fcaabb0c2085885f1edc58504aac30da927bf57851e4 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e4745af2 | sha256-cb721eeb7e95b6629e3d9b91975c31a00fb098c0f18e81949571bcdd324887f1 |
