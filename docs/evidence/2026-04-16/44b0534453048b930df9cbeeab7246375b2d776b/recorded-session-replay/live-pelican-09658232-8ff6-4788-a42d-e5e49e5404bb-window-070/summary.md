# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-070`
- winner mode: `graph_prior_only`
- trace hash: `sha256-af0623cd896f3d36aa832764b91c449eb65a56e502af4829ad2995082aa19cee`
- fixture hash: `sha256-729b2a143706d45b443dec7a409dfdba222ee805edd97aecb9fe78e30ae910a9`
- score hash: `sha256-82be90d056d0aa2381aa5ed46c834cca0784255b2b656cd6f4e9c6b99e732d2c`
- bundle hash: `sha256-56ad1c8cbadc8dda7655f8e5be15f71da69955969fca58e4f051f0978609601a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-72dc7dab3dc434226257b098b5889b33f6d9a175c84b5a7ecf9e06dde7b7bf77 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cc521dc67aa733618a33587f75afaf38b4189e4529975a2d3d2d9d5f57d128f8 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-368b058a2727899b3de661dd800615b5555a52853acd21e2afef40e77eac3fb1 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-cd3d015d2a64d66375100b76e5a3994c016024409063f5442e7b7fbf9d525a80 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9ae5f38b | sha256-a5b5c67934f7fde2966e097bf2aa15c7cc94a757cb311b8896aa05299d29c1b4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-9ae5f38b | sha256-2f9d0f66e2a377a7d1590975390097a29ef2794b91e3ff2cced1e2f064d42880 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-c54fb598 | sha256-6596b86214d21e4577c0149e54ef04f153721a3899df8ba628f43e2a6399193c |
