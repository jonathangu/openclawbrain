# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-163`
- winner mode: `graph_prior_only`
- trace hash: `sha256-37b9967646ced8e1a7e53e66d95e96c0d5cf9872e9f6cf5f223ff75c45212fe4`
- fixture hash: `sha256-e5562aca0bd9165edb9d4f0591f9dae6981c5299e9b8cff4453286d3a3e6c950`
- score hash: `sha256-f33ec77ce540b64b1ac542e6f775d565f0ad61be28317abdb95b6973aa656cff`
- bundle hash: `sha256-c3e58d9d6a035915ee6de59fbe92aa563e9e03a06c379bb5f295db23d140bbfb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e59e0368806e0012160cf4b2dfced7c5e08071a2c01bb62268694e031a82feac |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e94014cda62eab046587cdd7179837ae0ce8c72771d8503cffe4429742d7744a |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9e1056a211fe6c7a6ace909027d205ab994503264f0bc616e4911ee31a656942 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-669ade8f2b838257d817e09f47af52145225373ff71434f274a6c3968dee8515 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c393290a | sha256-ad5f5f406b501d21771462b415194a9cc07c5c5135a68e50ffb3246b29704c6a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c393290a | sha256-15a80fba3d8f8d590c9f804f0ea0afb6341a00b7a446f31ff5f80ba74a5820aa |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f4707cbf | sha256-4bdb04918b722009d4515ada504a2befcd96d374a893c28e26b350d520a62761 |
