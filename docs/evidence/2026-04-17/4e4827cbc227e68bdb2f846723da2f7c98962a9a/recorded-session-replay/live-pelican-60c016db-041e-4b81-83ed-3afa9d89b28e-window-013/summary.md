# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-013`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7e2a057c58ceb7779d689dcd4238dfbc3207e352fc341de03ac7a06d504301da`
- fixture hash: `sha256-737f6561e785d3bc05d3981f983d5cf16785ca63d2f46199fbc1baaeee1f2b69`
- score hash: `sha256-fc7212e6d9126bbdcc1f47fc612617e37f46ec389b9f20bb4dc2a5a39c4b1fc9`
- bundle hash: `sha256-96a3b8653249f1f248b33e78e55f33f665d706b2361db3882c2829da52711644`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1e7a79c157dc055e3ad83a213c22e42badb5ac82b3ed30aa50ada887959b805f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4570729663cbf102811f22158aa27b60670d9f5a6b0082f3ef412dbd5a462e0c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c8a07e97f70b0d87903969e01d2f510cd9a960262abdd9dac11fd9b8dff1bff8 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-84021f5e0dffffbb46921905d1c757bc7ab4ea660e5f0fde133356a9ad049dbe |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-08bdfe95 | sha256-9068680db65dfb52b46aa607cf27f9d5e2c77db874d94a1c5b8b4bda3250702d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-08bdfe95 | sha256-03920d3216dc90e972d1fdf9ceada72c812acd6438d6d42e09151a10f0eda477 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-2a2c2e68 | sha256-19dc575e58d76ffcfaf590160a8aa920a3dc74a07115e1e8d3f6d0cefc0a04be |
