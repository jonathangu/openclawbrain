# Recorded Session Replay Proof Bundle

- trace id: `live-main-b8b03b3e-6e68-4062-8dd5-0439897868c4-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9a2590462dc987ced28ec91e593a00f4b408387f6ec40a92d626a6087fcbd75f`
- fixture hash: `sha256-aace8a3fe4087409ebd528569ab1ac34f47ecd7317117709f7ec2907eaa6127c`
- score hash: `sha256-4e68a46ff2ca3178475ea9b1363ef7de47f18f732449a72572344d39a7732c4a`
- bundle hash: `sha256-a0f785607f8f98130169e7a63ec5aa763c46b867837a2aede0af5cc0746a7d46`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b2d2d9dd5ce486e4334796b2692780e0b5a1aabacd13eeb32d1dca3c57b5e799 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-645db52af1603b8afdface413e339b643870577626db00d179de82d20b3e245f |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-3698318910d77879fce522621f31c868af25f48d8d5b85e754e187c3e7721118 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8a88b5717bb6332c8ffc38df406642688d16b5cded58cc78e1ad1ad6d2333307 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cc48a267 | sha256-6b03d50edba34665f0275510191146e266ed7583b6c5733ee04db3f916b42e03 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cc48a267 | sha256-de2ba5ff6c6c936dc8d2384a3bbaf9f6876ea019407f3ec23bec5add1f1f3a33 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-abe062ae | sha256-9630ebbae264853748089dd926406ea2220fe8c645770113981464b6cce61cbb |
