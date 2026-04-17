# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-609a8ef08a0005ac8c6d28613bff0743081fda2af2229951c1bce5c2a71dd05c`
- fixture hash: `sha256-6a13457eafa6a8dea8911b77d2fb44eb3c714588ecda4ba2d46120f25504eae3`
- score hash: `sha256-299a75189dc80e3502d6036e5e5a74fa440c407dc022b9c8b1b2f1a7271b8b6a`
- bundle hash: `sha256-7b039d4ca9a13d0c04f37fa6f9cad387cf45d9756c22be8bb702722b32c21d9f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/4
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-529cbc550c979df64d974591190e0c1d456cbd8f7265be9e27a0fc5cbc417683 |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-f5f7e3d64f617ef303a75d6aecdfbae66ff297555909d5562bad0832b2ba5d65 |
| graph_prior_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-b987161cced61a6783782d12269aa4d2007d13dfa06a1533e6319a16e50da7d9 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-238afdb9ee32c267657bdecec2486cafe08d88fb3c08d122f7b660f004e395db |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-775f34bc | sha256-c9684b04649c6b1fc2cf889c6f47f4901f9fa1ef74b91bbf4d9a5a4267f6407c |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-775f34bc | sha256-9585912da933f52fef60cac029e13eb176649da536044258396cbcd2e4b6255d |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-d0ffc7dd | sha256-685e15e1619e81fd9318c1fa04b8e3a09a933a11c40405fffd436e40b3943257 |
