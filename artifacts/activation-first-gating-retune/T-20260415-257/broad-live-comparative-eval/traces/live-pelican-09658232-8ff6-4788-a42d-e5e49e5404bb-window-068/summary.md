# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-068`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d5b2e9dc9e67decfaf3c661978d40b3965c717607588db6b26b950194e4e66bc`
- fixture hash: `sha256-05d4de9ab3e3c70047bcf0e08acaa0f5e5762d96334a591c78e4a27669a8787c`
- score hash: `sha256-5ae16460a8753c7c515ae73350a5b910638b1cba19ed37736522ee4fb5b8caeb`
- bundle hash: `sha256-84bd04b76bbdbd24637fe2aeda764ce3eb95be06fdd4ad1eea7d31947a309c14`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-044b12081781ee7b9e9814feab1eb91fdf156b393d98255c7373c2abeeff9d8d |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e10fcb5a2bb01f97138513c1f9250f2b126d8d8bc4b72f0bdb552ad853fea9d7 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1c040261d1b6bbc9e626dc514b4f5f5ed43a56cb2c9600470f23efe43964f975 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e86e8402a7703905858998e1c11138704e9b71a1cabd731d922e1f8be79c48c1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-232f276c | sha256-f649b2af2f6d3f268218f41d33942defac185d36b92ea26f88b37af160813593 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-232f276c | sha256-f649b2af2f6d3f268218f41d33942defac185d36b92ea26f88b37af160813593 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b61534ab | sha256-1746f8c05fae9979b2c7826ac3e63e4860ec6b15c4ea1f422bc8f5fe6a03ca62 |
