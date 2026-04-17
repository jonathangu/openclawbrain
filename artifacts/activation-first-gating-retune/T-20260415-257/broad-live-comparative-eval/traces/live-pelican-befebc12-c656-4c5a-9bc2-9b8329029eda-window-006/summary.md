# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-006`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7f9684c38d91e55a983d42052df21e03bec407bc3f34393946fcda8e1b2d39f4`
- fixture hash: `sha256-5b6e1bbde60f4bcca2052f19249d943d07695521da1e7e8b46846e97b143bb5b`
- score hash: `sha256-aacaa1338b182befeb39519e1630c9e84e6f9888700c852aab3dec1921fffbd6`
- bundle hash: `sha256-1e5f61a6c3222f39f90b6f002f2765e5a7275d9c53b30ee1a741e7f1a94d569a`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-74a0ec494cc9b5ec66bff70ca0bc3e9262d5754f8a93ce7d222367da206ee232 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-657819b4a4e19d89bf4b0cb1450b647a6999cc863d6e8e2847c8ff8102c0b2a5 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-74a12386b3363ec1b052b89777203d4c8338764c608d188e3141fdfce082e06a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-0ac4a3e70cb6f6006ff9cfa37225de884d2189b3a5a4d8245a2561f9d7b63f1f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6f9c3024 | sha256-50cd5277296a511bd1990613c3c630d11d0caca30e8f0669071f8cf9623e321d |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-6f9c3024 | sha256-8ca493af5e3b067af46e5dbba4ccf85dfcadaacb8562da3a61b583850564e02d |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-b42e27ef | sha256-e5af97a61888b5e4e77a23be5db819d106f0bd10c1064f78a714e478b4c67027 |
