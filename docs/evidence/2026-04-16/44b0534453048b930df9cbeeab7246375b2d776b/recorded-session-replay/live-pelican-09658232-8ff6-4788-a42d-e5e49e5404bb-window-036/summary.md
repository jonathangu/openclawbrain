# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-036`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d855fd526b0432f6da4ae83a914585ce6467161a22fe45c628b20919e2994b08`
- fixture hash: `sha256-a059e9b8611b556f3c483b97168ab252147668d3316414532e38d0791f5cd0c4`
- score hash: `sha256-0e77e865431b8f172c2463f27327f41358cf3d78b744bd93a3798b1d48ec8d13`
- bundle hash: `sha256-b7d4b3bcd9f2ed71d3f2779789da33675f52c32f811799df37118c0fbe753672`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46808c7f90eba103441fec044b9224d9dea48b85cde7d0c53efec734a800db3f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-08586b3d3bf5ad568e8a9fab1d9bd5d803916ffbe737067e9cb5f9aee9882856 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f19dec3440517195e4081ab848146de441f9ed59add3491885b1aebb3096f77b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0867bca430a02398281e23c897df0c7ad4fdbe6f45904906de7ebadaefa207f9 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-30e4239e | sha256-a13efcf5c51480b86bd6cf1a5df9475048dc1234146d215be7771dc1837447e6 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-30e4239e | sha256-5be60a28f154b7cb143f996df203544a08ddde8419c612858122231d2b609411 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-8ec5676f | sha256-82e9e7c4067492c02fb8719633475769f01f8c31dedb1172bba0165d1370ab4e |
